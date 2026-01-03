"""
Vector Store module for MD&A Generator
Manages ChromaDB for storing and retrieving document embeddings
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from pathlib import Path
import warnings
import time

# Suppress warnings from google.generativeai about package deprecation
warnings.filterwarnings("ignore", category=FutureWarning)

from .schemas import DocumentChunk, Citation
from .config import settings


class VectorStore:
    """Manage ChromaDB vector store for financial document chunks"""
    
    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.collection_name
        
        # Ensure persist directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize Gemini for embeddings
        self._init_embeddings()
    
    def _init_embeddings(self) -> None:
        """Initialize embeddings based on provider"""
        self.provider = settings.embedding_provider
        self.api_keys = []
        
        if self.provider == "huggingface":
            # Local Embeddings (No Rate Limits!)
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading local embedding model: {settings.huggingface_model}...")
                self.hf_model = SentenceTransformer(settings.huggingface_model)
                self.use_gemini = False # Use HF instead
                print("Local embedding model loaded successfully.")
            except ImportError:
                print("Error: sentence-transformers not installed. Please run: pip install sentence-transformers")
                self.embedding_model = None
                self.use_gemini = False
            except Exception as e:
                print(f"Error loading HuggingFace model: {e}")
                self.embedding_model = None
                
        elif self.provider == "gemini":
            # Gemini API Embeddings
            if settings.gemini_api_key:
                self.api_keys.append(settings.gemini_api_key)
            if settings.gemini_api_key_2:
                self.api_keys.append(settings.gemini_api_key_2)
                
            if self.api_keys:
                self.current_key_idx = 0
                import google.generativeai as genai
                genai.configure(api_key=self.api_keys[0])
                self.embedding_model = settings.embedding_model
                self.use_gemini = True
                print(f"Initialized Gemini with {len(self.api_keys)} API keys.")
            else:
                self.embedding_model = None
                self.use_gemini = False
                print("Warning: Gemini API key not configured.")
        else:
             print(f"Unknown provider {self.provider}. Please configure correctly.")
             self.embedding_model = None
             self.use_gemini = False
    
    def _rotate_key(self) -> bool:
        """Rotate to next available API key. Returns True if rotated, False if no more keys."""
        if len(self.api_keys) <= 1:
            return False
            
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        new_key = self.api_keys[self.current_key_idx]
        genai.configure(api_key=new_key)
        print(f"  Rotating to API Key #{self.current_key_idx + 1} due to rate limit...")
        return True

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.provider == "huggingface" and hasattr(self, 'hf_model'):
            # Local HF generation
            embedding = self.hf_model.encode(text)
            return embedding.tolist()
            
        # Use the batch embedding function for a single text
        embeddings = self._generate_embeddings([text], task_type="retrieval_document")
        return embeddings[0] if embeddings else []
    
    def _get_query_embedding(self, text: str) -> List[float]:
        """Generate embedding for query"""
        if self.provider == "huggingface" and hasattr(self, 'hf_model'):
            # Local HF generation
            embedding = self.hf_model.encode(text)
            return embedding.tolist()

        # Use the batch embedding function for a single text
        embeddings = self._generate_embeddings([text], task_type="retrieval_query")
        return embeddings[0] if embeddings else []
    
    def _generate_embeddings(self, text_list: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        """Generate embeddings using configured provider"""
        if self.provider == "huggingface" and hasattr(self, 'hf_model'):
             embeddings = self.hf_model.encode(text_list)
             return embeddings.tolist()
             
        if not self.use_gemini:
            raise ValueError("Gemini API not configured and Local Embeddings not selected.")
            
        import google.generativeai as genai
        import time
        from google.api_core import exceptions
        
        embeddings = []
        batch_size = 1  # Reduced to 1 to be very safe with free tier limits
        
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i + batch_size]
            retries = 3
            
            for attempt in range(retries):
                try:
                    result = genai.embed_content(
                        model=self.embedding_model,
                        content=batch,
                        task_type=task_type,
                        title="Financial Statement" if task_type == "retrieval_document" else None
                    )
                    
                    if hasattr(result, 'embedding'):
                         embeddings.extend(result.embedding)
                    elif isinstance(result, dict) and 'embedding' in result:
                         embeddings.extend(result['embedding'])
                    else:
                        # Fallback for some response formats dictionary
                         embeddings.extend([e['embedding'] for e in result['embeddings']]) # type: ignore
                    
                    # Rate limit pause - conservative
                    time.sleep(2) 
                    break
                    
                except Exception as e:
                    is_rate_limit = "429" in str(e) or "Quota" in str(e) or "ResourceExhausted" in str(e)
                    
                    if is_rate_limit:
                        # Try rotating key first
                        if self._rotate_key():
                            print("  Retrying immediately with new key...")
                            time.sleep(1)
                            continue # Retry loop with new key
                        
                        # If rotation failed (only 1 key) or already rotated, use backoff
                        wait_time = (2 ** attempt * 5) + 5
                        print(f"Rate limit hit. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"Error generating embeddings: {e}")
                        break
            else: # If all retries fail
                print(f"Failed to generate embeddings for batch. Please check API quota.")
                raise Exception("Failed to generate embeddings via Gemini API after retries.")
        
        return embeddings
    

    
    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Add document chunks to the vector store"""
        if not chunks:
            return 0
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            embeddings.append(self._get_embedding(chunk.to_embedding_text()))
            documents.append(chunk.content)
            metadatas.append({
                "company": chunk.company,
                "period": chunk.period,
                "section": chunk.section,
                **{k: str(v) for k, v in chunk.metadata.items()}
            })
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def query(
        self, 
        query_text: str, 
        n_results: int = None,
        filter_company: Optional[str] = None,
        filter_section: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query the vector store for relevant chunks"""
        n_results = n_results or settings.top_k_results
        
        # Build where filter
        where_filter = None
        if filter_company or filter_section:
            conditions = []
            if filter_company:
                conditions.append({"company": filter_company})
            if filter_section:
                conditions.append({"section": filter_section})
            
            if len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = {"$and": conditions}
        
        # Get query embedding
        query_embedding = self._get_query_embedding(query_text)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results and results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'chunk_id': chunk_id,
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results.get('distances') else 0,
                    'relevance_score': 1 - results['distances'][0][i] if results.get('distances') else 1
                })
        
        return formatted_results
    
    def get_citations(self, query_text: str, n_results: int = 3) -> List[Citation]:
        """Get citations for a query"""
        results = self.query(query_text, n_results=n_results)
        
        citations = []
        for result in results:
            citations.append(Citation(
                chunk_id=result['chunk_id'],
                source_text=result['content'][:200],  # Truncate for display
                relevance_score=result.get('relevance_score', 0.0)
            ))
        
        return citations
    
    def clear_collection(self) -> None:
        """Clear all data from the collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            'collection_name': self.collection_name,
            'count': self.collection.count(),
            'persist_dir': self.persist_dir
        }
