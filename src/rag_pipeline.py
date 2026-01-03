"""
RAG Pipeline module for MD&A Generator
Orchestrates retrieval from vector store for LLM context
"""

from typing import List, Dict, Any, Optional
from .vector_store import VectorStore
from .schemas import DocumentChunk, Citation
from .config import settings


class RAGPipeline:
    """Orchestrate RAG retrieval for MD&A generation"""
    
    # Section-specific query templates
    SECTION_QUERIES = {
        'executive_summary': [
            "What are the key financial highlights and overall performance?",
            "Revenue growth and profitability trends",
            "Major financial metrics and KPIs"
        ],
        'revenue_trends': [
            "Revenue growth trends and drivers",
            "Year-over-year and quarter-over-quarter revenue changes",
            "Sales performance and market trends"
        ],
        'profitability_analysis': [
            "Profit margins and profitability metrics",
            "Gross margin, operating margin, net profit margin",
            "Cost analysis and expense trends"
        ],
        'financial_position': [
            "Balance sheet highlights and financial health",
            "Assets, liabilities, and equity position",
            "Cash position and liquidity",
            "Debt levels and capital structure"
        ],
        'risk_factors': [
            "Financial risks and concerns",
            "Declining metrics and negative trends",
            "Debt and liability concerns"
        ]
    }
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
    
    def retrieve_for_section(
        self, 
        section: str, 
        company: Optional[str] = None,
        n_results: int = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a specific MD&A section"""
        queries = self.SECTION_QUERIES.get(section, ["General financial information"])
        n_results = n_results or settings.top_k_results
        
        all_results = []
        seen_ids = set()
        
        for query in queries:
            results = self.vector_store.query(
                query_text=query,
                n_results=n_results,
                filter_company=company
            )
            
            for result in results:
                if result['chunk_id'] not in seen_ids:
                    seen_ids.add(result['chunk_id'])
                    all_results.append(result)
        
        # Sort by relevance and return top results
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return all_results[:n_results * 2]  # Return more for better context
    
    def build_context(
        self, 
        section: str, 
        company: Optional[str] = None
    ) -> tuple[str, List[Citation]]:
        """Build context string for LLM from retrieved chunks"""
        results = self.retrieve_for_section(section, company)
        
        if not results:
            return "", []
        
        context_parts = []
        citations = []
        
        for i, result in enumerate(results):
            context_parts.append(f"[Source {i+1}]")
            context_parts.append(result['content'])
            context_parts.append("")
            
            citations.append(Citation(
                chunk_id=result['chunk_id'],
                source_text=result['content'][:150],
                relevance_score=result.get('relevance_score', 0.0)
            ))
        
        context = "\n".join(context_parts)
        return context, citations
    
    def retrieve_all_context(self, company: Optional[str] = None) -> Dict[str, tuple[str, List[Citation]]]:
        """Retrieve context for all MD&A sections"""
        contexts = {}
        
        for section in self.SECTION_QUERIES.keys():
            context, citations = self.build_context(section, company)
            contexts[section] = (context, citations)
        
        return contexts
    
    def get_full_context(self, company: Optional[str] = None) -> str:
        """Get complete context from all available chunks"""
        all_results = []
        seen_ids = set()
        
        # Query for general financial data
        queries = [
            "Complete financial statements and performance",
            "Revenue, income, and profitability",
            "Assets, liabilities, and equity",
            "Growth trends and year-over-year changes"
        ]
        
        for query in queries:
            results = self.vector_store.query(
                query_text=query,
                n_results=10,
                filter_company=company
            )
            
            for result in results:
                if result['chunk_id'] not in seen_ids:
                    seen_ids.add(result['chunk_id'])
                    all_results.append(result)
        
        # Build comprehensive context
        context_parts = ["FINANCIAL DATA CONTEXT:"]
        for result in all_results:
            context_parts.append(f"\n{result['content']}")
        
        return "\n".join(context_parts)
    
    def index_data(self, chunks: List[DocumentChunk]) -> int:
        """Add chunks to the vector store"""
        return self.vector_store.add_chunks(chunks)
    
    def clear_index(self) -> None:
        """Clear the vector store index"""
        self.vector_store.clear_collection()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.vector_store.get_stats()
