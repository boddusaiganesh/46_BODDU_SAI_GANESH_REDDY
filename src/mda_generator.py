"""
MD&A Generator module
Uses Gemini LLM with RAG context to generate Management Discussion & Analysis sections
"""

import warnings
# Suppress FutureWarning from google.generativeai about package deprecation
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai
from typing import List, Dict, Any, Optional
from datetime import date

from .schemas import (
    MDADocument, MDASection, Citation, GenerationRequest,
    KPIResult, TrendDirection
)
from .rag_pipeline import RAGPipeline
from .config import settings


class MDAGenerator:
    """Generate MD&A documents using Gemini LLM with RAG retrieval"""
    
    # Prompt templates for each section
    PROMPTS = {
        'executive_summary': """You are a financial analyst writing the Executive Summary section of an MD&A report.
Based on the following financial data, write a concise executive summary highlighting:
- Overall company performance in the period
- Key revenue and profitability highlights
- Significant changes from prior periods
- Strategic financial position

FINANCIAL DATA:
{context}

CALCULATED KPIs:
{kpis}

Write a professional 2-3 paragraph executive summary. Use specific numbers from the data provided.
Reference the source data in your analysis. Be factual and cite specific metrics.""",

        'revenue_trends': """You are a financial analyst writing the Revenue Trends section of an MD&A report.
Based on the following financial data, analyze revenue trends:
- Revenue growth (YoY and QoQ)
- Key revenue drivers
- Comparison to prior periods
- Revenue trajectory outlook

FINANCIAL DATA:
{context}

Write a professional analysis of revenue trends. Use specific numbers and percentages.
Highlight significant changes and their potential drivers.""",

        'profitability_analysis': """You are a financial analyst writing the Profitability Analysis section of an MD&A report.
Based on the following financial data, analyze profitability:
- Gross margin trends
- Operating margin analysis
- Net profit margin
- Cost structure changes

FINANCIAL DATA:
{context}

Write a professional analysis of profitability. Use specific margin percentages and explain changes.
Discuss factors affecting profitability.""",

        'financial_position': """You are a financial analyst writing the Financial Position section of an MD&A report.
Based on the following financial data, analyze the company's financial health:
- Asset composition and changes
- Liability and debt levels
- Stockholders' equity
- Liquidity position (cash and current ratio)
- Capital structure

FINANCIAL DATA:
{context}

Write a professional analysis of financial position. Use specific figures from the balance sheet.
Comment on the company's financial strength and any concerns.""",

        'risk_factors': """You are a financial analyst writing the Risk Factors section of an MD&A report.
Based on the following financial data, identify and discuss potential risks:
- Declining metrics or negative trends
- High debt or leverage concerns
- Liquidity or solvency risks
- Operational risks evident in the numbers

FINANCIAL DATA:
{context}

Write a professional, balanced risk assessment. Be specific about quantifiable risks.
Also note any mitigating factors visible in the data."""
    }
    
    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self._init_llm()
    
    def _init_llm(self) -> None:
        """Initialize Gemini LLM"""
        if settings.validate_api_key():
            genai.configure(api_key=settings.gemini_api_key)
            self._discover_and_rank_models()
            if not self.available_models:
                # Fallback if discovery fails
                self.available_models = [settings.gemini_model, "gemini-1.5-flash", "gemini-pro"]
                print(f"Model discovery failed. Using default fallback: {self.available_models}")
            else:
                print(f"Discovered and ranked models: {self.available_models}")
            
            self.llm_available = True
        else:
            raise ValueError("Gemini API key not configured. Please set GEMINI_API_KEY in .env")

    def _discover_and_rank_models(self) -> None:
        """Discover available models and rank them by preference"""
        try:
            models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    models.append(m.name)
            
            # Ranking heuristic (Higher score = better/preferred)
            def score_model(name: str) -> int:
                name = name.lower()
                score = 0
                if "1.5" in name: score += 50
                if "pro" in name: score += 20
                if "flash" in name: score += 10 # Good fallback
                if "vision" in name: score -= 10 # Prefer text-optimized
                if "latest" in name: score += 5
                return score

            # Sort by score descending
            self.available_models = sorted(models, key=score_model, reverse=True)
            
            # Ensure configured model is first if valid
            configured = f"models/{settings.gemini_model}" if not settings.gemini_model.startswith("models/") else settings.gemini_model
            if configured in self.available_models:
                self.available_models.remove(configured)
                self.available_models.insert(0, configured)
                
        except Exception as e:
            print(f"Error listing models: {e}")
            self.available_models = []
    
    def generate(
        self, 
        company_name: str,
        kpis: Optional[List[KPIResult]] = None,
        request: Optional[GenerationRequest] = None
    ) -> MDADocument:
        """Generate complete MD&A document"""
        request = request or GenerationRequest(company_name=company_name)
        kpi_text = self._format_kpis(kpis) if kpis else "No KPI data available."
        
        doc = MDADocument(
            company_name=company_name,
            generation_date=date.today()
        )
        
        sections = request.include_sections
        
        if 'executive_summary' in sections:
            doc.executive_summary = self._generate_section(
                'executive_summary', company_name, kpi_text
            )
        
        if 'revenue_trends' in sections:
            doc.revenue_trends = self._generate_section(
                'revenue_trends', company_name, kpi_text
            )
        
        if 'profitability_analysis' in sections:
            doc.profitability_analysis = self._generate_section(
                'profitability_analysis', company_name, kpi_text
            )
        
        if 'financial_position' in sections:
            doc.financial_position = self._generate_section(
                'financial_position', company_name, kpi_text
            )
        
        if 'risk_factors' in sections:
            doc.risk_factors = self._generate_section(
                'risk_factors', company_name, kpi_text
            )
        
        return doc
    
    def _generate_section(
        self, 
        section_name: str, 
        company_name: str,
        kpi_text: str
    ) -> MDASection:
        """Generate a single MD&A section"""
        # Get context from RAG
        context, citations = self.rag_pipeline.build_context(section_name, company_name)
        
        if not context:
            context = "Limited financial data available. Please provide more detailed financial statements."
        
        # Build prompt
        prompt_template = self.PROMPTS.get(section_name, self.PROMPTS['executive_summary'])
        prompt = prompt_template.format(context=context, kpis=kpi_text)
        
        # Generate content
        # Generate content with model rotation
        content = "" # Initialize content
        if self.llm_available:
            last_error = None
            for model_name in self.available_models:
                try:
                    print(f"  🔄 Trying model: {model_name}...")
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=0.3,
                            max_output_tokens=4096
                        )
                    )
                    content = response.text
                    
                    # Check for truncation and attempt continuation
                    content = self._ensure_complete_response(model, prompt, content)
                    
                    print(f"  ✅ Success with: {model_name}")
                    break # Success!
                except Exception as e:
                    is_rate_limit = "429" in str(e) or "Quota" in str(e) or "ResourceExhausted" in str(e)
                    if is_rate_limit:
                        print(f"Warning: Rate limit hit for {model_name}. Rotating to next model...")
                        last_error = e
                        continue
                    else:
                        # Non-rate limit errors (like safety filters) might verify retry or just fail
                        print(f"Generation error ({model_name}): {e}")
                        last_error = e
                        continue # Try next model anyway? Or fail? User said "whenever limit gets hit".
                        # But aggressive rotation is safer for now.
            else:
                # Loop completed without success - Try Groq fallback
                if settings.groq_api_key:
                    print(f"All Gemini models exhausted. Falling back to Groq ({settings.groq_model})...")
                    content = self._call_groq(prompt)
                else:
                    raise Exception(f"Failed to generate section {section_name}. All models exhausted and no Groq API key configured. Last error: {last_error}")
        else:
            raise ValueError("LLM not available")
        
        # Format section title
        title_map = {
            'executive_summary': 'Executive Summary',
            'revenue_trends': 'Revenue Trends Analysis',
            'profitability_analysis': 'Profitability Analysis',
            'financial_position': 'Financial Position',
            'risk_factors': 'Risk Factors'
        }
        
        return MDASection(
            title=title_map.get(section_name, section_name.replace('_', ' ').title()),
            content=content,
            citations=citations,
            key_metrics=self._extract_key_metrics(context)
        )
    
    def _format_kpis(self, kpis: List[KPIResult]) -> str:
        """Format KPIs for prompt context"""
        if not kpis:
            return "No KPIs calculated."
        
        lines = ["Key Performance Indicators:"]
        for kpi in kpis:
            trend_icon = "↑" if kpi.trend == TrendDirection.INCREASING else "↓" if kpi.trend == TrendDirection.DECREASING else "→"
            lines.append(f"- {kpi.name}: {kpi.interpretation} {trend_icon}")
        
        return "\n".join(lines)
    
    def _extract_key_metrics(self, context: str) -> List[str]:
        """Extract key metric names from context"""
        metrics = []
        keywords = ['Revenue', 'Income', 'Profit', 'Assets', 'Liabilities', 'Cash', 'Margin', 'Growth']
        
        for keyword in keywords:
            if keyword.lower() in context.lower():
                metrics.append(keyword)
        
        return metrics[:5]  # Limit to top 5
    
    def _is_truncated(self, text: str) -> bool:
        """Check if response appears to be truncated mid-sentence"""
        if not text or len(text.strip()) < 50:
            return True
        
        text = text.strip()
        
        # Check for incomplete ending - no proper sentence ending
        valid_endings = ('.', '!', '?', ':', '|', '*', ')', ']', '"', "'", '`')
        if not text.endswith(valid_endings):
            return True
        
        # Check for common truncation patterns
        truncation_indicators = [
            ' and', ' or', ' the', ' a', ' an', ' to', ' of', ' in', ' for',
            ' with', ' by', ' from', ' at', ' on', ' is', ' are', ' was', ' were',
            '$', '%', ','
        ]
        
        last_20_chars = text[-20:].lower()
        for indicator in truncation_indicators:
            if last_20_chars.endswith(indicator):
                return True
        
        return False
    
    def _ensure_complete_response(self, model, original_prompt: str, content: str, max_continuations: int = 2) -> str:
        """Attempt to continue truncated responses"""
        if not self._is_truncated(content):
            return content
        
        print(f"    ⚠️ Response appears truncated, attempting continuation...")
        
        for attempt in range(max_continuations):
            try:
                continuation_prompt = f"""Continue the following text EXACTLY from where it was cut off. 
Do NOT repeat any content - just continue naturally from the last word/sentence.

TEXT TO CONTINUE:
{content[-500:]}

CONTINUE FROM HERE:"""
                
                response = model.generate_content(
                    continuation_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=2048
                    )
                )
                
                continuation = response.text.strip()
                if continuation:
                    content = content.rstrip() + " " + continuation
                    print(f"    ✅ Continuation {attempt + 1} added ({len(continuation)} chars)")
                    
                    if not self._is_truncated(content):
                        break
                        
            except Exception as e:
                print(f"    ⚠️ Continuation attempt {attempt + 1} failed: {e}")
                break
        
        return content

    def _call_groq(self, prompt: str) -> str:
        """Fallback to Groq API (Llama 3) when Gemini is exhausted"""
        import requests
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": settings.groq_model,
            "messages": [
                {"role": "system", "content": "You are a professional financial analyst writing MD&A sections."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2048
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")

    
    def save_report(self, doc: MDADocument, output_path: Optional[str] = None) -> str:
        """Save generated MD&A to file"""
        settings.ensure_directories()
        
        if output_path is None:
            filename = f"mda_{doc.company_name.replace(' ', '_')}_{doc.generation_date}.md"
            output_path = settings.output_dir / filename
        
        markdown_content = doc.to_markdown()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return str(output_path)
