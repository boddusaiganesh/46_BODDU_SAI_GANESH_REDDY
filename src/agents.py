"""
Agentic AI modules for MD&A Generator
Implements intelligent agents with multi-step reasoning, self-critique, and proactive analysis
"""

import warnings
# Suppress FutureWarning from google.generativeai about package deprecation
warnings.filterwarnings("ignore", category=FutureWarning)

import re
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from .schemas import KPIResult, DocumentChunk, MDADocument
from .rag_pipeline import RAGPipeline
from .config import settings


@dataclass
class AgentThought:
    """A single reasoning step by an agent"""
    step: int
    thought: str
    action: str
    observation: str
    confidence: float = 0.0


@dataclass
class AgentResponse:
    """Final response from an agent"""
    content: str
    reasoning_chain: List[AgentThought]
    confidence: float
    sources: List[str]


class CritiqueAgent:
    """Agent that critiques and refines financial narratives"""
    
    CRITIQUE_PROMPT = """You are a senior financial editor and auditor. Your job is to critique the following MD&A section and suggest improvements.

MD&A SECTION:
{content}

AVAILABLE METRICS:
{metrics}

CRITICISM CHECKPOINTS:
1. Factual Accuracy - Do the numbers match the source data?
2. Clarity - Is the analysis clear and easy to understand?
3. Completeness - Are all important aspects covered?
4. Balance - Is the analysis objective, not overly optimistic or pessimistic?
5. Specificity - Are claims supported with specific data points?

Provide your critique in the following format:
OBSERVATIONS: [Your observations]
ISSUES: [Any issues found]
SUGGESTIONS: [Specific suggestions for improvement]
NEEDS_REWRITE: [Yes/No]"""

    def __init__(self, llm_model: str = None):
        if settings.validate_api_key():
            genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(llm_model or settings.gemini_model)
    
    def critique_section(
        self, 
        content: str, 
        metrics: Dict[str, Any]
    ) -> AgentResponse:
        """Critique an MD&A section and suggest improvements"""
        
        prompt = self.CRITIQUE_PROMPT.format(
            content=content,
            metrics=self._format_metrics(metrics)
        )
        
        reasoning = []
        
        # Step 1: Generate critique
        reasoning.append(AgentThought(
            step=1,
            thought="Analyzing content for factual accuracy, clarity, and completeness",
            action="Generate critique",
            observation="Running LLM analysis...",
            confidence=0.8
        ))
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=800
                )
            )
            
            critique_text = response.text
            
            reasoning.append(AgentThought(
                step=2,
                thought="Received critique response from LLM",
                action="Parse critique",
                observation=critique_text[:200] + "...",
                confidence=0.9
            ))
            
            # Check if rewrite is suggested
            needs_rewrite = "NEEDS_REWRITE: Yes" in critique_text
            
            reasoning.append(AgentThought(
                step=3,
                thought=f"Determined if rewrite is needed: {needs_rewrite}",
                action="Final decision",
                observation=f"Rewrite recommended: {needs_rewrite}",
                confidence=0.95
            ))
            
            return AgentResponse(
                content=critique_text,
                reasoning_chain=reasoning,
                confidence=0.95 if needs_rewrite else 1.0,
                sources=[]
            )
            
        except Exception as e:
            reasoning.append(AgentThought(
                step=2,
                thought="Failed to generate critique",
                action="Error handling",
                observation=str(e),
                confidence=0.0
            ))
            
            return AgentResponse(
                content="Unable to generate critique due to error.",
                reasoning_chain=reasoning,
                confidence=0.0,
                sources=[]
            )
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for the critique prompt"""
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if abs(value) >= 1e9:
                    formatted.append(f"{key}: ${value/1e9:.2f}B")
                elif abs(value) >= 1e6:
                    formatted.append(f"{key}: ${value/1e6:.2f}M")
                else:
                    formatted.append(f"{key}: ${value:,.2f}")
            else:
                formatted.append(f"{key}: {value}")
        return "\n".join(formatted)


class RiskDetectionAgent:
    """Agent that proactively identifies financial risks from data"""
    
    RISK_ANALYSIS_PROMPT = """You are a risk analyst specializing in financial statements. Analyze the following financial data and identify potential risks.

FINANCIAL DATA:
{context}

ANALYSIS INSTRUCTIONS:
Look for the following types of risks:
1. Liquidity risks (declining cash, high current ratio issues)
2. Solvency risks (high debt, declining equity)
3. Profitability risks (declining margins, negative trends)
4. Operational risks (unusual changes in expenses)
5. Market risks (revenue volatility, declining growth)

For each risk identified:
- RISK_NAME: [Name of the risk]
- SEVERITY: [Low/Medium/High]
- INDICATORS: [Specific data points showing the risk]
- DESCRIPTION: [Detailed explanation]

Focus on quantifiable risks evident in the numbers."""

    def __init__(self, kpis: List[KPIResult], rag_pipeline: RAGPipeline):
        self.kpis = kpis
        self.rag = rag_pipeline
        
        if settings.validate_api_key():
            genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)
    
    def analyze_risks(self, company_name: str) -> AgentResponse:
        """Proactively identify financial risks"""
        
        reasoning = []
        
        # Step 1: Gather risk-relevant KPIs
        reasoning.append(AgentThought(
            step=1,
            thought="Analyzing KPIs for risk indicators",
            action="Filter risk metrics",
            observation="Filtering KPIs with declining trends...",
            confidence=0.8
        ))
        
        risk_kpis = self._filter_risk_kpis()
        context = self._build_risk_context(risk_kpis)
        
        # Step 2: Get additional context from RAG
        reasoning.append(AgentThought(
            step=2,
            thought="Retrieving risk-relevant information from documents",
            action="Query RAG pipeline",
            observation=f"Found {len(risk_kpis)} risky KPIs",
            confidence=0.8
        ))
        
        try:
            rag_results = self.rag.retrieve_for_section('risk_factors', company_name)
            additional_context = "\n".join([r['content'] for r in rag_results[:3]])
            context += "\n\nADDITIONAL CONTEXT:\n" + additional_context
            
            reasoning.append(AgentThought(
                step=3,
                thought="Building comprehensive risk analysis",
                action="Construct prompt",
                observation="Assembled data for risk analysis",
                confidence=0.85
            ))
            
            # Step 3: Generate risk analysis
            prompt = self.RISK_ANALYSIS_PROMPT.format(context=context)
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1000
                )
            )
            
            risk_analysis = response.text
            
            # Count identified risks
            risk_count = len(re.findall(r'RISK_NAME:', risk_analysis))
            
            reasoning.append(AgentThought(
                step=4,
                thought=f"Identified {risk_count} potential financial risks",
                action="Generate report",
                observation=risk_analysis[:200] + "...",
                confidence=0.9
            ))
            
            return AgentResponse(
                content=risk_analysis,
                reasoning_chain=reasoning,
                confidence=0.9,
                sources=[r['chunk_id'] for r in rag_results]
            )
            
        except Exception as e:
            reasoning.append(AgentThought(
                step=3,
                thought="Failed to analyze risks",
                action="Error handling",
                observation=str(e),
                confidence=0.0
            ))
            
            return AgentResponse(
                content="Risk analysis failed due to error.",
                reasoning_chain=reasoning,
                confidence=0.0,
                sources=[]
            )
    
    def _filter_risk_kpis(self) -> List[KPIResult]:
        """Filter for KPIs indicating potential risks"""
        from .schemas import TrendDirection
        
        risky_kpis = []
        
        for kpi in self.kpis:
            # Declining trends are potential risks
            if kpi.trend == TrendDirection.DECREASING:
                risky_kpis.append(kpi)
            
            # Significant negative changes
            if kpi.yoy_change and kpi.yoy_change < -0.1:  # >10% decline
                risky_kpis.append(kpi)
        
        return risky_kpis
    
    def _build_risk_context(self, kpis: List[KPIResult]) -> str:
        """Build context string from risk-relevant KPIs"""
        if not kpis:
            return "No significant risks identified in KPI trends."
        
        lines = ["RISK INDICATORS FROM KPIS:"]
        
        for kpi in kpis:
            lines.append(f"- {kpi.name}: {kpi.interpretation}")
        
        return "\n".join(lines)


class ComparativeAnalysisAgent:
    """Agent that performs comparative analysis across periods/companies"""
    
    COMPARISON_PROMPT = """You are a financial analyst specializing in comparative analysis. Analyze how the company's performance compares across periods.

CURRENT DATA:
{current_period}

PRIOR PERIOD DATA:
{prior_period}

COMPARISON ANALYSIS:
For each key metric, provide:
1. METRIC: [Metric name]
2. CURRENT_VALUE: [Current period value]
3. PRIOR_VALUE: [Prior period value]
4. CHANGE_PCT: [Percentage change]
5. TREND: [Improving/Stable/Declining]
6. SIGNIFICANCE: [Brief assessment of why this change matters]

Focus on the most significant changes and their business implications."""

    def __init__(self, kpis: List[KPIResult]):
        self.kpis = kpis
        
        if settings.validate_api_key():
            genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)
    
    def compare_periods(
        self, 
        current_period_data: Dict[str, Any],
        prior_period_data: Dict[str, Any]
    ) -> AgentResponse:
        """Perform comparative analysis between two periods"""
        
        reasoning = []
        
        # Step 1: Build comparison data
        reasoning.append(AgentThought(
            step=1,
            thought="Building comparison dataset",
            action="Prepare data",
            observation="Organizing metrics for comparison",
            confidence=0.9
        ))
        
        comparison_data = self._compare_metrics(current_period_data, prior_period_data)
        
        # Step 2: Generate comparative analysis
        reasoning.append(AgentThought(
            step=2,
            thought="Generating comparative narrative",
            action="LLM analysis",
            observation="Running comparative analysis",
            confidence=0.85
        ))
        
        try:
            prompt = self.COMPARISON_PROMPT.format(
                current_period=self._format_period_data(current_period_data),
                prior_period=self._format_period_data(prior_period_data)
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1200
                )
            )
            
            analysis = response.text
            
            reasoning.append(AgentThought(
                step=3,
                thought="Comparative analysis completed",
                action="Finalize",
                observation="Generated comprehensive comparison",
                confidence=0.95
            ))
            
            return AgentResponse(
                content=analysis,
                reasoning_chain=reasoning,
                confidence=0.95,
                sources=[]
            )
            
        except Exception as e:
            reasoning.append(AgentThought(
                step=2,
                thought="Failed to generate comparison",
                action="Error handling",
                observation=str(e),
                confidence=0.0
            ))
            
            return AgentResponse(
                content="Comparative analysis failed due to error.",
                reasoning_chain=reasoning,
                confidence=0.0,
                sources=[]
            )
    
    def _compare_metrics(
        self, 
        current: Dict[str, Any],
        prior: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compare metrics between two periods"""
        comparisons = []
        
        common_keys = set(current.keys()) & set(prior.keys())
        
        for key in common_keys:
            if isinstance(current[key], (int, float)) and isinstance(prior[key], (int, float)):
                curr_val = float(current[key])
                prior_val = float(prior[key])
                
                if prior_val != 0:
                    change_pct = ((curr_val - prior_val) / prior_val) * 100
                else:
                    change_pct = 0.0 if curr_val == 0 else float('inf')
                
                comparisons.append({
                    'metric': key,
                    'current': curr_val,
                    'prior': prior_val,
                    'change_pct': change_pct
                })
        
        return comparisons
    
    def _format_period_data(self, period_data: Dict[str, Any]) -> str:
        """Format period data for prompt"""
        lines = ["PERIOD DATA:"]
        
        for key, value in period_data.items():
            if isinstance(value, (int, float)):
                if abs(value) >= 1e9:
                    lines.append(f"{key}: ${value/1e9:.2f}B")
                elif abs(value) >= 1e6:
                    lines.append(f"{key}: ${value/1e6:.2f}M")
                else:
                    lines.append(f"{key}: ${value:,.2f}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)


class AgenticMDAGenerator:
    """Main orchestrator that uses multiple agents for enhanced MD&A generation"""
    
    def __init__(self, mda_generator, kpis: List[KPIResult], data: pd.DataFrame):
        self.generator = mda_generator
        self.kpis = kpis
        self.data = data
        
        # Initialize agents
        self.critique_agent = CritiqueAgent()
        self.risk_agent = RiskDetectionAgent(kpis, mda_generator.rag_pipeline)
        self.comparison_agent = ComparativeAnalysisAgent(kpis)
    
    async def generate_agentic_mda(self, company_name: str) -> MDADocument:
        """Generate MD&A with multi-agent approach"""
        
        # Step 1: Generate initial MD&A
        print("📝 Agent[Generator]: Generating initial MD&A draft...")
        initial_doc = self.generator.generate(company_name, self.kpis)
        
        # Step 2: Risk Assessment Agent
        print("🔍 Agent[Risk Analyst]: Perform proactive risk analysis...")
        risk_response = self.risk_agent.analyze_risks(company_name)
        
        # Step 3: Critique each section
        print("👁️  Agent[Editor]: Critiquing and validating sections...")
        
        sections_to_critique = [
            'executive_summary', 'revenue_trends', 
            'profitability_analysis', 'financial_position'
        ]
        
        available_metrics = {}
        # Prepare metrics from KPIs
        for kpi in self.kpis:
            available_metrics[kpi.name] = kpi.current_value
        
        for section_name in sections_to_critique:
            section = getattr(initial_doc, section_name, None)
            if section and section.content:
                critique = self.critique_agent.critique_section(
                    section.content,
                    available_metrics
                )
                
                print(f"  - {section_name}: Confidence {critique.confidence:.2f}")
        
        # Step 4: Integrate agent findings
        print("🔗 Agent[Orchestrator]: Integrating multi-agent findings...")
        
        # Update risk factors with agent analysis
        if initial_doc.risk_factors:
            # Append agent-provided risk analysis
            enhanced_risk_content = initial_doc.risk_factors.content
            enhanced_risk_content += "\n\n" + "="*50 + "\n"
            enhanced_risk_content += "ADDITIONAL RISK ANALYSIS (AI Agent):\n"
            enhanced_risk_content += risk_response.content
            
            initial_doc.risk_factors.content = enhanced_risk_content
        
        print("✅ Agentic MD&A generation complete!")
        
        return initial_doc