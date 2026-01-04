"""
Pydantic schemas for MD&A Generator
Defines data models for financial statements, KPIs, and generated output
"""

from typing import List, Optional, Dict, Any
from datetime import date
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class TrendDirection(str, Enum):
    """Enum for trend direction indicators"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class FinancialMetric(BaseModel):
    """Single financial metric with value and metadata"""
    name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Numeric value")
    unit: str = Field(default="USD", description="Unit of measurement")
    period: str = Field(..., description="Time period (e.g., Q1 2023)")
    category: str = Field(default="general", description="Category of metric")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Total Revenue",
                "value": 1500000000,
                "unit": "USD",
                "period": "Q4 2023",
                "category": "income"
            }
        }
    )


class FinancialStatement(BaseModel):
    """Complete financial statement for a company"""
    company_name: str = Field(..., description="Name of the company")
    ticker: Optional[str] = Field(None, description="Stock ticker symbol")
    cik: Optional[str] = Field(None, description="SEC CIK number")
    fiscal_year: int = Field(..., description="Fiscal year")
    fiscal_quarter: Optional[int] = Field(None, description="Fiscal quarter (1-4)")
    filing_date: Optional[date] = Field(None, description="Date of filing")
    metrics: List[FinancialMetric] = Field(default_factory=list)
    raw_data: Optional[Dict[str, Any]] = Field(None, description="Original raw data")


class KPIResult(BaseModel):
    """Calculated KPI with trend analysis"""
    name: str = Field(..., description="KPI name")
    current_value: float = Field(..., description="Current period value")
    previous_value: Optional[float] = Field(None, description="Previous period value")
    yoy_change: Optional[float] = Field(None, description="Year-over-year % change")
    qoq_change: Optional[float] = Field(None, description="Quarter-over-quarter % change")
    trend: TrendDirection = Field(default=TrendDirection.STABLE)
    interpretation: Optional[str] = Field(None, description="Brief interpretation")
    
    def calculate_trend(self, threshold: float = 0.05) -> TrendDirection:
        """Determine trend based on YoY change"""
        if self.yoy_change is None:
            return TrendDirection.STABLE
        if self.yoy_change > threshold:
            return TrendDirection.INCREASING
        elif self.yoy_change < -threshold:
            return TrendDirection.DECREASING
        return TrendDirection.STABLE


class DocumentChunk(BaseModel):
    """A chunk of financial document for RAG"""
    chunk_id: str = Field(..., description="Unique identifier")
    content: str = Field(..., description="Text content of the chunk")
    company: str = Field(..., description="Company name")
    period: str = Field(..., description="Time period")
    section: str = Field(default="general", description="Section type")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_embedding_text(self) -> str:
        """Format chunk for embedding generation"""
        return f"Company: {self.company} | Period: {self.period} | {self.content}"


class Citation(BaseModel):
    """Citation linking generated text to source chunk"""
    chunk_id: str = Field(..., description="Referenced chunk ID")
    source_text: str = Field(..., description="Quoted source text")
    relevance_score: float = Field(default=0.0, description="Relevance score 0-1")


class MDASection(BaseModel):
    """A section of the generated MD&A document"""
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Generated narrative content")
    citations: List[Citation] = Field(default_factory=list)
    key_metrics: List[str] = Field(default_factory=list)


class MDADocument(BaseModel):
    """Complete MD&A document output"""
    company_name: str = Field(..., description="Company name")
    generation_date: date = Field(default_factory=date.today)
    executive_summary: Optional[MDASection] = None
    revenue_trends: Optional[MDASection] = None
    profitability_analysis: Optional[MDASection] = None
    financial_position: Optional[MDASection] = None
    risk_factors: Optional[MDASection] = None
    
    def to_markdown(self) -> str:
        """Convert MD&A document to formatted markdown"""
        sections = []
        sections.append(f"# Management Discussion & Analysis: {self.company_name}")
        sections.append(f"*Generated on {self.generation_date}*\n")
        sections.append("---\n")
        
        if self.executive_summary:
            sections.append(self._format_section(self.executive_summary))
        if self.revenue_trends:
            sections.append(self._format_section(self.revenue_trends))
        if self.profitability_analysis:
            sections.append(self._format_section(self.profitability_analysis))
        if self.financial_position:
            sections.append(self._format_section(self.financial_position))
        if self.risk_factors:
            sections.append(self._format_section(self.risk_factors))
        
        # Add citations section
        all_citations = self._collect_all_citations()
        if all_citations:
            sections.append("\n---\n## Sources and Citations\n")
            for i, citation in enumerate(all_citations, 1):
                sections.append(f"[{i}] {citation.source_text[:100]}... (Score: {citation.relevance_score:.2f})")
        
        return "\n".join(sections)
    
    def _format_section(self, section: MDASection) -> str:
        """Format a single section to markdown"""
        md = f"## {section.title}\n\n{section.content}\n"
        if section.key_metrics:
            md += "\n**Key Metrics:** " + ", ".join(section.key_metrics) + "\n"
        return md
    
    def _collect_all_citations(self) -> List[Citation]:
        """Collect all citations from all sections"""
        citations = []
        for section in [self.executive_summary, self.revenue_trends, 
                       self.profitability_analysis, self.financial_position, 
                       self.risk_factors]:
            if section and section.citations:
                citations.extend(section.citations)
        return citations


class GenerationRequest(BaseModel):
    """Request for MD&A generation"""
    company_name: str
    include_sections: List[str] = Field(
        default=["executive_summary", "revenue_trends", "profitability_analysis", 
                 "financial_position", "risk_factors"]
    )
    max_tokens_per_section: int = Field(default=500)
    temperature: float = Field(default=0.3, ge=0, le=1)
