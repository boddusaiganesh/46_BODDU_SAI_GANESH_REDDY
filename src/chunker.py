"""
Document Chunker module for MD&A Generator
Handles splitting financial data and narratives into semantic chunks for RAG
"""

import uuid
from typing import List, Dict, Any
import pandas as pd
from .schemas import DocumentChunk, KPIResult
from .config import settings


class DocumentChunker:
    """Split financial data into semantic chunks for embedding and retrieval"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    def chunk_dataframe(self, df: pd.DataFrame, company: str) -> List[DocumentChunk]:
        """Convert financial dataframe into document chunks"""
        chunks = []
        
        # Get period column
        period_col = None
        for col in df.columns:
            if 'quarter' in col.lower() or 'period' in col.lower():
                period_col = col
                break
        
        # Create chunks for each period
        for idx, row in df.iterrows():
            period = row[period_col] if period_col else f"Period {idx}"
            
            # Create different section chunks
            chunks.extend(self._create_income_chunks(row, company, period))
            chunks.extend(self._create_balance_sheet_chunks(row, company, period))
            chunks.extend(self._create_cash_flow_chunks(row, company, period))
            
        # Create comparison chunks
        if len(df) > 1:
            chunks.extend(self._create_comparison_chunks(df, company))
        
        return chunks
    
    def _create_income_chunks(self, row: pd.Series, company: str, period: str) -> List[DocumentChunk]:
        """Create chunks for income statement data"""
        chunks = []
        
        income_metrics = ['TotalRevenue', 'NetIncome', 'GrossProfit', 'OperatingIncome', 
                         'CostOfRevenue', 'OperatingExpenses', 'ResearchAndDevelopment']
        
        content_parts = [f"Income Statement for {company} - {period}:"]
        for metric in income_metrics:
            if metric in row.index and pd.notna(row[metric]):
                value = row[metric]
                formatted = self._format_value(value)
                content_parts.append(f"- {self._format_metric_name(metric)}: {formatted}")
        
        if len(content_parts) > 1:
            content = "\n".join(content_parts)
            chunks.append(DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                company=company,
                period=period,
                section="income_statement",
                metadata={"metrics": income_metrics}
            ))
        
        return chunks
    
    def _create_balance_sheet_chunks(self, row: pd.Series, company: str, period: str) -> List[DocumentChunk]:
        """Create chunks for balance sheet data"""
        chunks = []
        
        asset_metrics = ['TotalAssets', 'CurrentAssets', 'CashAndCashEquivalents']
        liability_metrics = ['TotalLiabilities', 'CurrentLiabilities', 'LongTermDebt', 'StockholdersEquity']
        
        # Assets chunk
        asset_parts = [f"Assets for {company} - {period}:"]
        for metric in asset_metrics:
            if metric in row.index and pd.notna(row[metric]):
                value = row[metric]
                formatted = self._format_value(value)
                asset_parts.append(f"- {self._format_metric_name(metric)}: {formatted}")
        
        if len(asset_parts) > 1:
            chunks.append(DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                content="\n".join(asset_parts),
                company=company,
                period=period,
                section="assets",
                metadata={"metrics": asset_metrics}
            ))
        
        # Liabilities chunk
        liability_parts = [f"Liabilities & Equity for {company} - {period}:"]
        for metric in liability_metrics:
            if metric in row.index and pd.notna(row[metric]):
                value = row[metric]
                formatted = self._format_value(value)
                liability_parts.append(f"- {self._format_metric_name(metric)}: {formatted}")
        
        if len(liability_parts) > 1:
            chunks.append(DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                content="\n".join(liability_parts),
                company=company,
                period=period,
                section="liabilities_equity",
                metadata={"metrics": liability_metrics}
            ))
        
        return chunks
    
    def _create_cash_flow_chunks(self, row: pd.Series, company: str, period: str) -> List[DocumentChunk]:
        """Create chunks for cash flow related data"""
        chunks = []
        
        if 'CashAndCashEquivalents' in row.index and pd.notna(row['CashAndCashEquivalents']):
            content = f"Cash Position for {company} - {period}:\n"
            content += f"- Cash and Cash Equivalents: {self._format_value(row['CashAndCashEquivalents'])}"
            
            chunks.append(DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                company=company,
                period=period,
                section="cash_flow",
                metadata={"metrics": ["CashAndCashEquivalents"]}
            ))
        
        return chunks
    
    def _create_comparison_chunks(self, df: pd.DataFrame, company: str) -> List[DocumentChunk]:
        """Create chunks comparing periods for trend analysis"""
        chunks = []
        
        if len(df) < 2:
            return chunks
        
        period_col = None
        for col in df.columns:
            if 'quarter' in col.lower() or 'period' in col.lower():
                period_col = col
                break
        
        current = df.iloc[0]
        previous = df.iloc[1]
        
        current_period = current[period_col] if period_col else "Current"
        previous_period = previous[period_col] if period_col else "Previous"
        
        comparison_parts = [f"Quarter-over-Quarter Comparison for {company}:"]
        comparison_parts.append(f"Comparing {current_period} vs {previous_period}")
        
        key_metrics = ['TotalRevenue', 'NetIncome', 'GrossProfit', 'TotalAssets']
        for metric in key_metrics:
            if metric in current.index and metric in previous.index:
                curr_val = current[metric]
                prev_val = previous[metric]
                if pd.notna(curr_val) and pd.notna(prev_val) and prev_val != 0:
                    change = ((curr_val - prev_val) / prev_val) * 100
                    direction = "increased" if change > 0 else "decreased"
                    comparison_parts.append(
                        f"- {self._format_metric_name(metric)}: {direction} by {abs(change):.1f}% "
                        f"(from {self._format_value(prev_val)} to {self._format_value(curr_val)})"
                    )
        
        if len(comparison_parts) > 2:
            chunks.append(DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                content="\n".join(comparison_parts),
                company=company,
                period=f"{current_period} vs {previous_period}",
                section="comparison",
                metadata={"comparison_type": "qoq"}
            ))
        
        # YoY comparison if we have 4+ quarters
        if len(df) >= 4:
            yoy_current = df.iloc[0]
            yoy_previous = df.iloc[3]
            
            yoy_current_period = yoy_current[period_col] if period_col else "Current Year"
            yoy_previous_period = yoy_previous[period_col] if period_col else "Previous Year"
            
            yoy_parts = [f"Year-over-Year Comparison for {company}:"]
            yoy_parts.append(f"Comparing {yoy_current_period} vs {yoy_previous_period}")
            
            for metric in key_metrics:
                if metric in yoy_current.index and metric in yoy_previous.index:
                    curr_val = yoy_current[metric]
                    prev_val = yoy_previous[metric]
                    if pd.notna(curr_val) and pd.notna(prev_val) and prev_val != 0:
                        change = ((curr_val - prev_val) / prev_val) * 100
                        direction = "increased" if change > 0 else "decreased"
                        yoy_parts.append(
                            f"- {self._format_metric_name(metric)}: {direction} by {abs(change):.1f}% YoY"
                        )
            
            if len(yoy_parts) > 2:
                chunks.append(DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    content="\n".join(yoy_parts),
                    company=company,
                    period=f"{yoy_current_period} vs {yoy_previous_period}",
                    section="comparison",
                    metadata={"comparison_type": "yoy"}
                ))
        
        return chunks
    
    def chunk_kpis(self, kpis: List[KPIResult], company: str) -> List[DocumentChunk]:
        """Create chunks from calculated KPIs"""
        chunks = []
        
        # Group KPIs by category
        growth_kpis = [k for k in kpis if 'Growth' in k.name or 'Position' in k.name]
        margin_kpis = [k for k in kpis if 'Margin' in k.name]
        ratio_kpis = [k for k in kpis if 'Ratio' in k.name or 'Equity' in k.name]
        
        # Growth metrics chunk
        if growth_kpis:
            content = f"Growth Metrics for {company}:\n"
            for kpi in growth_kpis:
                content += f"- {kpi.interpretation}\n"
            chunks.append(DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                content=content.strip(),
                company=company,
                period="Latest",
                section="growth_metrics",
                metadata={"kpi_type": "growth"}
            ))
        
        # Margin metrics chunk
        if margin_kpis:
            content = f"Profitability Margins for {company}:\n"
            for kpi in margin_kpis:
                content += f"- {kpi.interpretation}\n"
            chunks.append(DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                content=content.strip(),
                company=company,
                period="Latest",
                section="profitability",
                metadata={"kpi_type": "margins"}
            ))
        
        # Financial ratios chunk
        if ratio_kpis:
            content = f"Financial Ratios for {company}:\n"
            for kpi in ratio_kpis:
                content += f"- {kpi.interpretation}\n"
            chunks.append(DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                content=content.strip(),
                company=company,
                period="Latest",
                section="ratios",
                metadata={"kpi_type": "ratios"}
            ))
        
        return chunks
    
    def _format_value(self, value: float) -> str:
        """Format numeric value for display"""
        if abs(value) >= 1e9:
            return f"${value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.2f}K"
        else:
            return f"${value:.2f}"
    
    def _format_metric_name(self, name: str) -> str:
        """Convert camelCase metric names to readable format"""
        import re
        # Insert space before capitals and convert to title case
        formatted = re.sub(r'([A-Z])', r' \1', name).strip()
        return formatted.title()
