"""
KPI Calculator module for MD&A Generator
Computes Year-over-Year, Quarter-over-Quarter changes and financial ratios
"""

import pandas as pd
from typing import List, Dict, Optional, Tuple
from .schemas import KPIResult, TrendDirection


class KPICalculator:
    """Calculate key financial metrics and trend analysis"""
    
    # Define which metrics to calculate and their display names
    CORE_KPIS = {
        'revenue_growth': ('TotalRevenue', 'Revenue Growth'),
        'net_income_growth': ('NetIncome', 'Net Income Growth'),
        'gross_margin': ('GrossProfit', 'Gross Margin'),
        'operating_margin': ('OperatingIncome', 'Operating Margin'),
        'profit_margin': ('NetIncome', 'Profit Margin'),
        'asset_growth': ('TotalAssets', 'Asset Growth'),
        'cash_position': ('CashAndCashEquivalents', 'Cash Position'),
        'rd_intensity': ('ResearchAndDevelopment', 'R&D Intensity'),
    }
    
    RATIO_KPIS = {
        'current_ratio': ('CurrentAssets', 'CurrentLiabilities', 'Current Ratio'),
        'debt_to_equity': ('TotalLiabilities', 'StockholdersEquity', 'Debt to Equity'),
        'roe': ('NetIncome', 'StockholdersEquity', 'Return on Equity'),
    }
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._prepare_data()
    
    def _prepare_data(self) -> None:
        """Prepare dataframe for calculations"""
        # Identify the period column
        self.period_col = None
        for col in self.df.columns:
            if 'quarter' in col.lower() or 'period' in col.lower():
                self.period_col = col
                break
        
        if self.period_col:
            # Sort by period (newest first for easier comparison)
            self.df = self.df.sort_values(self.period_col, ascending=False).reset_index(drop=True)
    
    def calculate_all_kpis(self) -> List[KPIResult]:
        """Calculate all KPIs and return results"""
        results = []
        
        # Calculate growth KPIs
        for kpi_key, (metric_col, display_name) in self.CORE_KPIS.items():
            if metric_col in self.df.columns:
                result = self._calculate_growth_kpi(metric_col, display_name)
                if result:
                    results.append(result)
        
        # Calculate ratio KPIs
        for kpi_key, (num_col, denom_col, display_name) in self.RATIO_KPIS.items():
            if num_col in self.df.columns and denom_col in self.df.columns:
                result = self._calculate_ratio_kpi(num_col, denom_col, display_name)
                if result:
                    results.append(result)
        
        # Calculate margin KPIs
        margin_results = self._calculate_margin_kpis()
        results.extend(margin_results)
        
        return results
    
    def _calculate_growth_kpi(self, metric_col: str, display_name: str) -> Optional[KPIResult]:
        """Calculate YoY and QoQ growth for a metric"""
        if len(self.df) < 2:
            return None
        
        try:
            current_value = float(self.df[metric_col].iloc[0])
            previous_qoq = float(self.df[metric_col].iloc[1]) if len(self.df) > 1 else None
            previous_yoy = float(self.df[metric_col].iloc[3]) if len(self.df) > 3 else None
            
            qoq_change = None
            yoy_change = None
            
            if previous_qoq and previous_qoq != 0:
                qoq_change = (current_value - previous_qoq) / previous_qoq
            
            if previous_yoy and previous_yoy != 0:
                yoy_change = (current_value - previous_yoy) / previous_yoy
            
            # Determine trend
            ref_change = yoy_change if yoy_change is not None else qoq_change
            if ref_change is not None:
                if ref_change > 0.05:
                    trend = TrendDirection.INCREASING
                elif ref_change < -0.05:
                    trend = TrendDirection.DECREASING
                else:
                    trend = TrendDirection.STABLE
            else:
                trend = TrendDirection.STABLE
            
            interpretation = self._generate_interpretation(display_name, current_value, yoy_change, qoq_change)
            
            return KPIResult(
                name=display_name,
                current_value=current_value,
                previous_value=previous_yoy or previous_qoq,
                yoy_change=yoy_change,
                qoq_change=qoq_change,
                trend=trend,
                interpretation=interpretation
            )
        except (IndexError, TypeError, ValueError):
            return None
    
    def _calculate_ratio_kpi(self, num_col: str, denom_col: str, display_name: str) -> Optional[KPIResult]:
        """Calculate financial ratio KPI"""
        try:
            num_current = float(self.df[num_col].iloc[0])
            denom_current = float(self.df[denom_col].iloc[0])
            
            if denom_current == 0:
                return None
            
            current_ratio = num_current / denom_current
            
            # Calculate previous ratios
            previous_ratio = None
            yoy_change = None
            
            if len(self.df) > 3:
                num_prev = float(self.df[num_col].iloc[3])
                denom_prev = float(self.df[denom_col].iloc[3])
                if denom_prev != 0:
                    previous_ratio = num_prev / denom_prev
                    yoy_change = (current_ratio - previous_ratio) / previous_ratio if previous_ratio != 0 else None
            
            trend = TrendDirection.STABLE
            if yoy_change:
                if yoy_change > 0.1:
                    trend = TrendDirection.INCREASING
                elif yoy_change < -0.1:
                    trend = TrendDirection.DECREASING
            
            interpretation = f"{display_name} is {current_ratio:.2f}"
            if yoy_change:
                change_pct = yoy_change * 100
                interpretation += f", {'improved' if yoy_change > 0 else 'declined'} by {abs(change_pct):.1f}% YoY"
            
            return KPIResult(
                name=display_name,
                current_value=current_ratio,
                previous_value=previous_ratio,
                yoy_change=yoy_change,
                trend=trend,
                interpretation=interpretation
            )
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            return None
    
    def _calculate_margin_kpis(self) -> List[KPIResult]:
        """Calculate margin-based KPIs"""
        results = []
        
        margin_configs = [
            ('GrossProfit', 'TotalRevenue', 'Gross Margin'),
            ('OperatingIncome', 'TotalRevenue', 'Operating Margin'),
            ('NetIncome', 'TotalRevenue', 'Net Profit Margin'),
        ]
        
        for profit_col, revenue_col, display_name in margin_configs:
            if profit_col in self.df.columns and revenue_col in self.df.columns:
                try:
                    current_profit = float(self.df[profit_col].iloc[0])
                    current_revenue = float(self.df[revenue_col].iloc[0])
                    
                    if current_revenue == 0:
                        continue
                    
                    current_margin = (current_profit / current_revenue) * 100
                    
                    previous_margin = None
                    yoy_change = None
                    
                    if len(self.df) > 3:
                        prev_profit = float(self.df[profit_col].iloc[3])
                        prev_revenue = float(self.df[revenue_col].iloc[3])
                        if prev_revenue != 0:
                            previous_margin = (prev_profit / prev_revenue) * 100
                            yoy_change = (current_margin - previous_margin) / previous_margin if previous_margin != 0 else None
                    
                    trend = TrendDirection.STABLE
                    if yoy_change:
                        if yoy_change > 0.05:
                            trend = TrendDirection.INCREASING
                        elif yoy_change < -0.05:
                            trend = TrendDirection.DECREASING
                    
                    interpretation = f"{display_name} is {current_margin:.1f}%"
                    if previous_margin:
                        diff = current_margin - previous_margin
                        interpretation += f", {'up' if diff > 0 else 'down'} {abs(diff):.1f} percentage points YoY"
                    
                    results.append(KPIResult(
                        name=display_name,
                        current_value=current_margin,
                        previous_value=previous_margin,
                        yoy_change=yoy_change,
                        trend=trend,
                        interpretation=interpretation
                    ))
                except (IndexError, TypeError, ValueError):
                    continue
        
        return results
    
    def _generate_interpretation(
        self, 
        metric_name: str, 
        current: float, 
        yoy_change: Optional[float], 
        qoq_change: Optional[float]
    ) -> str:
        """Generate human-readable interpretation of KPI"""
        parts = [f"{metric_name}: ${current/1e9:.2f}B"]
        
        if yoy_change is not None:
            direction = "increased" if yoy_change > 0 else "decreased"
            parts.append(f"{direction} {abs(yoy_change)*100:.1f}% year-over-year")
        
        if qoq_change is not None:
            direction = "up" if qoq_change > 0 else "down"
            parts.append(f"{direction} {abs(qoq_change)*100:.1f}% quarter-over-quarter")
        
        return ", ".join(parts)
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics for quick overview"""
        stats = {}
        
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numeric_cols:
            if len(self.df) > 0:
                stats[col] = {
                    'current': float(self.df[col].iloc[0]),
                    'mean': float(self.df[col].mean()),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max()),
                }
        
        return stats
    
    def generate_trend_analysis(self) -> str:
        """Generate textual trend analysis summary"""
        kpis = self.calculate_all_kpis()
        
        improving = [k for k in kpis if k.trend == TrendDirection.INCREASING]
        declining = [k for k in kpis if k.trend == TrendDirection.DECREASING]
        stable = [k for k in kpis if k.trend == TrendDirection.STABLE]
        
        analysis = []
        analysis.append(f"Overall: {len(improving)} metrics improving, {len(declining)} declining, {len(stable)} stable.")
        
        if improving:
            analysis.append(f"Improving: {', '.join([k.name for k in improving])}")
        
        if declining:
            analysis.append(f"Declining: {', '.join([k.name for k in declining])}")
        
        return " ".join(analysis)
