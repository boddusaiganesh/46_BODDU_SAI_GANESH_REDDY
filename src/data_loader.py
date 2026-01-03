"""
Data Loader module for MD&A Generator
Handles loading and preprocessing of financial statement data from SEC filings
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from .schemas import FinancialStatement, FinancialMetric
from .config import settings


class DataLoader:
    """Load and preprocess financial statement data from various formats"""
    
    # Standard financial metrics mapping (XBRL Tags -> Readable Columns)
    METRIC_CATEGORIES = {
        'revenue': ['Revenues', 'SalesRevenueNet', 'SalesRevenueGoodsNet', 'RevenueFromContractWithCustomerExcludingAssessedTax'],
        'income': ['NetIncomeLoss', 'ProfitLoss', 'OperatingIncomeLoss', 'GrossProfit'],
        'assets': ['Assets', 'AssetsCurrent', 'CashAndCashEquivalentsAtCarryingValue'],
        'liabilities': ['Liabilities', 'LiabilitiesCurrent', 'LongTermDebt'],
        'equity': ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 'RetainedEarningsAccumulatedDeficit'],
        'expenses': ['OperatingExpenses', 'CostOfGoodsAndServicesSold', 'ResearchAndDevelopmentExpense', 'SellingGeneralAndAdministrativeExpense']
    }
    
    # Reverse mapping for easier lookup
    XBRL_MAP = {
        'Revenues': 'TotalRevenue',
        'SalesRevenueNet': 'TotalRevenue',
        'SalesRevenueGoodsNet': 'TotalRevenue',
        'RevenueFromContractWithCustomerExcludingAssessedTax': 'TotalRevenue',
        'NetIncomeLoss': 'NetIncome',
        'ProfitLoss': 'NetIncome',
        'OperatingIncomeLoss': 'OperatingIncome',
        'GrossProfit': 'GrossProfit',
        'Assets': 'TotalAssets',
        'AssetsCurrent': 'CurrentAssets',
        'CashAndCashEquivalentsAtCarryingValue': 'CashAndCashEquivalents',
        'Liabilities': 'TotalLiabilities',
        'LiabilitiesCurrent': 'CurrentLiabilities',
        'LongTermDebt': 'LongTermDebt',
        'StockholdersEquity': 'StockholdersEquity',
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest': 'StockholdersEquity',
        'RetainedEarningsAccumulatedDeficit': 'RetainedEarnings',
        'OperatingExpenses': 'OperatingExpenses',
        'CostOfGoodsAndServicesSold': 'CostOfRevenue',
        'ResearchAndDevelopmentExpense': 'ResearchAndDevelopment',
        'SellingGeneralAndAdministrativeExpense': 'SGA'
    }

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_sec_data(self, company_ticker: str = None) -> pd.DataFrame:
        """Load SEC financial data from JSON files in data/sec_data/"""
        sec_dir = self.data_dir / "sec_data"
        if not sec_dir.exists():
            print(f"Warning: {sec_dir} not found. Falling back to sample data.")
            return self.load_sample_data()
        
        all_metrics = []
        
        # Iterate through all quarter files
        for json_file in sec_dir.glob("*.json"):
            print(f"      Processing {json_file.name}...", flush=True)
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 1. Parse SUB.TXT (Company Info)
                # Format: adsh	cik	name	sic ...
                if 'sub.txt' not in data or 'num.txt' not in data:
                    continue
                    
                sub_lines = data['sub.txt'].split('\n')
                sub_headers = sub_lines[0].split('\t')
                
                # Create dictionary mapping adsh -> company info
                companies = {}
                ticker_to_adsh = {}
                
                for line in sub_lines[1:]:
                    if not line.strip(): continue
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        adsh = parts[0]
                        cik = parts[1]
                        name = parts[2]
                        # Use CIK as proxy for ticker if not available, or fuzzy match name
                        companies[adsh] = {'name': name, 'cik': cik}
                
                # Filter for specific company if requested
                target_adsh = []
                if company_ticker:
                    for adsh, info in companies.items():
                        if company_ticker.lower() in info['name'].lower():
                            target_adsh.append(adsh)
                
                if company_ticker and not target_adsh:
                    continue # Company not found in this quarter
                
                # 2. Parse NUM.TXT (Financial Numbers)
                # Format: adsh	tag	version	coreg	ddate	qtrs	uom	value	footnote
                num_lines = data['num.txt'].split('\n')
                
                quarter_metrics = {} # Key: (adsh, ddate) -> Value: {metric: value}
                
                for line in num_lines[1:]:
                    if not line.strip(): continue
                    parts = line.split('\t')
                    
                    if len(parts) < 8: continue
                    
                    adsh = parts[0]
                    tag = parts[1]
                    ddate = parts[4] # YYYYMMDD
                    value = parts[7]
                    
                    # Only process if it's a target company (or all if no filter)
                    if company_ticker and adsh not in target_adsh:
                        continue
                        
                    # Map XBRL tag to our readable metric name
                    if tag in self.XBRL_MAP:
                        metric_name = self.XBRL_MAP[tag]
                        
                        key = (adsh, ddate)
                        if key not in quarter_metrics:
                            quarter_metrics[key] = {
                                "Company": companies[adsh]['name'],
                                "Quarter": f"{ddate[:4]}-Q{((int(ddate[4:6])-1)//3)+1}", # Approx quarter
                                "PeriodDate": ddate
                            }
                        
                        try:
                            val = float(value)
                            # Pick the largest value if duplicates (common in XBRL for consolidated vs detail)
                            if metric_name in quarter_metrics[key]:
                                if abs(val) > abs(quarter_metrics[key][metric_name]):
                                    quarter_metrics[key][metric_name] = val
                            else:
                                quarter_metrics[key][metric_name] = val
                        except ValueError:
                            pass
                
                all_metrics.extend(quarter_metrics.values())
                
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                continue

        if not all_metrics:
            print("No data found for the specified criteria.")
            return pd.DataFrame() # Empty DF
            
        return pd.DataFrame(all_metrics)

    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load financial data from CSV file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        df = pd.read_csv(path)
        return self._normalize_dataframe(df)
    
    def load_json(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load financial data from JSON file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")
        
        return self._normalize_dataframe(df)
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and data types"""
        # Standardize column names
        df.columns = df.columns.str.strip()
        
        # Convert numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Clean currency and comma formatting
                    cleaned = df[col].str.replace(',', '', regex=False).str.replace('$', '', regex=False)
                    numeric_result = pd.to_numeric(cleaned, errors='coerce')
                    # Only use if at least some values converted successfully
                    if numeric_result.notna().any():
                        df[col] = numeric_result.fillna(df[col])
                except (AttributeError, TypeError):
                    pass
        
        return df
    
    def extract_financial_statement(
        self, 
        df: pd.DataFrame, 
        company_name: str,
        fiscal_year: int,
        fiscal_quarter: Optional[int] = None
    ) -> FinancialStatement:
        """Extract structured financial statement from dataframe"""
        
        metrics = []
        period = f"Q{fiscal_quarter} {fiscal_year}" if fiscal_quarter else str(fiscal_year)
        
        for category, metric_names in self.METRIC_CATEGORIES.items():
            for metric_name in metric_names:
                # Look for matching columns (mapped names)
                mapped_name = self.XBRL_MAP.get(metric_name, metric_name)
                matching_cols = [col for col in df.columns if mapped_name.lower() == col.lower()]
                
                for col in matching_cols:
                    values = df[col].dropna()
                    if len(values) > 0:
                        value = float(values.iloc[-1])  # Use most recent value
                        metrics.append(FinancialMetric(
                            name=col,
                            value=value,
                            unit="USD",
                            period=period,
                            category=category
                        ))
        
        return FinancialStatement(
            company_name=company_name,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            metrics=metrics,
            raw_data=df.to_dict()
        )
    
    def load_sample_data(self) -> pd.DataFrame:
        """Load or create sample financial data for demonstration"""
        sample_file = self.data_dir / "sample_financials.csv"
        
        if sample_file.exists():
            return self.load_csv(sample_file)
        
        # Create sample data
        sample_data = self._create_sample_data()
        df = pd.DataFrame(sample_data)
        df.to_csv(sample_file, index=False)
        return df
    
    def _create_sample_data(self) -> List[Dict[str, Any]]:
        """Create realistic sample financial data"""
        return [
            # Q4 2023
            {
                "Company": "TechCorp Inc.",
                "Quarter": "Q4 2023",
                "TotalRevenue": 5250000000,
                "NetIncome": 892500000,
                "GrossProfit": 2362500000,
                "OperatingIncome": 1050000000,
                "TotalAssets": 28750000000,
                "CurrentAssets": 12500000000,
                "CashAndCashEquivalents": 4500000000,
                "TotalLiabilities": 14375000000,
                "CurrentLiabilities": 5750000000,
                "LongTermDebt": 6250000000,
                "StockholdersEquity": 14375000000,
                "OperatingExpenses": 1312500000,
                "ResearchAndDevelopment": 787500000,
                "CostOfRevenue": 2887500000
            },
            # Q3 2023
            {
                "Company": "TechCorp Inc.",
                "Quarter": "Q3 2023",
                "TotalRevenue": 4850000000,
                "NetIncome": 776000000,
                "GrossProfit": 2182500000,
                "OperatingIncome": 970000000,
                "TotalAssets": 27500000000,
                "CurrentAssets": 11875000000,
                "CashAndCashEquivalents": 4125000000,
                "TotalLiabilities": 13750000000,
                "CurrentLiabilities": 5500000000,
                "LongTermDebt": 6000000000,
                "StockholdersEquity": 13750000000,
                "OperatingExpenses": 1212500000,
                "ResearchAndDevelopment": 727500000,
                "CostOfRevenue": 2667500000
            },
            # Q4 2022 (YoY comparison)
            {
                "Company": "TechCorp Inc.",
                "Quarter": "Q4 2022",
                "TotalRevenue": 4500000000,
                "NetIncome": 720000000,
                "GrossProfit": 2025000000,
                "OperatingIncome": 900000000,
                "TotalAssets": 25000000000,
                "CurrentAssets": 10625000000,
                "CashAndCashEquivalents": 3750000000,
                "TotalLiabilities": 12500000000,
                "CurrentLiabilities": 5000000000,
                "LongTermDebt": 5625000000,
                "StockholdersEquity": 12500000000,
                "OperatingExpenses": 1125000000,
                "ResearchAndDevelopment": 675000000,
                "CostOfRevenue": 2475000000
            },
            # Q3 2022
            {
                "Company": "TechCorp Inc.",
                "Quarter": "Q3 2022",
                "TotalRevenue": 4250000000,
                "NetIncome": 637500000,
                "GrossProfit": 1912500000,
                "OperatingIncome": 850000000,
                "TotalAssets": 24375000000,
                "CurrentAssets": 10250000000,
                "CashAndCashEquivalents": 3500000000,
                "TotalLiabilities": 12187500000,
                "CurrentLiabilities": 4875000000,
                "LongTermDebt": 5500000000,
                "StockholdersEquity": 12187500000,
                "OperatingExpenses": 1062500000,
                "ResearchAndDevelopment": 637500000,
                "CostOfRevenue": 2337500000
            }
        ]
    
    def get_company_data(self, company_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe for specific company"""
        company_col = None
        for col in df.columns:
            if 'company' in col.lower() or 'name' in col.lower():
                company_col = col
                break
        
        if company_col:
            return df[df[company_col].str.contains(company_name, case=False, na=False)]
        return df
    
    def get_periods(self, df: pd.DataFrame) -> List[str]:
        """Extract available periods from dataframe"""
        period_col = None
        for col in df.columns:
            if 'quarter' in col.lower() or 'period' in col.lower():
                period_col = col
                break
        
        if period_col:
            return df[period_col].unique().tolist()
        return []
