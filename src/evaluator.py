"""
Evaluator and Guardrails module for MD&A Generator
Implements factual consistency checks, metric accuracy verification, and output quality metrics
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from .schemas import MDADocument, MDASection, Citation, KPIResult


@dataclass
class GuardrailCheck:
    """Result of a guardrail check"""
    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for generated MD&A"""
    guardrail_results: List[GuardrailCheck]
    overall_score: float
    factuality_score: float
    citation_score: float
    consistency_score: float
    completeness_score: float
    passed_all_guardrails: bool
    recommendations: List[str]


class MDAGuardrails:
    """Guardrails for MD&A generation quality control"""
    
    # Financial number regex patterns
    MONEY_PATTERN = r'\$[\d,]+\.?\d*[KMB]?\b'
    PERCENTAGE_PATTERN = r'\d+\.?\d*%|\d+\.?\d* percent'
    NUMBER_PATTERN = r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b(?:\s*(?:billion|milion|million|k|thousand))?\b'
    
    def __init__(self, kpis: List[KPIResult], original_data: pd.DataFrame):
        self.kpis = kpis
        self.original_data = original_data
        self.metric_values = self._extract_metric_values()
    
    def _extract_metric_values(self) -> Dict[str, float]:
        """Extract actual metric values from source data"""
        values = {}
        for kpi in self.kpis:
            values[kpi.name.lower()] = kpi.current_value
        return values
    
    def evaluate_document(self, mda_doc: MDADocument) -> EvaluationReport:
        """Run comprehensive evaluation on MD&A document"""
        guardrail_results = []
        
        # 1. Factual consistency check
        fact_check = self.check_factual_consistency(mda_doc)
        guardrail_results.append(fact_check)
        
        # 2. Citation coverage check
        cit_check = self.check_citation_coverage(mda_doc)
        guardrail_results.append(cit_check)
        
        # 3. Section completeness check
        comp_check = self.check_section_completeness(mda_doc)
        guardrail_results.append(comp_check)
        
        # 4. Metric accuracy check
        acc_check = self.check_metric_accuracy(mda_doc)
        guardrail_results.append(acc_check)
        
        # 5. Content quality check
        qual_check = self.check_content_quality(mda_doc)
        guardrail_results.append(qual_check)
        
        # 6. Financial reasonableness check
        reason_check = self.check_financial_reasonableness(mda_doc)
        guardrail_results.append(reason_check)
        
        # Calculate scores
        factuality_score = fact_check.score if fact_check.passed else 0.0
        citation_score = cit_check.score
        consistency_score = acc_check.score
        completeness_score = comp_check.score
        overall_score = (factuality_score + citation_score + consistency_score + completeness_score) / 4
        
        passed_all = all(g.passed for g in guardrail_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(guardrail_results)
        
        return EvaluationReport(
            guardrail_results=guardrail_results,
            overall_score=overall_score,
            factuality_score=factuality_score,
            citation_score=citation_score,
            consistency_score=consistency_score,
            completeness_score=completeness_score,
            passed_all_guardrails=passed_all,
            recommendations=recommendations
        )
    
    def check_factual_consistency(self, mda_doc: MDADocument) -> GuardrailCheck:
        """Check that numerical claims don't contradict source data"""
        inconsistencies = []
        sections_checked = 0
        
        sections = [
            mda_doc.executive_summary, mda_doc.revenue_trends,
            mda_doc.profitability_analysis, mda_doc.financial_position
        ]
        
        for section in sections:
            if not section or not section.content:
                continue
            
            sections_checked += 1
            content = section.content.lower()
            
            # Check each KPI for consistency
            for kpi_name, actual_value in self.metric_values.items():
                # Extract numbers mentioned in text similar to this KPI
                if kpi_name in content:
                    extracted_values = self._extract_similar_values(content, kpi_name)
                    for extracted in extracted_values:
                        # Allow 5% tolerance for rounding
                        if not self._values_match(actual_value, extracted, tolerance=0.05):
                            inconsistencies.append({
                                'kpi': kpi_name,
                                'expected': actual_value,
                                'found': extracted,
                                'section': section.title
                            })
        
        passed = len(inconsistencies) == 0
        score = max(0.0, 1.0 - (len(inconsistencies) * 0.1))
        
        message = f"Found {len(inconsistencies)} potential factual inconsistencies" if inconsistencies else "All numerical claims are consistent with source data"
        
        return GuardrailCheck(
            name="Factual Consistency",
            passed=passed,
            score=min(1.0, score),
            message=message,
            details={'inconsistencies': inconsistencies, 'sections_checked': sections_checked}
        )
    
    def check_citation_coverage(self, mda_doc: MDADocument) -> GuardrailCheck:
        """Ensure key claims have supporting citations"""
        sections = [
            mda_doc.executive_summary, mda_doc.revenue_trends,
            mda_doc.profitability_analysis, mda_doc.financial_position,
            mda_doc.risk_factors
        ]
        
        total_sections = len([s for s in sections if s])
        sections_with_citations = len([s for s in sections if s and s.citations])
        
        # Count total claim sentences
        total_claims = 0
        total_citations = 0
        
        for section in sections:
            if not section or not section.content:
                continue
            
            # Count sentences with numerical data
            sentences = re.split(r'[.!?]+', section.content)
            claim_sentences = [s for s in sentences if re.search(self.MONEY_PATTERN, s) or re.search(self.PERCENTAGE_PATTERN, s)]
            total_claims += len(claim_sentences)
            total_citations += len(section.citations)
        
        # Citation coverage ratio
        coverage = min(1.0, total_citations / max(1, total_claims * 0.5))  # Aim for 1 citation per 2 claims
        
        passed = coverage >= 0.5  # At least 50% coverage
        score = coverage
        
        message = f"{total_citations} citations for {total_claims} key claims ({coverage*100:.1f}% coverage)"
        
        return GuardrailCheck(
            name="Citation Coverage",
            passed=passed,
            score=score,
            message=message,
            details={
                'total_claims': total_claims,
                'total_citations': total_citations,
                'coverage_ratio': coverage,
                'sections_with_citations': sections_with_citations,
                'total_sections': total_sections
            }
        )
    
    def check_section_completeness(self, mda_doc: MDADocument) -> GuardrailCheck:
        """Verify all required sections are present with sufficient content"""
        required_sections = [
            'executive_summary', 'revenue_trends', 'profitability_analysis',
            'financial_position', 'risk_factors'
        ]
        
        present_sections = []
        missing_sections = []
        insufficient_content = []
        
        min_word_count = 100  # Minimum words per section
        
        for section_name in required_sections:
            section = getattr(mda_doc, section_name, None)
            if section:
                word_count = len(section.content.split())
                if word_count >= min_word_count:
                    present_sections.append(section_name)
                else:
                    insufficient_content.append({
                        'section': section_name,
                        'word_count': word_count,
                        'required': min_word_count
                    })
            else:
                missing_sections.append(section_name)
        
        completeness = len(present_sections) / len(required_sections)
        passed = len(missing_sections) == 0 and len(insufficient_content) == 0
        
        message = f"{len(present_sections)}/{len(required_sections)} sections complete"
        if insufficient_content:
            message += f", {len(insufficient_content)} sections need more content"
        
        return GuardrailCheck(
            name="Section Completeness",
            passed=passed,
            score=completeness,
            message=message,
            details={
                'present': present_sections,
                'missing': missing_sections,
                'insufficient': insufficient_content
            }
        )
    
    def check_metric_accuracy(self, mda_doc: MDADocument) -> GuardrailCheck:
        """Verify calculated KPIs match what's reported in MD&A"""
        discrepancies = []
        
        sections = [
            mda_doc.executive_summary, mda_doc.revenue_trends,
            mda_doc.profitability_analysis, mda_doc.financial_position
        ]
        
        for section in sections:
            if not section or not section.content:
                continue
            
            # Extract claimed percentages from section
            content_lower = section.content.lower()
            
            # Check KPI trends
            for kpi in self.kpis:
                kpi_name = kpi.name.lower()
                
                # Look for percentage mentions related to this KPI
                if kpi_name in content_lower:
                    # Extract claims about growth/changes
                    if kpi.yoy_change is not None:
                        claimed_change = self._extract_percentage_claim(content_lower, kpi_name)
                        if claimed_change:
                            actual_pct = abs(kpi.yoy_change * 100)
                            if abs(claimed_change - actual_pct) > 5:  # 5% tolerance
                                discrepancies.append({
                                    'kpi': kpi.name,
                                    'claimed': claimed_change,
                                    'actual': actual_pct,
                                    'section': section.title
                                })
        
        passed = len(discrepancies) == 0
        score = max(0.0, 1.0 - len(discrepancies) * 0.15)
        
        message = f"Metric accuracy: {len(discrepancies)} discrepancies found"
        
        return GuardrailCheck(
            name="Metric Accuracy",
            passed=passed,
            score=score,
            message=message,
            details={'discrepancies': discrepancies}
        )
    
    def check_content_quality(self, mda_doc: MDADocument) -> GuardrailCheck:
        """Assess writing quality and professional tone"""
        quality_issues = []
        sections_checked = 0
        
        sections = [
            mda_doc.executive_summary, mda_doc.revenue_trends,
            mda_doc.profitability_analysis, mda_doc.financial_position,
            mda_doc.risk_factors
        ]
        
        for section in sections:
            if not section or not section.content:
                continue
            
            sections_checked += 1
            content = section.content
            
            # Check for common quality issues
            if len(content) < 200:
                quality_issues.append({
                    'section': section.title,
                    'issue': 'content_too_short',
                    'detail': f'Only {len(content)} characters'
                })
            
            # Check for repetitive sentences (exact matches of 10+ words)
            sentences = re.split(r'[.!?]+', content)
            sentences_long = [s.strip().lower() for s in sentences if len(s.split()) >= 10]
            if len(sentences_long) > len(set(sentences_long)):
                duplicates = len(sentences_long) - len(set(sentences_long))
                quality_issues.append({
                    'section': section.title,
                    'issue': 'repetitive_content',
                    'detail': f'{duplicates} duplicate sentences'
                })
            
            # Check for placeholder text
            placeholders = ['TODO', 'FIXME', 'XXX', '...', '[PLACEHOLDER]', 'CONTENT HERE']
            for placeholder in placeholders:
                if placeholder in content:
                    quality_issues.append({
                        'section': section.title,
                        'issue': 'placeholder_text',
                        'detail': f'Contains placeholder: {placeholder}'
                    })
        
        passed = len(quality_issues) == 0
        score = max(0.5, 1.0 - len(quality_issues) * 0.1)
        
        message = f"Content quality: {len(quality_issues)} issues found across {sections_checked} sections"
        
        return GuardrailCheck(
            name="Content Quality",
            passed=passed,
            score=score,
            message=message,
            details={'issues': quality_issues, 'sections_checked': sections_checked}
        )
    
    def check_financial_reasonableness(self, mda_doc: MDADocument) -> GuardrailCheck:
        """Check for logical financial relationships"""
        unreasonable_claims = []
        
        # Extract key financial relationships from content
        sections = [mda_doc.executive_summary, mda_doc.revenue_trends, mda_doc.profitability_analysis]
        
        for section in sections:
            if not section or not section.content:
                continue
            
            content = section.content.lower()
            
            # Check for impossible profit margins
            if 'profit margin' in content or 'margin' in content:
                margins = re.findall(r'margin.*?(\d+\.?\d*)\s*%', content)
                for margin_str in margins:
                    margin = float(margin_str)
                    if margin > 100 or margin < -100:
                        unreasonable_claims.append({
                            'section': section.title,
                            'issue': 'unrealistic_margin',
                            'value': margin,
                            'reason': 'Margins should be between -100% and 100%'
                        })
            
            # Check for impossible growth rates
            growth_claims = re.findall(r'(?:grew|increased|decreased).*?(\d+\.?\d*)\s*%', content)
            for growth_str in growth_claims:
                growth = float(growth_str)
                if growth > 10000:  # >10,000% is likely an error
                    unreasonable_claims.append({
                        'section': section.title,
                        'issue': 'unrealistic_growth',
                        'value': growth,
                        'reason': 'Growth rate >10,000% is questionable'
                    })
        
        passed = len(unreasonable_claims) == 0
        
        # Score based on severity
        score = 1.0 if passed else max(0.0, 1.0 - (len(unreasonable_claims) * 0.2))
        
        message = f"Financial reasonableness: {len(unreasonable_claims)} claims flagged"
        
        return GuardrailCheck(
            name="Financial Reasonableness",
            passed=passed,
            score=score,
            message=message,
            details={'flags': unreasonable_claims}
        )
    
    def _extract_similar_values(self, text: str, metric_name: str) -> List[float]:
        """Extract numeric values near mentions of a metric"""
        values = []
        
        # Find mentions of the metric
        metric_variants = [
            metric_name,
            metric_name.replace('_', ' '),
            metric_name.replace('_', ''),
            metric_name.replace(' ', '')
        ]
        
        for variant in metric_variants:
            if variant in text:
                # Extract nearby numbers (within 50 characters)
                idx = text.find(variant)
                context = text[max(0, idx-50):idx+len(variant)+50]
                numbers = re.findall(r'\$[\d,]+\.?\d*[KMB]?|[\d,]+\.?\d*[KMB]', context)
                for num in numbers:
                    parsed = self._parse_financial_number(num)
                    if parsed is not None:
                        values.append(parsed)
        
        return values
    
    def _parse_financial_number(self, num_str: str) -> Optional[float]:
        """Parse financial number string to float"""
        num_str = num_str.replace('$', '').replace(',', '').strip()
        
        multiplier = 1.0
        num_str_lower = num_str.lower()
        
        if 'b' in num_str_lower:
            multiplier = 1e9
            num_str = num_str_lower.replace('b', '')
        elif 'm' in num_str_lower:
            multiplier = 1e6
            num_str = num_str_lower.replace('m', '')
        elif 'k' in num_str_lower:
            multiplier = 1e3
            num_str = num_str_lower.replace('k', '')
        
        try:
            return float(num_str) * multiplier
        except ValueError:
            return None
    
    def _values_match(self, a: float, b: float, tolerance: float = 0.05) -> bool:
        """Check if two values match within tolerance"""
        if a == 0 or b == 0:
            return abs(a - b) < tolerance
        return abs((a - b) / b) <= tolerance
    
    def _extract_percentage_claim(self, text: str, metric_name: str) -> Optional[float]:
        """Extract percentage claim related to a metric"""
        # Look for patterns like "revenue grew 15%", "up 10%", etc.
        context_start = max(0, text.find(metric_name) - 100)
        context_end = min(len(text), text.find(metric_name) + 200)
        context = text[context_start:context_end]
        
        # Look for percentage indicators
        percentage_patterns = [
            r'(?:grew|increased|decreased|up|down|rose|fell).*?(\d+\.?\d*)\s*%',
            r'(\d+\.?\d*)\s*%.*?(?:growth|increase|decrease|change)'
        ]
        
        for pattern in percentage_patterns:
            match = re.search(pattern, context)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _generate_recommendations(self, guardrail_results: List[GuardrailCheck]) -> List[str]:
        """Generate improvement recommendations based on guardrail results"""
        recommendations = []
        
        for result in guardrail_results:
            if not result.passed:
                if result.name == "Factual Consistency":
                    recommendations.append("Review numerical claims for accuracy and cross-reference with source data")
                elif result.name == "Citation Coverage":
                    recommendations.append("Add more citations to support key financial claims")
                elif result.name == "Section Completeness":
                    recommendations.append("Expand sections with insufficient content to meet minimum word count")
                elif result.name == "Metric Accuracy":
                    recommendations.append("Verify that reported percentages match calculated KPIs")
                elif result.name == "Content Quality":
                    recommendations.append("Improve writing quality: reduce repetition, remove placeholders, increase content depth")
                elif result.name == "Financial Reasonableness":
                    recommendations.append("Review flagged claims for logical reasonableness")
        
        return recommendations
    
    def get_quality_summary(self, mda_doc: MDADocument) -> str:
        """Get a quick summary of MD&A quality"""
        report = self.evaluate_document(mda_doc)
        
        summary = f"""
        MD&A Quality Summary
        =====================
        Overall Score: {report.overall_score*100:.1f}%
        
        Detailed Scores:
        - Factual Consistency: {report.factuality_score*100:.1f}%
        - Citation Coverage: {report.citation_score*100:.1f}%
        - Metric Accuracy: {report.consistency_score*100:.1f}%
        - Section Completeness: {report.completeness_score*100:.1f}%
        
        Guardrails Passed: {report.passed_all_guardrails}
        """
        
        if report.recommendations:
            summary += "\nRecommendations:\n"
            for rec in report.recommendations:
                summary += f"  - {rec}\n"
        
        return summary