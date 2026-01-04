"""
Deployable Streamlit UI for MD&A Generator
Allows users to upload their own JSON data and configure API keys
"""

import streamlit as st
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="MD&A Generator - Deploy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stAlert {
        margin-top: 1rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .api-section {
        background-color: #e8f4ea;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'financial_df' not in st.session_state:
    st.session_state.financial_df = None
if 'generated_mda' not in st.session_state:
    st.session_state.generated_mda = None
if 'eval_report' not in st.session_state:
    st.session_state.eval_report = None
if 'kpis' not in st.session_state:
    st.session_state.kpis = None
if 'temp_data_dir' not in st.session_state:
    st.session_state.temp_data_dir = None


def configure_api_keys(gemini_key: str, groq_key: str = None):
    """Configure API keys in environment"""
    if gemini_key:
        os.environ['GEMINI_API_KEY'] = gemini_key
        st.session_state.api_key_configured = True
        return True
    return False


def process_uploaded_files(uploaded_files) -> pd.DataFrame:
    """Process uploaded JSON files and return a DataFrame"""
    all_data = []
    
    for uploaded_file in uploaded_files:
        try:
            content = json.load(uploaded_file)
            
            # Handle different JSON structures
            if isinstance(content, list):
                all_data.extend(content)
            elif isinstance(content, dict):
                # Check for common SEC data structures
                if 'data' in content:
                    all_data.extend(content['data'] if isinstance(content['data'], list) else [content['data']])
                else:
                    all_data.append(content)
                    
        except json.JSONDecodeError as e:
            st.error(f"Error parsing {uploaded_file.name}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Normalize column names
    column_mapping = {
        'totalRevenue': 'TotalRevenue',
        'netIncome': 'NetIncome',
        'operatingIncome': 'OperatingIncome',
        'totalAssets': 'TotalAssets',
        'totalLiabilities': 'TotalLiabilities',
        'stockholdersEquity': 'StockholdersEquity',
        'currentAssets': 'CurrentAssets',
        'currentLiabilities': 'CurrentLiabilities',
        'cashAndCashEquivalents': 'CashAndCashEquivalents',
        'researchAndDevelopment': 'ResearchAndDevelopment',
        'company': 'Company',
        'companyName': 'Company',
        'ticker': 'Ticker',
        'period': 'Period',
        'fiscalYear': 'FiscalYear',
        'fiscalQuarter': 'FiscalQuarter',
        'revenue': 'TotalRevenue',
        'net_income': 'NetIncome',
        'total_revenue': 'TotalRevenue',
        'total_assets': 'TotalAssets',
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    return df


def save_uploaded_files_to_temp(uploaded_files) -> str:
    """Save uploaded files to a temporary directory"""
    temp_dir = tempfile.mkdtemp()
    sec_data_dir = os.path.join(temp_dir, "sec_data")
    os.makedirs(sec_data_dir, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(sec_data_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
    
    return temp_dir


def load_data_from_temp(temp_dir: str) -> pd.DataFrame:
    """Load data from temporary directory using DataLoader"""
    from src.data_loader import DataLoader
    
    # Create a custom data loader pointing to temp directory
    loader = DataLoader(data_dir=Path(temp_dir))
    df = loader.load_sec_data()
    
    return df


def generate_mda_report(company_name: str, df: pd.DataFrame, use_agentic: bool = False):
    """Generate MD&A report from the loaded data"""
    from src.kpi_calculator import KPICalculator
    from src.chunker import DocumentChunker
    from src.rag_pipeline import RAGPipeline
    from src.mda_generator import MDAGenerator
    from src.evaluator import MDAGuardrails
    
    # Calculate KPIs
    kpi_calc = KPICalculator(df)
    kpis = kpi_calc.calculate_all_kpis()
    st.session_state.kpis = kpis
    
    # Create chunks
    chunker = DocumentChunker()
    chunks = chunker.chunk_dataframe(df, company_name)
    chunks.extend(chunker.chunk_kpis(kpis, company_name))
    
    # Index in RAG
    rag = RAGPipeline()
    rag.clear_index()
    rag.index_data(chunks)
    
    # Generate MD&A
    generator = MDAGenerator(rag)
    
    if use_agentic:
        import asyncio
        from src.agents import AgenticMDAGenerator
        agentic_gen = AgenticMDAGenerator(generator, kpis, df)
        mda_doc = asyncio.run(agentic_gen.generate_agentic_mda(company_name))
    else:
        mda_doc = generator.generate(company_name, kpis)
    
    # Evaluate
    guardrails = MDAGuardrails(kpis, df)
    eval_report = guardrails.evaluate_document(mda_doc)
    
    return mda_doc, eval_report, kpis


def display_mda_section(section):
    """Display a single MD&A section"""
    if section:
        st.markdown(f"### {section.title}")
        st.markdown(section.content)
        
        if section.key_metrics:
            st.markdown("**Key Metrics:** " + ", ".join(section.key_metrics))
        
        if section.citations:
            with st.expander(f"📚 Citations ({len(section.citations)})"):
                for i, citation in enumerate(section.citations, 1):
                    st.markdown(f"**[{i}]** {citation.source_text[:200]}... (Score: {citation.relevance_score:.2f})")


def display_kpis(kpis):
    """Display KPIs in a table"""
    kpi_data = []
    for kpi in kpis:
        trend_icon = "📈" if kpi.trend.value == "increasing" else "📉" if kpi.trend.value == "decreasing" else "➡️"
        
        if kpi.current_value >= 1e9:
            current_val = f"${kpi.current_value/1e9:.2f}B"
        elif kpi.current_value >= 1e6:
            current_val = f"${kpi.current_value/1e6:.2f}M"
        else:
            current_val = f"${kpi.current_value:,.2f}"
        
        kpi_data.append({
            'KPI': kpi.name,
            'Current': current_val,
            'YoY Change': f"{kpi.yoy_change*100:.1f}%" if kpi.yoy_change else "N/A",
            'QoQ Change': f"{kpi.qoq_change*100:.1f}%" if kpi.qoq_change else "N/A",
            'Trend': trend_icon
        })
    
    st.dataframe(pd.DataFrame(kpi_data), use_container_width=True, hide_index=True)


# Main Application
def main():
    # Header
    st.markdown("# 📊 AI-Powered MD&A Generator")
    st.markdown("### Upload Your Financial Data & Generate Professional Reports")
    st.markdown("---")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Section
        st.subheader("🔑 API Keys")
        
        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Enter your Gemini API key",
            help="Get your API key from https://aistudio.google.com/apikey"
        )
        
        groq_key = st.text_input(
            "Groq API Key (Optional)",
            type="password",
            placeholder="Enter Groq API key (fallback)",
            help="Optional: Used as fallback when Gemini rate limit is hit"
        )
        
        if st.button("💾 Save API Keys", use_container_width=True):
            if gemini_key:
                os.environ['GEMINI_API_KEY'] = gemini_key
                if groq_key:
                    os.environ['GROQ_API_KEY'] = groq_key
                st.session_state.api_key_configured = True
                st.success("✅ API keys configured!")
            else:
                st.error("❌ Gemini API key is required")
        
        # Show API status
        if st.session_state.api_key_configured:
            st.success("🔐 API Keys: Configured")
        else:
            st.warning("⚠️ API Keys: Not configured")
        
        st.divider()
        
        # Generation Options
        st.subheader("🎛️ Generation Options")
        use_agentic = st.checkbox(
            "Use Agentic AI Mode",
            value=False,
            help="Enable multi-agent processing for enhanced analysis"
        )
        
        st.divider()
        
        # Info
        st.markdown("**Features:**")
        st.markdown("- 📁 Upload JSON financial data")
        st.markdown("- 🤖 AI-powered analysis")
        st.markdown("- 📊 KPI calculations")
        st.markdown("- 📝 Professional MD&A reports")
        st.markdown("- ✅ Quality guardrails")
    
    # Main Content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Data", "📊 Generate Report", "📄 View Report", "📈 Analysis"])
    
    with tab1:
        st.header("Upload Financial Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Supported Formats:**
            - SEC Financial Statement JSON files
            - Custom JSON with financial metrics
            - Multiple files from quarterly/annual reports
            """)
            
            uploaded_files = st.file_uploader(
                "Upload JSON Files",
                type=['json'],
                accept_multiple_files=True,
                help="Upload one or more JSON files containing financial data"
            )
            
            if uploaded_files:
                st.session_state.uploaded_files = uploaded_files
                st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
                
                # Show uploaded files
                with st.expander("📋 Uploaded Files"):
                    for f in uploaded_files:
                        st.markdown(f"- {f.name} ({f.size / 1024:.1f} KB)")
        
        with col2:
            st.markdown("**Quick Start:**")
            st.markdown("1. Enter your Gemini API key")
            st.markdown("2. Upload JSON files")
            st.markdown("3. Click 'Load Data'")
            st.markdown("4. Generate report!")
        
        st.divider()
        
        # Load Data Button
        if st.session_state.uploaded_files:
            company_name_input = st.text_input(
                "Company Name",
                value="",
                placeholder="Enter company name (or leave blank to auto-detect)"
            )
            
            if st.button("📥 Load & Process Data", use_container_width=True, type="primary"):
                with st.spinner("Processing uploaded files..."):
                    try:
                        # Process uploaded files
                        df = process_uploaded_files(st.session_state.uploaded_files)
                        
                        if df.empty:
                            st.error("❌ No valid data found in uploaded files")
                        else:
                            st.session_state.financial_df = df
                            st.session_state.data_loaded = True
                            
                            # Auto-detect company name if not provided
                            if not company_name_input:
                                if 'Company' in df.columns:
                                    company_name_input = df['Company'].iloc[0]
                                else:
                                    company_name_input = "Unknown Company"
                            
                            st.session_state.company_name = company_name_input
                            
                            st.success(f"✅ Data loaded successfully! Found {len(df)} records for {company_name_input}")
                            
                            # Preview data
                            st.subheader("Data Preview")
                            st.dataframe(df.head(10), use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"❌ Error processing files: {e}")
    
    with tab2:
        st.header("Generate MD&A Report")
        
        if not st.session_state.api_key_configured:
            st.warning("⚠️ Please configure your Gemini API key in the sidebar first.")
        elif not st.session_state.data_loaded:
            st.warning("⚠️ Please upload and load financial data first.")
        else:
            st.success(f"✅ Ready to generate report for: **{st.session_state.get('company_name', 'Unknown')}**")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                **Report Sections:**
                - Executive Summary
                - Revenue Trends Analysis
                - Profitability Analysis
                - Financial Position
                - Risk Factors
                """)
            
            with col2:
                st.metric("Records", len(st.session_state.financial_df))
                if 'Period' in st.session_state.financial_df.columns:
                    periods = st.session_state.financial_df['Period'].nunique()
                    st.metric("Periods", periods)
            
            st.divider()
            
            if st.button("🚀 Generate MD&A Report", use_container_width=True, type="primary"):
                progress = st.progress(0, text="Initializing...")
                
                try:
                    progress.progress(20, text="Calculating KPIs...")
                    
                    progress.progress(50, text="Generating MD&A sections...")
                    
                    mda_doc, eval_report, kpis = generate_mda_report(
                        st.session_state.company_name,
                        st.session_state.financial_df,
                        use_agentic
                    )
                    
                    progress.progress(90, text="Finalizing...")
                    
                    st.session_state.generated_mda = mda_doc
                    st.session_state.eval_report = eval_report
                    st.session_state.kpis = kpis
                    
                    progress.progress(100, text="Complete!")
                    time.sleep(0.5)
                    progress.empty()
                    
                    st.success("✅ MD&A Report generated successfully!")
                    st.balloons()
                    
                except Exception as e:
                    progress.empty()
                    st.error(f"❌ Error generating report: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
    
    with tab3:
        st.header("View Generated Report")
        
        if st.session_state.generated_mda:
            mda_doc = st.session_state.generated_mda
            
            # Report header
            st.info(f"📊 **{mda_doc.company_name}** | Generated: {mda_doc.generation_date}")
            
            # Download button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                markdown_content = mda_doc.to_markdown()
                st.download_button(
                    label="📥 Download Report (Markdown)",
                    data=markdown_content,
                    file_name=f"mda_{mda_doc.company_name.replace(' ', '_')}_{mda_doc.generation_date}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            st.divider()
            
            # Display sections
            display_mda_section(mda_doc.executive_summary)
            st.divider()
            display_mda_section(mda_doc.revenue_trends)
            st.divider()
            display_mda_section(mda_doc.profitability_analysis)
            st.divider()
            display_mda_section(mda_doc.financial_position)
            st.divider()
            display_mda_section(mda_doc.risk_factors)
            
        else:
            st.info("📝 Generate a report first to view it here.")
    
    with tab4:
        st.header("Analysis & Quality Metrics")
        
        if st.session_state.eval_report:
            eval_report = st.session_state.eval_report
            
            # Quality scores
            st.subheader("📊 Quality Scores")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall", f"{eval_report.overall_score*100:.0f}%")
            with col2:
                st.metric("Factuality", f"{eval_report.factuality_score*100:.0f}%")
            with col3:
                st.metric("Citations", f"{eval_report.citation_score*100:.0f}%")
            with col4:
                st.metric("Completeness", f"{eval_report.completeness_score*100:.0f}%")
            
            if eval_report.passed_all_guardrails:
                st.success("✅ All quality guardrails passed!")
            else:
                st.warning("⚠️ Some guardrails need attention")
            
            # Guardrail details
            st.divider()
            st.subheader("🛡️ Guardrail Results")
            
            for check in eval_report.guardrail_results:
                status = "✅" if check.passed else "❌"
                with st.expander(f"{status} {check.name} ({check.score*100:.0f}%)"):
                    st.write(check.message)
            
            # Recommendations
            if eval_report.recommendations:
                st.divider()
                st.subheader("💡 Recommendations")
                for i, rec in enumerate(eval_report.recommendations, 1):
                    st.markdown(f"{i}. {rec}")
        
        # KPIs
        if st.session_state.kpis:
            st.divider()
            st.subheader("📈 Calculated KPIs")
            display_kpis(st.session_state.kpis)
        
        # Data preview
        if st.session_state.financial_df is not None:
            st.divider()
            st.subheader("📋 Source Data")
            st.dataframe(st.session_state.financial_df, use_container_width=True, hide_index=True)
        
        if not st.session_state.eval_report and not st.session_state.kpis:
            st.info("📊 Generate a report to see analysis metrics here.")
    
    # Footer
    st.divider()
    st.markdown("---")
    st.markdown("**MD&A Generator** | AI-Powered Financial Analysis | Built with Streamlit + Gemini")


if __name__ == "__main__":
    main()
