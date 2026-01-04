"""
Streamlit UI for MD&A Generator
Interactive web interface for financial MD&A generation
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import time
from datetime import datetime

from src.data_loader import DataLoader
from src.kpi_calculator import KPICalculator
from src.chunker import DocumentChunker
from src.rag_pipeline import RAGPipeline
from src.mda_generator import MDAGenerator
from src.evaluator import MDAGuardrails

# Page configuration
st.set_page_config(
    page_title="MD&A Generator",
    page_icon="Stock",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'generated_mda' not in st.session_state:
    st.session_state.generated_mda = None
if 'eval_report' not in st.session_state:
    st.session_state.eval_report = None
if 'company_data' not in st.session_state:
    st.session_state.company_data = None
if 'kpis' not in st.session_state:
    st.session_state.kpis = None


def load_data(company_ticker: str, use_sec: bool = False):
    """Load financial data"""
    loader = DataLoader()

    with st.spinner("Loading financial data..."):
        if use_sec and company_ticker:
            df = loader.load_sec_data(company_ticker)
            if df.empty:
                st.warning(f"No SEC data found for {company_ticker}. Using sample data instead.")
                df = loader.load_sample_data()
        else:
            df = loader.load_sample_data()

    return df


def calculate_kpis(df):
    """Calculate KPIs from financial data"""
    with st.spinner("Calculating KPIs..."):
        kpi_calc = KPICalculator(df)
        kpis = kpi_calc.calculate_all_kpis()
        trend = kpi_calc.generate_trend_analysis()
    return kpis, trend


def generate_mda(company_name: str, df, kpis, use_agentic: bool = False):
    """Generate MD&A document"""
    with st.spinner("Indexing documents and generating MD&A..."):
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
            st.info("Using Agentic AI mode with multi-agent processing...")
            import asyncio
            from src.agents import AgenticMDAGenerator

            agentic_gen = AgenticMDAGenerator(generator, kpis, df)
            mda_doc = asyncio.run(agentic_gen.generate_agentic_mda(company_name))
        else:
            mda_doc = generator.generate(company_name, kpis)

    return mda_doc


def evaluate_mda(mda_doc, kpis, df):
    """Evaluate generated MD&A"""
    with st.spinner("Running guardrails evaluation..."):
        guardrails = MDAGuardrails(kpis, df)
        report = guardrails.evaluate_document(mda_doc)
    return report


def display_metrics(metrics):
    """Display metrics in a nice format"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Score", f"{metrics['overall_score']*100:.1f}%")
    with col2:
        st.metric("Factuality", f"{metrics['factuality_score']*100:.1f}%")
    with col3:
        st.metric("Citations", f"{metrics['citation_score']*100:.1f}%")
    with col4:
        st.metric("Completeness", f"{metrics['completeness_score']*100:.1f}%")


def display_kpis(kpis):
    """Display KPIs in a table"""
    kpi_data = []
    for kpi in kpis:
        # Determine trend icon
        if kpi.trend.value == "increasing":
            trend_icon = "UP"
        elif kpi.trend.value == "decreasing":
            trend_icon = "DOWN"
        else:
            trend_icon = "STABLE"

        # Format current value
        if kpi.current_value < 1000000:
            current_val = f"${kpi.current_value:,.2f}"
        else:
            current_val = f"${kpi.current_value/1000000:.2f}M"

        # Format YoY change
        if kpi.yoy_change:
            yoy_val = f"{kpi.yoy_change*100:.1f}%"
        else:
            yoy_val = "N/A"

        # Format QoQ change
        if kpi.qoq_change:
            qoq_val = f"{kpi.qoq_change*100:.1f}%"
        else:
            qoq_val = "N/A"

        kpi_data.append({
            'KPI': kpi.name,
            'Current': current_val,
            'YoY Change': yoy_val,
            'QoQ Change': qoq_val,
            'Trend': trend_icon
        })

    st.dataframe(pd.DataFrame(kpi_data), use_container_width=True, hide_index=True)


def display_mda_section(section):
    """Display a single MD&A section"""
    if section:
        st.markdown(f"### {section.title}")
        st.markdown(section.content)

        if section.key_metrics:
            st.markdown("**Key Metrics:**")
            st.markdown(", ".join(section.key_metrics))

        if section.citations:
            with st.expander(f"Citations ({len(section.citations)})"):
                for i, citation in enumerate(section.citations, 1):
                    st.markdown(f"**[{i}]** {citation.source_text[:200]}... (Relevance: {citation.relevance_score:.2f})")


# Main application
def main():
    # Header
    st.markdown("# 📊 MD&A Generator")
    st.markdown("AI-powered Management Discussion & Analysis Draft Generation")

    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # Data source
        st.subheader("Data Source")
        use_sec = st.checkbox("Use SEC Dataset", value=False)
        company_ticker = st.text_input("Company Ticker", placeholder="e.g., AAPL, MSFT", value="")

        if not use_sec:
            company_ticker = ""

        # Generation mode
        st.subheader("Generation Mode")
        use_agentic = st.checkbox("Agentic AI (Multi-Agent)", value=False, help="Use multiple AI agents for enhanced MD&A")

        # Advanced options
        st.subheader("Advanced")
        run_evaluation = st.checkbox("Run Quality Evaluation", value=True)
        output_filename = st.text_input("Output Filename", value="mda_report")

        st.divider()

        st.markdown("**Features:**")
        st.markdown("- RAG retrieval (ChromaDB)")
        st.markdown("- Local embeddings (HuggingFace)")
        st.markdown("- LLM generation (Gemini)")
        st.markdown("- Guardrails & evaluation")
        st.markdown("- Agentic AI agents")

    # Main content area
    tab1, tab2, tab3 = st.tabs(["Generate", "Results", "Analysis"])

    with tab1:
        st.header("Generate MD&A Report")

        # Company input
        col1, col2 = st.columns([2, 1])
        with col1:
            company_name = st.text_input("Company Name", value="TechCorp Inc.", placeholder="Enter company name")
        with col2:
            st.write("")
            st.write("")
            generate_btn = st.button("Generate MD&A", use_container_width=True, type="primary")

        if generate_btn:
            if not company_name:
                st.error("Please enter a company name")
                return

            # Progress bar
            progress = st.progress(0, text="Initializing...")

            # Step 1: Load data
            progress.progress(20, text="Loading financial data...")
            df = load_data(company_ticker if use_sec else company_name, use_sec)
            st.session_state.company_data = df

            # Step 2: Calculate KPIs
            progress.progress(40, text="Calculating KPIs...")
            kpis, trend = calculate_kpis(df)
            st.session_state.kpis = kpis

            # Update company name from data
            if 'Company' in df.columns and len(df) > 0:
                # Only override if using SEC data (where we want the real name)
                # If using sample data (demo), keep the user's input name for simulation
                if use_sec:
                    company_name = df['Company'].iloc[0]

            # Step 3: Generate MD&A
            progress.progress(70, text="Generating MD&A sections...")
            mda_doc = generate_mda(company_name, df, kpis, use_agentic)
            st.session_state.generated_mda = mda_doc

            # Step 4: Evaluate
            if run_evaluation:
                progress.progress(90, text="Running quality evaluation...")
                eval_report = evaluate_mda(mda_doc, kpis, df)
                st.session_state.eval_report = eval_report

            progress.progress(100, text="Complete!")
            time.sleep(0.5)
            progress.empty()

            # Success message
            st.success(f"MD&A report generated for {company_name}!")

            if use_agentic:
                st.info("Agentic AI mode used - enhanced with multi-agent processing")

    with tab2:
        if st.session_state.generated_mda:
            st.header("MD&A Report Results")

            mda_doc = st.session_state.generated_mda

            # Display company info
            st.info(f"Company: {mda_doc.company_name} | Generated: {mda_doc.generation_date} | Sections: 5")

            st.divider()

            # Executive Summary
            st.markdown("## Executive Summary")
            display_mda_section(mda_doc.executive_summary)

            st.divider()

            # Revenue Trends
            st.markdown("## Revenue Trends")
            display_mda_section(mda_doc.revenue_trends)

            st.divider()

            # Profitability Analysis
            st.markdown("## Profitability Analysis")
            display_mda_section(mda_doc.profitability_analysis)

            st.divider()

            # Financial Position
            st.markdown("## Financial Position")
            display_mda_section(mda_doc.financial_position)

            st.divider()

            # Risk Factors
            st.markdown("## Risk Factors")
            display_mda_section(mda_doc.risk_factors)

            # Download button
            st.divider()
            st.subheader("Download Report")

            markdown_content = mda_doc.to_markdown()
            st.download_button(
                label="Download Markdown",
                data=markdown_content,
                file_name=f"{output_filename}.md",
                mime="text/markdown"
            )

            # Display all citations
            st.divider()
            with st.expander("View All Citations"):
                all_citations = []
                sections = [
                    mda_doc.executive_summary, mda_doc.revenue_trends,
                    mda_doc.profitability_analysis, mda_doc.financial_position,
                    mda_doc.risk_factors
                ]

                for section in sections:
                    if section and section.citations:
                        for citation in section.citations:
                            all_citations.append({
                                'Section': section.title,
                                'Source': citation.source_text[:100],
                                'Relevance': f"{citation.relevance_score:.2f}"
                            })

                if all_citations:
                    st.dataframe(pd.DataFrame(all_citations), use_container_width=True, hide_index=True)

        else:
            st.info("Generate an MD&A report first to see results here.")

    with tab3:
        if st.session_state.generated_mda:
            st.header("Analysis & Quality Metrics")

            # Quality Evaluation
            if st.session_state.eval_report:
                st.subheader("Quality Evaluation")

                eval_report = st.session_state.eval_report

                # Display overall metrics
                metrics = {
                    'overall_score': eval_report.overall_score,
                    'factuality_score': eval_report.factuality_score,
                    'citation_score': eval_report.citation_score,
                    'completeness_score': eval_report.completeness_score
                }
                display_metrics(metrics)

                # Status indicator
                if eval_report.passed_all_guardrails:
                    st.success("All guardrails passed!")
                else:
                    st.warning("Some guardrails did not pass. See details below.")

                # Detailed guardrail results
                st.divider()
                st.subheader("Detailed Guardrail Results")

                for i, check in enumerate(eval_report.guardrail_results, 1):
                    status_emoji = "PASS" if check.passed else "FAIL"
                    with st.expander(f"{status_emoji} {check.name} ({check.score*100:.1f}%)"):
                        st.write(f"Message: {check.message}")
                        if check.details:
                            st.json(check.details)

                # Recommendations
                if eval_report.recommendations:
                    st.divider()
                    st.subheader("Recommendations")
                    for i, rec in enumerate(eval_report.recommendations, 1):
                        st.write(f"{i}. {rec}")

            # KPIs
            if st.session_state.kpis:
                st.divider()
                st.subheader("Calculated KPIs")

                # Summary stats
                kpis = st.session_state.kpis
                improving = len([k for k in kpis if k.trend.value == "increasing"])
                declining = len([k for k in kpis if k.trend.value == "decreasing"])
                stable = len([k for k in kpis if k.trend.value == "stable"])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Improving", improving, delta="metrics")
                with col2:
                    st.metric("Declining", declining, delta="metrics", delta_color="inverse")
                with col3:
                    st.metric("Stable", stable, delta="metrics")

                # KPI table
                st.divider()
                display_kpis(kpis)

            # Raw data preview
            st.divider()
            st.subheader("Source Data Preview")

            if st.session_state.company_data is not None:
                df = st.session_state.company_data

                # Format display
                display_df = df.copy()
                numeric_cols = display_df.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"${x/1e9:.2f}B" if abs(x) >= 1e9 else f"${x/1e6:.2f}M" if abs(x) >= 1e6 else f"${x:,.2f}"
                    )

                st.dataframe(display_df, use_container_width=True, hide_index=True)

        else:
            st.info("Generate an MD&A report first to see analysis here.")

    # Footer
    st.divider()
    st.markdown("---")
    st.markdown("MD&A Generator | Finance Hackathon Application")
    st.markdown("RAG + LLM + Agentic AI | Built with Streamlit")


if __name__ == "__main__":
    main()
