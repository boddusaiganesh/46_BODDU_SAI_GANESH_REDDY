"""
MD&A Generator - Main Entry Point
Automated Management Discussion & Analysis draft generation from financial data
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings
from src.data_loader import DataLoader
from src.kpi_calculator import KPICalculator
from src.chunker import DocumentChunker
from src.rag_pipeline import RAGPipeline
from src.mda_generator import MDAGenerator
from src.evaluator import MDAGuardrails
from src.api_service import run_api_server


def main():
    """Main entry point for MD&A Generator"""
    parser = argparse.ArgumentParser(
        description="Generate MD&A drafts from financial statement data"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Path to financial data file (CSV or JSON)"
    )
    parser.add_argument(
        "--company", "-c",
        type=str,
        default="TechCorp Inc.",
        help="Company name for the report"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for generated MD&A"
    )
    parser.add_argument(
        "--sec",
        action="store_true",
        help="Use SEC financial data (requires downloaded dataset)"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Company ticker/name for SEC data (e.g., 'Apple', 'Microsoft')"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use built-in sample data for demonstration"
    )
    parser.add_argument(
        "--no-evaluation",
        action="store_true",
        help="Skip guardrails evaluation"
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Use agentic AI with multi-agent approach"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run FastAPI server instead of CLI"
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Run Streamlit web UI"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port"
    )

    args = parser.parse_args()

    # Run Streamlit if requested
    if args.streamlit:
        print("🎨 Starting Streamlit Web UI...")
        print(f"   URL: http://localhost:8501")
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
        return 0

    # Run API server if requested
    if args.api:
        print("🚀 Starting FastAPI Server...")
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Docs: http://localhost:{args.port}/docs")
        run_api_server(host=args.host, port=args.port)
        return 0
    
    print("=" * 60)
    print("MD&A Generator")
    print("Automated Management Discussion & Analysis Draft Generation")
    print("=" * 60)
    
    # Check API key
    if not settings.validate_api_key():
        print("\n[WARNING] Gemini API key not configured.")
        print("Set GEMINI_API_KEY in your .env file for full functionality.")
        print("Proceeding with mock generation for demonstration.\n")
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Check for SEC data conflict
    if args.sec and not args.ticker:
        print("\n[ERROR] --ticker is required when using --sec data.")
        return 1
        
    # Step 1: Load data
    print("\n[1/6] Loading financial data...")
    loader = DataLoader()
    
    if args.sec:
        print(f"      Loading SEC data for {args.ticker}...")
        df = loader.load_sec_data(args.ticker)
        if df.empty:
            print(f"      [ERROR] No data found for {args.ticker} in SEC dataset.")
            return 1
        print(f"      OK - Loaded {len(df)} records for {args.ticker}")
        
        # Update company name to format found in data if possible, or just use ticker
        try:
             # Try to get the canonical name from the data if available
             args.company = df['Company'].iloc[0]
        except:
             args.company = args.ticker
        
    elif args.sample or args.data is None:
        df = loader.load_sample_data()
        print("      OK - Loaded sample financial data")
    else:
        data_path = Path(args.data)
        if data_path.suffix == '.csv':
            df = loader.load_csv(data_path)
        elif data_path.suffix == '.json':
            df = loader.load_json(data_path)
        else:
            print(f"      ERROR - Unsupported file format {data_path.suffix}")
            return 1
        print(f"      OK - Loaded data from {data_path}")
    
    # Step 2: Calculate KPIs
    print("\n[2/6] Calculating KPIs...")
    kpi_calc = KPICalculator(df)
    kpis = kpi_calc.calculate_all_kpis()
    print(f"      OK - Calculated {len(kpis)} KPIs")
    
    # Display trend analysis
    trend_analysis = kpi_calc.generate_trend_analysis()
    print(f"      {trend_analysis}")
    
    # Step 3: Create chunks
    print("\n[3/6] Creating document chunks...")
    chunker = DocumentChunker()
    chunks = chunker.chunk_dataframe(df, args.company)
    chunks.extend(chunker.chunk_kpis(kpis, args.company))
    print(f"      OK - Created {len(chunks)} document chunks")
    
    # Step 4: Index chunks
    print("\n[4/6] Indexing in vector store...")
    rag = RAGPipeline()
    rag.clear_index()  # Start fresh
    indexed = rag.index_data(chunks)
    print(f"      OK - Indexed {indexed} chunks in ChromaDB")
    
    # Step 5: Generate MD&A
    print("\n[5/6] Generating MD&A sections...")
    generator = MDAGenerator(rag)

    if args.agentic:
        print("      Using Agentic AI mode...")
        import asyncio
        from src.agents import AgenticMDAGenerator

        agentic_gen = AgenticMDAGenerator(generator, kpis, df)
        mda_doc = asyncio.run(agentic_gen.generate_agentic_mda(args.company))
    else:
        mda_doc = generator.generate(args.company, kpis)

    print(f"      OK - Generated MD&A document")
    
    # Step 6: Run evaluation (if enabled)
    evaluation_report = None
    if not args.no_evaluation:
        print("\n[6/7] Running guardrails evaluation...")
        guardrails = MDAGuardrails(kpis, df)
        evaluation_report = guardrails.evaluate_document(mda_doc)
        print(f"      OK - Overall score: {evaluation_report.overall_score*100:.1f}%")
        print(f"      Factuality: {evaluation_report.factuality_score*100:.1f}%")
        print(f"      Citations: {evaluation_report.citation_score*100:.1f}%")
        print(f"      Guardrails Passed: {evaluation_report.passed_all_guardrails}")

    # Step 7: Save report
    print(f"\n[{6 if args.no_evaluation else 7}/{6 if args.no_evaluation else 7}] Saving report...")
    output_path = generator.save_report(mda_doc, args.output)

    # If evaluation ran, append quality summary to report
    if evaluation_report:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write("\n---\n")
            f.write("## Quality Evaluation\n\n")
            f.write(f"- **Overall Score**: {evaluation_report.overall_score*100:.1f}%\n")
            f.write(f"- **Factuality Score**: {evaluation_report.factuality_score*100:.1f}%\n")
            f.write(f"- **Citation Score**: {evaluation_report.citation_score*100:.1f}%\n")
            f.write(f"- **Consistency Score**: {evaluation_report.consistency_score*100:.1f}%\n")
            f.write(f"- **Completeness Score**: {evaluation_report.completeness_score*100:.1f}%\n")
            f.write(f"- **Guardrails Passed**: {evaluation_report.passed_all_guardrails}\n")

            if evaluation_report.recommendations:
                f.write("\n### Recommendations\n")
                for rec in evaluation_report.recommendations:
                    f.write(f"- {rec}\n")

    print(f"      OK - Saved report to: {output_path}")

    # Display summary
    print("\n" + "=" * 60)
    print("MD&A Generation Complete!")
    print("=" * 60)
    print(f"Company: {args.company}")
    print(f"Report saved to: {output_path}")
    print(f"Sections generated: 5")
    print(f"Mode: {'Agentic AI 🔥' if args.agentic else 'Standard'}")
    citation_count = sum(len(s.citations) for s in [
        mda_doc.executive_summary, mda_doc.revenue_trends,
        mda_doc.profitability_analysis, mda_doc.financial_position,
        mda_doc.risk_factors
    ] if s)
    print(f"Total citations: {citation_count}")

    if evaluation_report:
        print("\n" + "-" * 60)
        print("Quality Evaluation:")
        print(f"  Overall Score: {evaluation_report.overall_score*100:.1f}%")
        print(f"  Factuality: {evaluation_report.factuality_score*100:.1f}%")
        print(f"  Citations: {evaluation_report.citation_score*100:.1f}%")
        print(f"  Guardrails: {'✅ PASSED' if evaluation_report.passed_all_guardrails else '❌ FAILED'}")

    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
