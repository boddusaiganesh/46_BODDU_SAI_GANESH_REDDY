# Comprehensive Codebase Analysis: AI-Powered MD&A Generator

## 1. Executive Summary
The project is a sophisticated financial reporting tool designed to automate the creation of **Management Discussion & Analysis (MD&A)** drafts. It leverages a modern AI stack combining **Retrieval Augmented Generation (RAG)**, **Large Language Models (Google Gemini)**, and **Agentic AI** workflows to produce high-quality, fact-checked financial narratives from raw data.

**Key Strengths:**
- **Hybrid Architecture:** Successfully integrates deterministic calculations (KPIs) with probabilistic generation (LLMs).
- **Agentic Workflow:** Goes beyond simple prompting by using specialized agents to critique and refine content.
- **Robust Guardrails:** Implements a strict evaluation framework to minimize hallucinations, a critical feature for financial compliance.
- **Flexibility:** Supports both raw SEC filings (JSON) and simplified CSV inputs, accessible via CLI, API, or Web UI.

## 2. System Architecture

The application follows a linear pipeline with feedback loops in the Agentic mode:

```mermaid
graph TD
    Data[Data Source<br/>(SEC JSON / CSV)] --> Loader[Data Loader]
    Loader --> KPI[KPI Calculator]
    Loader --> Chunker[Document Chunker]
    KPI --> Chunker
    Chunker --> Vector[ChromaDB Vector Store]
    
    subgraph "Generation Layer"
        Vector --> RAG[RAG Pipeline]
        RAG --> Generator[MD&A Generator]
        KPI --> Generator
    end
    
    subgraph "Agentic Layer (Optional)"
        Generator --> Agents[Agent Orchestrator]
        Agents --> Risk[Risk Agent]
        Agents --> Critique[Critique Agent]
        Agents --> Compare[Comparative Agent]
        Risk --> Refuter[Refinement]
        Critique --> Refuter
    end
    
    Refuter --> Guardrails[Evaluator / Guardrails]
    Generator --> Guardrails
    Guardrails --> Report[Final MD&A Report]
```

## 3. Core Component Analysis

### A. Data Ingestion & Processing
- **`src/data_loader.py`**: A robust module capable of parsing complex **SEC XBRL** data structures. It maps diverse tags (e.g., `SalesRevenueNet`, `Revenues`) to a standardized schema, ensuring consistency across different company reporting styles.
- **`src/kpi_calculator.py`**: The financial engine. It computes 14+ metrics, including specific ratios like **Debt-to-Equity** and **R&D Intensity**. It uniquely generates text-based interpretations (e.g., "TotalRevenue: $5.25B increased 15% year-over-year") which serves as high-quality context for the LLM.

### B. RAG & Retrieval
- **`src/rag_pipeline.py`** & **`src/vector_store.py`**: 
  - Implementation: Uses **ChromaDB** for persistent vector storage.
  - Embeddings: Supports hybrid mode. Defaults to local **HuggingFace** embeddings (`all-MiniLM-L6-v2`) for cost-efficiency. helping to avoid external API costs. Also supports **Gemini Embeddings** with a smart custom throttling implementation (batch_size=1) to navigate free-tier rate limits.
  - Strategy: Chunks both raw financial tables and calculated KPIs to ground the LLM.

### C. Generative Logic
- **`src/mda_generator.py`**: 
  - **Prompt Engineering**: Contains highly specific prompts for 5 report sections (Executive Summary, Revenue Trends, Profitability, Financial Position, Risk Factors).
  - **Resilience**: Implements an intelligent **Model Rotation** mechanism. If the main Gemini model hits a rate limit, it automatically fails over to alternative models (e.g., Flash -> Pro -> Vision).
  - **Fallback Strategy**: If all Gemini models are exhausted, it seamlessly fails over to **Groq (Llama 3 70B)** to ensure generation continuity.

### D. Agentic AI (`src/agents.py`)
This is the standout feature of the codebase. Instead of a single pass generation, it employs a multi-agent system:
1. **`RiskDetectionAgent`**: Proactively scans numerical trends for negatives (e.g., declining margins) to synthesize a "Risk Analysis" section that a standard happy-path prompt might miss.
2. **`CritiqueAgent`**: Acts as a reviewer. It reads the generated text and compares it against the provided metrics to flag hallucinations or vague claims.
3. **`ComparativeAnalysisAgent`**: Focuses on time-series analysis, comparing current period vs. prior period.

### E. Quality Assurance (`src/evaluator.py`)
A comprehensive "Guardrails" implementation that scores every generated report:
- **Factual Consistency**: Extracts numbers from text and matches them to the source dataframe within a 5% tolerance.
- **Citation Coverage**: Checks if claims are cited.
- **Financial Reasonableness**: Detects logic errors (e.g., a margin > 100%).
- **Scorecard**: Returns a 0-100% quality score used to grade the output.

## 4. Interfaces

- **`main.py`**: A unified CLI supporting all modes (`--sec`, `--agentic`, `--no-evaluation`).
- **`streamlit_app.py`**: A polished, demo-ready UI. Features real-time progress tracking, KPI visualization, and side-by-side report viewing with citations.
- **`src/api_service.py`**: A production-ready **FastAPI** backend.
  - Features: Asynchronous job management, unique Job IDs for progress tracking, and **Batch Generation** endpoints (`/batch/generate`) for processing multiple companies simultaneously.

## 5. Technology Stack
- **Language**: Python 3.10+
- **LLM**: Google Gemini (1.5 Flash/Pro)
- **Vector DB**: ChromaDB
- **Web Frameworks**: FastAPI (Backend), Streamlit (Frontend)
- **Data Science**: Pandas, NumPy
- **Testing**: Pytest

## 6. Conclusion
The `Hackathon` project is a feature-complete, architecturally sound solution. It effectively addresses the challenge of financial reporting by mitigating LLM limitations (hallucinations) through strict RAG pipelines and deterministic guardrails. The addition of Agentic workflows for self-correction places it in the "Advanced" category of AI applications.

## 7. Verification Log
*Analyzed on: 2026-01-02*
- **Codebase Integrity**: Verified. All 13 core source files present and functional.
- **Dependency Check**: `requirements.txt` aligns with imported modules.
- **Configuration**: `.env` handling via `pydantic-settings` is secure and robust.
