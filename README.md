# 📊 AI-Powered MD&A Generator (Finance Hackathon Assignment F1)

> **Automated Management Discussion & Analysis (MD&A) Report Generation using RAG, LLMs, and Agentic AI**

---

## 📝 Problem Statement

**Domain**: Finance (Assignment F1)
**Task**: Automated MD&A Draft from Financials (RAG + Summarization)
**Objective**: Generate a professional first-draft MD&A (Management Discussion & Analysis) narrative from tabular financial statement extracts.

### The Challenge
Management Discussion & Analysis (MD&A) sections in financial reports are critical for investors but require significant manual effort to draft. Financial analysts must:
1.  **Analyze Data**: Review extensive financial statement data (Income Statement, Balance Sheet).
2.  **Calculate Metrics**: Compute Year-over-Year (YoY) and Quarter-over-Quarter (QoQ) KPIs.
3.  **Identify Trends**: Spot significant changes, risks, and financial health indicators.
4.  **Draft Narrative**: Synthesize findings into coherent, compliant narratives.
5.  **Fact-Checking**: Ensure every claim is backed by the source data.

This manual process is time-consuming (taking hours to days), error-prone, and inconsistent.

### The Solution
We have built an **AI-Powered MD&A Generator** that automates this entire workflow. It ingests raw financial data and produces a structured, cited, and fact-checked MD&A draft in minutes.

**Key capabilities:**
*   **Data Ingestion**: Loads SEC financial data (JSON) or standard CSVs.
*   **Financial Engine**: Automatically calculates 14+ key financial ratios and growth metrics.
*   **RAG Architecture**: Uses Retrieval Augmented Generation to ground LLM outputs in actual data, minimizing hallucinations.
*   **Agentic AI**: A multi-agent system where specialized agents (Risk, Critique, Comparative) work together to refine the analysis.
*   **Guardrails**: A strict evaluation framework that verifies factual consistency and financial reasonableness.

---

## 🛠️ Technical Implementation

### Architecture Overview

```mermaid
graph TD
    Data[Data Source (SEC/CSV)] --> Loader[Data Loader]
    Loader --> KPI[KPI Calculator]
    Loader --> Chunker[Document Chunker]
    KPI --> Chunker
    Chunker --> Vector[ChromaDB Vector Store]
    
    subgraph "Agentic GenAI Core"
        Vector --> RAG[RAG Pipeline]
        RAG --> Generator[Base Generator]
        Generator --> Agents[Agent Orchestrator]
        Agents --> Risk[Risk Agent]
        Agents --> Critique[Critique Agent]
    end
    
    Risk --> Refuter[Refinement & Integration]
    Critique --> Refuter
    Refuter --> Guardrails[Guardrails Evaluator]
    Guardrails --> Report[Final MD&A Report]
```

### Technology Stack
*   **Language**: Python 3.10+
*   **LLM**: Google Gemini (1.5 Flash / Pro) via `google-generativeai`
*   **Embeddings**: Local HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`) via ChromaDB (No API costs for embeddings!)
*   **Vector Store**: ChromaDB (Persistent local storage)
*   **Data Processing**: Pandas, NumPy
*   **API/Backend**: FastAPI (Async job processing)
*   **Frontend**: Streamlit (Interactive dashboard)
*   **Quality Control**: Custom Guardrails & Pydantic validation

---

## ✨ Key Features & Innovation

### 1. Agentic AI Workflow
We go beyond simple "chat with data". When `--agentic` mode is enabled:
*   **Risk Detection Agent**: Proactively scans the dataframe for declining metrics (e.g., shrinking margins) and writes a specific "Risk Factors" section.
*   **Critique Agent**: Acts as an automated editor. It reviews the generated draft against the calculated KPIs to flag factual errors or vague claims.
*   **Comparative Agent**: Analyzes multi-period trends to provide decent historical context.

### 2. Guardrails & Evaluation Structure
Financial reports must be accurate. Our `evaluator.py` module performs 6 automated checks on every output:
*   **Factual Consistency**: Extracts numbers from the text and compares them to the source data (5% tolerance).
*   **Citation Coverage**: Ensures strict referencing of source chunks.
*   **Financial Reasonableness**: Flags logical impossibilities (e.g., >100% profit margin).
*   **Metric Accuracy**: Verifies that calculated YoY % changes match the narrative.

### 3. Rate Limit Resilience
Designed for hackathon constraints:
*   **Local Embeddings**: We use HuggingFace embeddings locally strictly to save bandwidth and API quotas.
*   **Model Rotation**: If Gemini 1.5 Flash hits a rate limit, the system automatically fails over to Gemini 1.5 Pro or other available models.
*   **Provider Fallback (Groq)**: If ALL Gemini models are exhausted, the system automatically switches to **Groq (Llama 3)** for uninterrupted generation.

---

## 🚀 Installation & Usage

### Prerequisites
*   Python 3.10+
*   Google Gemini API Key

### Setup
```bash
# 1. Clone & Enter
git clone <repo-url>
cd Hackathon

# 2. Virtual Env
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Environment
copy .env.example .env
# Edit .env and add: GEMINI_API_KEY=your_key_here
```

### Running the Application

**Option A: Streamlit UI (Deployable Version)**
This is the main application allowing file uploads and API key configuration.
```bash
streamlit run streamlit_app.py
```
Features:
- 📁 Upload JSON financial data files directly in the UI
- 🔑 Configure Gemini/Groq API keys in the sidebar
- 🚀 No need for `.env` file - all configuration done in browser

**Option B: Legacy UI (Local Development)**
The original development version that uses local files and `.env`.
```bash
streamlit run legacy_app.py
```

**Option C: Command Line (Fast)**
Generate a report for the sample data.
```bash
python main.py --sample --agentic
```


---

## 📊 Evaluation Metrics & Results

We evaluate our system using a composite Quality Score (0-100%):
*   **Factuality**: >95% target
*   **Completeness**: >4/5 sections present
*   **Citation Density**: >0.5 citations per claim

*Metrics are automatically calculated and appended to every generated report.*

---

## 📁 Repository Structure

*   `main.py`: CLI Entry point
*   `streamlit_app.py`: Web Interface
*   `src/`: Core source code
    *   `agents.py`: Agentic AI logic (Critique, Risk, Comparative)
    *   `mda_generator.py`: LLM Prompting & Logic
    *   `evaluator.py`: Guardrails Implementation
    *   `kpi_calculator.py`: Financial Math Engine
    *   `data_loader.py`: SEC/CSV Data Parsing
*   `notebooks/`: Reproducible Jupyter notebooks
*   `tests/`: Pytest suite

---

## 📝 Submission Checklist Compliance
- [x] **Updated README.md**: Comprehensive problem & solution description.
- [x] **Reproducible Code**: Full `requirements.txt` and `main.py` provided.
- [x] **Innovation**: Implemented Agentic workflow & Custom Guardrails.
- [x] **AI Utilization**: Uses RAG, embeddings, and Gemini Pro/Flash.
- [x] **Impact**: Automates a critical financial workflow.
- [x] **Presentation**: Streamlit UI + Markdown Reports.
- [x] **Data Requirement**: Supports Kaggle SEC dataset (Finance F1).

---

**License**: MIT