"""
FastAPI Service for MD&A Generator
RESTful API with async support for MD&A generation
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from .data_loader import DataLoader
from .kpi_calculator import KPICalculator
from .chunker import DocumentChunker
from .rag_pipeline import RAGPipeline
from .mda_generator import MDAGenerator
from .schemas import MDADocument
from .evaluator import MDAGuardrails, EvaluationReport

# Initialize FastAPI app
app = FastAPI(
    title="MD&A Generator API",
    description="AI-powered Management Discussion & Analysis report generation",
    version="1.0.0"
)

# In-memory storage for async job status
job_store: Dict[str, Dict[str, Any]] = {}


# Pydantic models for API
class GenerationRequest(BaseModel):
    company_name: str = Field(..., description="Company name for MD&A report")
    ticker: Optional[str] = Field(None, description="Company ticker symbol")
    use_sec_data: bool = Field(False, description="Use SEC dataset if available")
    enable_evaluation: bool = Field(True, description="Run guardrail evaluation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "company_name": "TechCorp Inc.",
                "ticker": "AAPL",
                "use_sec_data": False,
                "enable_evaluation": True
            }
        }


class GenerationResponse(BaseModel):
    job_id: str
    status: str
    company_name: str
    created_at: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing",
                "company_name": "TechCorp Inc.",
                "created_at": "2024-01-02T12:00:00Z"
            }
        }


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    company_name: str
    created_at: str
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: int
    current_step: str
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "company_name": "TechCorp Inc.",
                "created_at": "2024-01-02T12:00:00Z",
                "updated_at": "2024-01-02T12:05:00Z",
                "completed_at": "2024-01-02T12:05:00Z",
                "progress": 100,
                "current_step": "completed",
                "error": None
            }
        }


class CompletedJobResponse(JobStatusResponse):
    report_path: str
    sections_generated: int
    total_citations: int
    evaluation_report: Optional[Dict[str, Any]] = None


class BatchGenerationRequest(BaseModel):
    companies: List[str] = Field(..., description="List of company names or tickers")
    use_sec_data: bool = Field(True, description="Use SEC dataset if available")
    enable_evaluation: bool = Field(True, description="Run guardrail evaluation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "companies": ["AAPL", "MSFT", "GOOGL"],
                "use_sec_data": True,
                "enable_evaluation": True
            }
        }


class BatchGenerationResponse(BaseModel):
    batch_id: str
    job_ids: List[str]
    status: str
    total_jobs: int
    created_at: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    services: Dict[str, str]


# Background task function
async def generate_mda_background(
    job_id: str,
    request: GenerationRequest
) -> None:
    """Background task to generate MD&A report"""
    
    def update_job(progress: int, step: str):
        job_store[job_id]['progress'] = progress
        job_store[job_id]['current_step'] = step
        job_store[job_id]['updated_at'] = datetime.now().isoformat()
    
    try:
        company = request.company_name
        update_job(10, "Loading financial data")
        
        # Step 1: Load data
        loader = DataLoader()
        
        if request.use_sec_data and request.ticker:
            # Load SEC data
            df = loader.load_sec_data(request.ticker)
            if df.empty:
                job_store[job_id]['status'] = 'failed'
                job_store[job_id]['error'] = f"No SEC data found for ticker: {request.ticker}"
                job_store[job_id]['updated_at'] = datetime.now().isoformat()
                return
        else:
            # Use sample data
            df = loader.load_sample_data()
        
        if 'Company' in df.columns and len(df) > 0:
            company = df['Company'].iloc[0]
        
        await asyncio.sleep(0.1)  # Yield control
        update_job(25, "Calculating KPIs")
        
        # Step 2: Calculate KPIs
        kpi_calc = KPICalculator(df)
        kpis = kpi_calc.calculate_all_kpis()
        
        await asyncio.sleep(0.1)
        update_job(40, "Creating document chunks")
        
        # Step 3: Create chunks
        chunker = DocumentChunker()
        chunks = chunker.chunk_dataframe(df, company)
        chunks.extend(chunker.chunk_kpis(kpis, company))
        
        await asyncio.sleep(0.1)
        update_job(55, "Indexing in vector store")
        
        # Step 4: Index chunks
        rag = RAGPipeline()
        rag_indexed = rag.index_data(chunks)
        
        await asyncio.sleep(0.1)
        update_job(70, "Generating MD&A sections")
        
        # Step 5: Generate MD&A
        generator = MDAGenerator(rag)
        mda_doc = generator.generate(company, kpis)
        
        await asyncio.sleep(0.1)
        update_job(85, "Evaluating output" if request.enable_evaluation else "Saving report")
        
        # Step 6: Evaluate (if enabled)
        evaluation_report = None
        if request.enable_evaluation:
            guardrails = MDAGuardrails(kpis, df)
            eval_report = guardrails.evaluate_document(mda_doc)
            evaluation_report = {
                'overall_score': eval_report.overall_score,
                'factuality_score': eval_report.factuality_score,
                'citation_score': eval_report.citation_score,
                'consistency_score': eval_report.consistency_score,
                'completeness_score': eval_report.completeness_score,
                'passed_all_guardrails': eval_report.passed_all_guardrails,
                'recommendations': eval_report.recommendations,
                'guardrail_results': [
                    {
                        'name': g.name,
                        'passed': g.passed,
                        'score': g.score,
                        'message': g.message
                    } for g in eval_report.guardrail_results
                ]
            }
        
        # Step 7: Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mda_{company.replace(' ', '_')}_{timestamp}.md"
        report_path = generator.save_report(mda_doc)
        
        await asyncio.sleep(0.1)
        update_job(100, "completed")
        
        # Store results
        job_store[job_id]['status'] = 'completed'
        job_store[job_id]['completed_at'] = datetime.now().isoformat()
        job_store[job_id]['report_path'] = str(report_path)
        job_store[job_id]['sections_generated'] = 5
        job_store[job_id]['total_citations'] = sum(
            len(s.citations) for s in [
                mda_doc.executive_summary, mda_doc.revenue_trends,
                mda_doc.profitability_analysis, mda_doc.financial_position,
                mda_doc.risk_factors
            ] if s
        )
        job_store[job_id]['evaluation_report'] = evaluation_report
        job_store[job_id]['updated_at'] = datetime.now().isoformat()
        
    except Exception as e:
        job_store[job_id]['status'] = 'failed'
        job_store[job_id]['error'] = str(e)
        job_store[job_id]['updated_at'] = datetime.now().isoformat()


# API Endpoints

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        services={
            "data_loader": "ready",
            "kpi_calculator": "ready",
            "rag_pipeline": "ready",
            "llm_generator": "ready",
            "evaluator": "ready"
        }
    )


@app.post("/generate", response_model=GenerationResponse, status_code=202)
async def generate_mda(request: GenerationRequest, background_tasks: BackgroundTasks):
    """
    Generate MD&A report for a company
    
    This endpoint initiates an asynchronous generation job and returns immediately
    with a job_id that can be used to track progress.
    """
    job_id = str(uuid.uuid4())
    
    job_store[job_id] = {
        'job_id': job_id,
        'status': 'processing',
        'company_name': request.company_name,
        'created_at': datetime.now().isoformat(),
        'updated_at': None,
        'completed_at': None,
        'progress': 0,
        'current_step': 'initializing',
        'error': None
    }
    
    # Start background task
    background_tasks.add_task(generate_mda_background, job_id, request)
    
    return GenerationResponse(
        job_id=job_id,
        status="processing",
        company_name=request.company_name,
        created_at=job_store[job_id]['created_at']
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of an MD&A generation job
    
    Returns current progress, current step, and other metadata.
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_store[job_id]
    
    return JobStatusResponse(
        job_id=job['job_id'],
        status=job['status'],
        company_name=job['company_name'],
        created_at=job['created_at'],
        updated_at=job.get('updated_at'),
        completed_at=job.get('completed_at'),
        progress=job['progress'],
        current_step=job['current_step'],
        error=job.get('error')
    )


@app.get("/jobs/{job_id}/result", response_model=CompletedJobResponse)
async def get_job_result(job_id: str):
    """
    Get the completed MD&A generation result
    
    Only available for completed jobs. Includes report metadata and evaluation results.
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_store[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job['status']}"
        )
    
    return CompletedJobResponse(
        job_id=job['job_id'],
        status=job['status'],
        company_name=job['company_name'],
        created_at=job['created_at'],
        updated_at=job.get('updated_at'),
        completed_at=job.get('completed_at'),
        progress=job['progress'],
        current_step=job['current_step'],
        error=job.get('error'),
        report_path=job['report_path'],
        sections_generated=job['sections_generated'],
        total_citations=job['total_citations'],
        evaluation_report=job.get('evaluation_report')
    )


@app.get("/jobs/{job_id}/download")
async def download_report(job_id: str):
    """
    Download the generated MD&A markdown report
    
    Returns the markdown file directly.
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_store[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job['status']}"
        )
    
    report_path = job.get('report_path')
    if not report_path:
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(
        path=report_path,
        media_type='text/markdown',
        filename=report_path.split('/')[-1]
    )


@app.post("/batch/generate", response_model=BatchGenerationResponse, status_code=202)
async def generate_batch_mda(request: BatchGenerationRequest, background_tasks: BackgroundTasks):
    """
    Generate MD&A reports for multiple companies in batch
    
    Creates multiple generation jobs, one for each company.
    """
    batch_id = str(uuid.uuid4())
    job_ids = []
    
    for company in request.companies:
        job_id = str(uuid.uuid4())
        job_ids.append(job_id)
        
        job_store[job_id] = {
            'job_id': job_id,
            'status': 'processing',
            'company_name': company,
            'created_at': datetime.now().isoformat(),
            'updated_at': None,
            'completed_at': None,
            'progress': 0,
            'current_step': 'initializing',
            'error': None,
            'batch_id': batch_id
        }
        
        # Create generation request for this company
        gen_request = GenerationRequest(
            company_name=company,
            ticker=company,
            use_sec_data=request.use_sec_data,
            enable_evaluation=request.enable_evaluation
        )
        
        # Start background task
        background_tasks.add_task(generate_mda_background, job_id, gen_request)
    
    return BatchGenerationResponse(
        batch_id=batch_id,
        job_ids=job_ids,
        status="processing",
        total_jobs=len(job_ids),
        created_at=datetime.now().isoformat()
    )


@app.get("/batch/{batch_id}")
async def get_batch_status(batch_id: str):
    """
    Get status of all jobs in a batch
    
    Returns aggregated status across all jobs.
    """
    batch_jobs = [
        job for job in job_store.values()
        if job.get('batch_id') == batch_id
    ]
    
    if not batch_jobs:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    total = len(batch_jobs)
    completed = sum(1 for job in batch_jobs if job['status'] == 'completed')
    failed = sum(1 for job in batch_jobs if job['status'] == 'failed')
    processing = sum(1 for job in batch_jobs if job['status'] == 'processing')
    
    overall_status = 'processing' if processing > 0 else ('completed' if failed == 0 else 'partial')
    
    return {
        'batch_id': batch_id,
        'status': overall_status,
        'total_jobs': total,
        'completed': completed,
        'failed': failed,
        'processing': processing,
        'jobs': [
            {
                'job_id': job['job_id'],
                'company_name': job['company_name'],
                'status': job['status'],
                'progress': job['progress']
            } for job in batch_jobs
        ]
    }


@app.get("/jobs")
async def list_all_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    List all generation jobs
    
    Supports filtering by status and limiting the number of results.
    """
    jobs = list(job_store.values())
    
    if status:
        jobs = [job for job in jobs if job['status'] == status]
    
    # Sort by created_at descending
    jobs.sort(key=lambda x: x['created_at'], reverse=True)
    
    return {
        'total': len(jobs),
        'jobs': jobs[:limit]
    }


@app.get("/metrics")
async def get_metrics():
    """
    Get system metrics and statistics

    Returns summary statistics about generated reports and system performance.
    """
    all_jobs = list(job_store.values())
    
    total_jobs = len(all_jobs)
    completed_jobs = [j for j in all_jobs if j['status'] == 'completed']
    failed_jobs = [j for j in all_jobs if j['status'] == 'failed']
    
    avg_score = 0.0
    if completed_jobs:
        scores = [j.get('evaluation_report', {}).get('overall_score', 0) for j in completed_jobs]
        if scores:
            avg_score = sum(scores) / len(scores)
    
    total_citations = sum(j.get('total_citations', 0) for j in completed_jobs)
    
    return {
        'total_jobs': total_jobs,
        'completed_jobs': len(completed_jobs),
        'failed_jobs': len(failed_jobs),
        'success_rate': len(completed_jobs) / total_jobs if total_jobs > 0 else 0.0,
        'average_quality_score': avg_score,
        'total_citations_generated': total_citations,
        'timestamp': datetime.now().isoformat()
    }


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server"""
    uvicorn.run(
        "src.api_service:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    run_api_server()