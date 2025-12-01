"""
NeuronMap Web Server
===================

Serves the web frontend and API endpoints.
Integrates FastAPI with Jinja2 templates.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, Depends, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Add src to path
sys.path.insert(0, os.path.abspath("."))

from src.api.rest_api import NeuronMapAPI, AnalysisRequest, ProjectCreateRequest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_server")

# Initialize API Logic
# We reuse the logic from rest_api.py
api_logic = NeuronMapAPI()

app = FastAPI(
    title="NeuronMap Web Interface",
    description="Web interface for NeuronMap",
    version="1.0.0"
)

# Mount Static Files
static_dir = Path("web/static")
if not static_dir.exists():
    logger.warning(f"Static directory not found: {static_dir}")
    # Create dummy if not exists to avoid crash
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount Outputs Directory
outputs_dir = Path("data/outputs")
if not outputs_dir.exists():
    os.makedirs(outputs_dir, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

# Setup Templates
templates_dir = Path("web/templates")
if not templates_dir.exists():
    logger.error(f"Templates directory not found: {templates_dir}")
    raise FileNotFoundError("Templates directory not found")

templates = Jinja2Templates(directory=str(templates_dir))

# ------------------------------------------------------------------
# Page Routes (Frontend)
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analysis", response_class=HTMLResponse)
async def analysis(request: Request):
    """Analysis configuration page."""
    return templates.TemplateResponse("analysis.html", {"request": request})

@app.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    """Results browser."""
    return templates.TemplateResponse("results.html", {"request": request})

@app.get("/visualization", response_class=HTMLResponse)
async def visualization(request: Request):
    """Visualization studio."""
    return templates.TemplateResponse("visualization.html", {"request": request})

@app.get("/multi_model", response_class=HTMLResponse)
async def multi_model(request: Request):
    """Multi-model comparison."""
    return templates.TemplateResponse("multi_model.html", {"request": request})

@app.get("/advanced_analytics", response_class=HTMLResponse)
async def advanced_analytics_page(request: Request):
    """Advanced analytics tools."""
    return templates.TemplateResponse("advanced_analytics.html", {"request": request})

@app.get("/analysis_zoo", response_class=HTMLResponse)
async def analysis_zoo(request: Request):
    """Analysis Zoo."""
    return templates.TemplateResponse("analysis_zoo.html", {"request": request})

# ------------------------------------------------------------------
# API Routes (Backend)
# ------------------------------------------------------------------

@app.get("/api/stats")
async def get_stats():
    """Get system statistics for dashboard."""
    # In a real system, query the database/orchestrator
    # For now, return dummy/calculated data
    projects = api_logic.orchestrator.list_projects()
    total_analyses = 0
    # We would need to count experiments across projects
    
    return {
        "total_analyses": len(projects) * 5, # Mock
        "models_analyzed": 3,
        "layers_processed": 1250,
        "visualizations_created": 42
    }

@app.get("/api/recent-activity")
async def get_recent_activity():
    """Get recent activity log."""
    return {
        "activities": [
            {"icon": "check", "color": "success", "message": "Analysis completed: GPT-2 Layer 5", "timestamp": "2 mins ago"},
            {"icon": "play", "color": "primary", "message": "Started analysis: BERT Sentiment", "timestamp": "15 mins ago"},
            {"icon": "plus", "color": "info", "message": "New project created: Language Circuits", "timestamp": "1 hour ago"}
        ]
    }

@app.get("/api/models")
async def get_models():
    """Get available models."""
    # Wrap the list in a dict as expected by frontend
    return {"models": [m.name for m in api_logic.available_models]}

@app.get("/api/list-layers")
async def list_layers(model: str):
    """List layers for a given model."""
    # In a real implementation, we would load the model config or use the adapter
    # For now, return dummy layers based on model name
    layers = []
    if "gpt2" in model:
        layers = [f"transformer.h.{i}" for i in range(12)]
    elif "bert" in model:
        layers = [f"bert.encoder.layer.{i}" for i in range(12)]
    else:
        layers = [f"layer_{i}" for i in range(10)]
    return {"layers": layers}

@app.post("/api/analyze")
async def run_analysis_form(
    model: str = Form(...),
    customModel: str = Form(None),
    device: str = Form("auto"),
    questionsText: str = Form(None),
    questions: UploadFile = File(None),
    targetLayers: str = Form(None),
    advanced: bool = Form(False),
    visualize: bool = Form(True)
):
    """Handle analysis form submission."""
    
    # Determine model name
    model_name = customModel if model == "custom" else model
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")

    # Determine input text
    input_text = ""
    if questionsText:
        input_text = questionsText
    elif questions:
        content = await questions.read()
        input_text = content.decode("utf-8")
    
    if not input_text:
        raise HTTPException(status_code=400, detail="Input text is required")

    # Parse layers
    layers = []
    if targetLayers:
        layers = [l.strip() for l in targetLayers.split(",") if l.strip()]

    # Create request object
    request = AnalysisRequest(
        model_name=model_name,
        input_text=input_text,
        layers=layers if layers else None,
        analysis_type="advanced" if advanced else "basic",
        config={"device": device, "visualize": visualize}
    )

    # Submit job
    job_id = api_logic.create_job("analysis", request=request.dict())
    
    # Run in background (using the existing async method)
    # Note: In production, use BackgroundTasks
    await api_logic.run_analysis(request, job_id)
    
    # The run_analysis method in rest_api.py returns the experiment_id in the result
    # We need to return { "analysis_id": job_id } to match frontend
    return {"analysis_id": job_id, "status": "submitted"}

@app.get("/api/analysis-status/{job_id}")
async def get_analysis_status(job_id: str):
    """Get status of an analysis job."""
    if job_id not in api_logic.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = api_logic.jobs[job_id]
    
    # Map internal status to frontend expected format
    response = {
        "progress": job.progress * 100 if job.progress else 0,
        "message": f"Status: {job.status}",
        "completed": job.status == "completed",
        "error": job.error
    }
    
    if job.status == "completed" and job.result:
        # If completed, we might want to fetch the actual experiment results
        # The job result contains 'experiment_id'
        experiment_id = job.result.get("experiment_id")
        project_id = job.result.get("project_id")
        
        if experiment_id and project_id:
            # Check experiment status in orchestrator
            experiment = api_logic.orchestrator.project_manager.get_experiment(project_id, experiment_id)
            if experiment:
                exp_status = experiment.get("status")
                if exp_status == "running":
                    response["completed"] = False
                    response["message"] = "Processing in queue..."
                    response["progress"] = 50
                elif exp_status == "completed":
                    response["completed"] = True
                    response["message"] = "Analysis complete!"
                    response["progress"] = 100
                    response["experiment_id"] = experiment_id
                    response["project_id"] = project_id
                    response["results"] = {
                        "model": experiment.get("config", {}).get("model_name"),
                        "question_count": 1, # Placeholder
                        "layer_count": 12, # Placeholder
                        "duration": "5s" # Placeholder
                    }
                elif exp_status == "failed":
                    response["completed"] = False
                    response["error"] = experiment.get("error", "Unknown error")
    
    return response

@app.post("/api/cancel-analysis/{job_id}")
async def cancel_analysis(job_id: str):
    """Cancel an analysis job."""
    # Not implemented in orchestrator yet
    return {"status": "cancelled"}

@app.get("/api/download-results/{job_id}")
async def download_results(job_id: str):
    """Download results."""
    # Placeholder
    return {"message": "Download not implemented"}

@app.post("/api/visualize")
async def create_visualization(
    dataSource: str = Form(...),
    visualizationType: str = Form(...),
    analysisId: str = Form(None),
    layerFilter: str = Form(None),
    includeStatistics: bool = Form(True),
    highResolution: bool = Form(False),
    interactiveMode: bool = Form(True)
):
    """Create visualization."""
    
    # Determine project_id and experiment_id
    # For now, we assume analysisId is passed as "project_id:experiment_id" or just "experiment_id"
    # If just experiment_id, we might need to search for it.
    # But the frontend currently doesn't have a clear way to pass project_id.
    # Let's assume the user pastes "project_id:experiment_id" or we search.
    
    project_id = None
    experiment_id = None
    
    if dataSource == "analysis_id" and analysisId:
        if ":" in analysisId:
            project_id, experiment_id = analysisId.split(":")
        else:
            # Try to find the experiment in all projects
            projects = api_logic.orchestrator.list_projects()
            for p in projects:
                # This is inefficient but works for now
                # We need a way to look up experiment -> project
                # Let's assume the first project for now or search
                # For this demo, let's just require project_id:experiment_id
                # OR, we can search the project manager if it supports it.
                pass
            
            # Fallback: if we can't find it, raise error
            # But wait, run_analysis returns just experiment_id (UUID).
            # The user won't know the project_id easily.
            # We should probably search.
            
            # Let's try to find it in the default project first
            # Or iterate all projects.
            found = False
            for p in api_logic.orchestrator.list_projects():
                exp = api_logic.orchestrator.project_manager.get_experiment(p['id'], analysisId)
                if exp:
                    project_id = p['id']
                    experiment_id = analysisId
                    found = True
                    break
            
            if not found:
                raise HTTPException(status_code=404, detail="Analysis ID not found in any project")

    elif dataSource == "recent":
        # Get most recent experiment
        # Not implemented yet
        raise HTTPException(status_code=501, detail="Recent analysis not implemented")
    
    if not project_id or not experiment_id:
         raise HTTPException(status_code=400, detail="Invalid analysis ID")

    try:
        # Generate visualization
        # The orchestrator.generate_visualization returns a path
        viz_path = api_logic.orchestrator.generate_visualization(
            project_id, 
            experiment_id, 
            visualizationType
        )
        
        # Convert path to URL
        # viz_path is absolute. We need to make it relative to data/outputs
        # The plugin saves to data/outputs/plugins/visualization/...
        
        rel_path = os.path.relpath(viz_path, os.path.abspath("data/outputs"))
        viz_url = f"/outputs/{rel_path}"
        
        # Construct response
        # If it's an HTML dashboard
        if str(viz_path).endswith(".html"):
            return {
                "visualization_id": f"viz_{datetime.now().timestamp()}",
                "dashboard_url": viz_url
            }
        else:
            # It's an image
            return {
                "visualization_id": f"viz_{datetime.now().timestamp()}",
                "plots": [
                    {
                        "type": "image",
                        "url": viz_url,
                        "title": f"{visualizationType} Visualization"
                    }
                ]
            }

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects")
async def list_projects():
    return api_logic.orchestrator.list_projects()

# ------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------

def run_server(host="0.0.0.0", port=8000, reload=False):
    uvicorn.run("src.web_server:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    run_server(reload=True)
