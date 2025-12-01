"""
NeuronMap REST API

Provides a comprehensive REST API for all NeuronMap functionality.
Supports analysis, visualization, model management, and more.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import traceback

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    # Graceful degradation if FastAPI not installed
    FastAPI = None
    HTTPException = None
    BaseModel = None

# Internal imports
from core.orchestrator import SystemOrchestrator
from config.config_manager import ConfigManager
from utils.error_handling import NeuronMapError, setup_error_handling
from utils.monitoring import setup_monitoring

# Set up logging
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses

class ProjectCreateRequest(BaseModel):
    """Request model for creating a project."""
    name: str
    description: Optional[str] = ""

class AnalysisRequest(BaseModel):
    """Request model for analysis endpoint."""
    project_id: Optional[str] = Field(None, description="Project ID to associate with")
    model_name: str = Field(..., description="Name of the model to analyze")
    input_text: str = Field(..., description="Text to analyze")
    layers: Optional[List[Union[int, str]]] = Field(None, description="Specific layers to analyze")
    analysis_type: str = Field("basic", description="Type of analysis to perform")
    config: Optional[Dict[str, Any]] = Field(None, description="Analysis configuration")


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    analysis_id: str = Field(..., description="Unique identifier for this analysis")
    status: str = Field(..., description="Analysis status")
    results: Optional[Dict[str, Any]] = Field(None, description="Analysis results")
    metadata: Dict[str, Any] = Field(..., description="Analysis metadata")
    created_at: datetime = Field(..., description="Creation timestamp")


class ModelInfo(BaseModel):
    """Model information response."""
    name: str
    type: str
    parameters: Optional[int] = None
    layers: Optional[int] = None
    supported_tasks: List[str]
    description: Optional[str] = None


class VisualizationRequest(BaseModel):
    """Request for creating visualizations."""
    analysis_id: str = Field(..., description="Analysis ID to visualize")
    plot_type: str = Field(..., description="Type of plot to create")
    config: Optional[Dict[str, Any]] = Field(
        None, description="Visualization configuration")


class JobStatus(BaseModel):
    """Background job status."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class NeuronMapAPI:
    """Main API class for NeuronMap REST service."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the API."""
        # Set up error handling and monitoring
        setup_error_handling()
        self.monitor = setup_monitoring()

        # Load configuration
        self.config_manager = ConfigManager()
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = self.config_manager.get_default_config()

        # Initialize System Orchestrator
        self.orchestrator = SystemOrchestrator()

        # In-memory storage for jobs (orchestrator handles results)
        self.jobs: Dict[str, JobStatus] = {}
        self.results: Dict[str, Any] = {}
        
        # Available models cache
        self.available_models: List[ModelInfo] = []
        self._load_available_models()

        logger.info("NeuronMap API initialized with System Orchestrator")

    def _load_available_models(self):
        """Load information about available models."""
        # This would typically load from a configuration file or database
        self.available_models = [
            ModelInfo(
                name="bert-base-uncased",
                type="huggingface",
                parameters=110_000_000,
                layers=12,
                supported_tasks=["text-analysis", "sentiment", "token-classification"],
                description="BERT base model, uncased"
            ),
            ModelInfo(
                name="gpt2",
                type="huggingface",
                parameters=117_000_000,
                layers=12,
                supported_tasks=["text-generation", "analysis"],
                description="GPT-2 base model"
            ),
            ModelInfo(
                name="roberta-base",
                type="huggingface",
                parameters=125_000_000,
                layers=12,
                supported_tasks=["text-analysis", "sentiment", "classification"],
                description="RoBERTa base model"
            )
        ]

    def create_job(self, job_type: str, **kwargs) -> str:
        """Create a new background job."""
        job_id = str(uuid.uuid4())
        job = JobStatus(
            job_id=job_id,
            status="pending",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.jobs[job_id] = job

        logger.info(f"Created job {job_id} of type {job_type}")
        return job_id

    def update_job(self,
                   job_id: str,
                   status: str,
                   progress: Optional[float] = None,
                   result: Optional[Dict[str,
                                         Any]] = None,
                   error: Optional[str] = None):
        """Update job status."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = status
            job.updated_at = datetime.now()

            if progress is not None:
                job.progress = progress
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error

            logger.info(f"Updated job {job_id}: status={status}, progress={progress}")

    async def run_analysis(self, request: AnalysisRequest, job_id: str):
        """Run analysis in background using Orchestrator."""
        try:
            self.update_job(job_id, "running", progress=0.0)

            # Ensure project exists
            project_id = request.project_id
            if not project_id:
                # Create default project if none provided
                project_id = self.orchestrator.create_project("Default Project", "Auto-generated project")
            
            self.update_job(job_id, "running", progress=0.1)

            # Submit analysis pipeline via Orchestrator (Async Task Queue)
            experiment_id = self.orchestrator.submit_analysis_pipeline(
                project_id=project_id,
                model_name=request.model_name,
                input_data=request.input_text,
                analysis_types=[request.analysis_type]
            )

            # The job is now "submitted" to the internal queue.
            # We can mark this API job as completed, returning the experiment_id
            # The client can then poll the experiment status.

            self.update_job(job_id, "completed", progress=1.0, result={
                'project_id': project_id,
                'experiment_id': experiment_id,
                'status': 'submitted_to_queue'
            })

            logger.info(f"Analysis submitted for job {job_id}, experiment_id: {experiment_id}")

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(f"Job {job_id} failed: {error_msg}")
            logger.error(traceback.format_exc())
            self.update_job(job_id, "failed", error=error_msg)

    async def create_visualization_job(self, request: VisualizationRequest, job_id: str):
        """Create visualization in background."""
        try:
            self.update_job(job_id, "running", progress=0.1)

            # We need project_id and experiment_id from the request
            # Assuming analysis_id in request maps to experiment_id for now
            # In a real update, we'd update VisualizationRequest model too
            
            # For this fix, we'll assume the user passes "project_id:experiment_id" as analysis_id
            if ":" in request.analysis_id:
                project_id, experiment_id = request.analysis_id.split(":")
            else:
                raise ValueError("Invalid analysis_id format. Use 'project_id:experiment_id'")

            self.update_job(job_id, "running", progress=0.3)

            # Use Orchestrator to generate visualization
            viz_path = self.orchestrator.generate_visualization(
                project_id, 
                experiment_id, 
                request.plot_type
            )

            self.update_job(job_id, "completed", progress=1.0, result={
                "visualization_id": job_id,
                "path": str(viz_path),
                "type": request.plot_type
            })

        except Exception as e:
            error_msg = f"Visualization failed: {str(e)}"
            logger.error(f"Visualization job {job_id} failed: {error_msg}")
            self.update_job(job_id, "failed", error=error_msg)

    def _process_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process results to make them JSON serializable."""
        import torch
        import numpy as np

        def convert_tensor(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensor(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensor(item) for item in obj]
            else:
                return obj

        return convert_tensor(results)


def _setup_cors_middleware(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic here
    return {"user_id": "anonymous"}

def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create FastAPI application."""
    if FastAPI is None:
        raise ImportError("FastAPI is required for the REST API. Install with: pip install fastapi uvicorn")

    api = NeuronMapAPI(config_path)

    app = FastAPI(
        title="NeuronMap API",
        description="REST API for neural network activation analysis",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    _setup_cors_middleware(app)

    # API Routes

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Welcome to NeuronMap API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }

    @app.get("/models", response_model=List[ModelInfo])
    async def list_models():
        """List available models."""
        return api.available_models

    @app.post("/analyze", response_model=Dict[str, str])
    async def create_analysis(
        request: AnalysisRequest,
        background_tasks: BackgroundTasks,
        user=Depends(get_current_user)
    ):
        """Create a new analysis job."""
        try:
            job_id = api.create_job("analysis", request=request.dict())

            background_tasks.add_task(api.run_analysis, request, job_id)

            return {
                "job_id": job_id,
                "status": "submitted",
                "message": "Analysis job submitted successfully"
            }

        except Exception as e:
            logger.error(f"Failed to create analysis: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/jobs/{job_id}", response_model=JobStatus)
    async def get_job_status(job_id: str):
        """Get job status."""
        if job_id not in api.jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        return api.jobs[job_id]

    @app.get("/results/{analysis_id}")
    async def get_analysis_results(analysis_id: str):
        """Get analysis results."""
        if analysis_id not in api.results:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return api.results[analysis_id]

    @app.get("/projects", response_model=List[Dict[str, Any]])
    async def list_projects():
        """List all projects."""
        return api.orchestrator.list_projects()

    @app.post("/projects", response_model=Dict[str, str])
    async def create_project(request: ProjectCreateRequest):
        """Create a new project."""
        project_id = api.orchestrator.create_project(request.name, request.description)
        return {"project_id": project_id, "message": "Project created successfully"}

    @app.post("/visualize", response_model=Dict[str, str])
    async def create_visualization(
        request: VisualizationRequest,
        background_tasks: BackgroundTasks
    ):
        """Create visualization from analysis results."""
        # if request.analysis_id not in api.results:
        #     raise HTTPException(status_code=404, detail="Analysis not found")

        try:
            job_id = api.create_job("visualization", request=request.dict())

            background_tasks.add_task(
                api.create_visualization_job,
                request,
                job_id
            )

            return {
                "job_id": job_id,
                "status": "submitted",
                "message": "Visualization job submitted successfully"
            }

        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/results/{project_id}/{experiment_id}")
    async def get_experiment_results(project_id: str, experiment_id: str):
        """Get experiment results."""
        experiment = api.orchestrator.project_manager.get_experiment(project_id, experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
            
        # In a real app, we might want to load the full results from the file
        # For now, return metadata
        return experiment

    @app.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        """Upload file for analysis."""
        try:
            upload_dir = Path("./uploads")
            upload_dir.mkdir(exist_ok=True)

            file_path = upload_dir / file.filename

            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            return {
                "filename": file.filename,
                "size": len(content),
                "message": "File uploaded successfully"
            }

        except Exception as e:
            logger.error(f"File upload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket, client_id: str):
        """WebSocket endpoint for real-time updates."""
        await websocket.accept()

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)

                await websocket.send_text(json.dumps({
                    "type": "response",
                    "client_id": client_id,
                    "message": f"Received: {message}"
                }))

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await websocket.close()

    app.state.api = api

    return app

# Additional methods for NeuronMapAPI

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuronMap API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    app = create_app(config_path=args.config)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port
    )
