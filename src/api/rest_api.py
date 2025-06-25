"""
NeuronMap REST API

Provides a comprehensive REST API for all NeuronMap functionality.
Supports analysis, visualization, model management, and more.
"""

import asyncio
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
    from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    # Graceful degradation if FastAPI not installed
    FastAPI = None
    HTTPException = None
    BaseModel = None

# Internal imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.neuron_map import NeuronMap
from config.config_manager import ConfigManager
from utils.error_handling import NeuronMapError, setup_error_handling
from utils.monitoring import setup_monitoring
from visualization.advanced_plots import create_interactive_dashboard

# Set up logging
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class AnalysisRequest(BaseModel):
    """Request model for analysis endpoint."""
    model_name: str = Field(..., description="Name of the model to analyze")
    input_text: str = Field(..., description="Text to analyze")
    layers: Optional[List[int]] = Field(None, description="Specific layers to analyze")
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
    config: Optional[Dict[str, Any]] = Field(None, description="Visualization configuration")

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

        # Initialize NeuronMap core
        self.neuron_map = NeuronMap(config=self.config)

        # In-memory storage for jobs and results
        self.jobs: Dict[str, JobStatus] = {}
        self.results: Dict[str, Dict[str, Any]] = {}

        # Available models cache
        self.available_models: List[ModelInfo] = []
        self._load_available_models()

        logger.info("NeuronMap API initialized")

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

    def update_job(self, job_id: str, status: str, progress: Optional[float] = None,
                   result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
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
        """Run analysis in background."""
        try:
            self.update_job(job_id, "running", progress=0.0)

            # Load model
            self.update_job(job_id, "running", progress=0.2)
            model_config = {
                'name': request.model_name,
                'type': 'huggingface'  # Default type
            }

            if request.config:
                model_config.update(request.config.get('model', {}))

            model = self.neuron_map.load_model(model_config)

            # Prepare analysis parameters
            self.update_job(job_id, "running", progress=0.4)
            analysis_params = {
                'layers': request.layers or [0, 6, 11],  # Default layers
                'include_attention': True,
                'include_hidden_states': True
            }

            if request.config:
                analysis_params.update(request.config.get('analysis', {}))

            # Run analysis
            self.update_job(job_id, "running", progress=0.6)

            if request.analysis_type == "sentiment":
                results = self.neuron_map.analyze_sentiment(
                    model=model,
                    texts=[request.input_text],
                    **analysis_params
                )
            elif request.analysis_type == "interpretability":
                results = self.neuron_map.analyze_interpretability(
                    model=model,
                    input_text=request.input_text,
                    **analysis_params
                )
            else:  # basic analysis
                results = self.neuron_map.generate_activations(
                    text=request.input_text,
                    **analysis_params
                )

            # Process and store results
            self.update_job(job_id, "running", progress=0.9)

            # Convert tensors to lists for JSON serialization
            processed_results = self._process_results_for_json(results)

            # Store results
            analysis_id = str(uuid.uuid4())
            self.results[analysis_id] = {
                'analysis_id': analysis_id,
                'request': request.dict(),
                'results': processed_results,
                'metadata': {
                    'model_name': request.model_name,
                    'analysis_type': request.analysis_type,
                    'layers_analyzed': analysis_params['layers'],
                    'created_at': datetime.now().isoformat()
                }
            }

            # Complete job
            self.update_job(job_id, "completed", progress=1.0, result={'analysis_id': analysis_id})

            logger.info(f"Analysis completed for job {job_id}, analysis_id: {analysis_id}")

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(f"Job {job_id} failed: {error_msg}")
            logger.error(traceback.format_exc())
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

def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create FastAPI application."""
    if FastAPI is None:
        raise ImportError("FastAPI is required for the REST API. Install with: pip install fastapi uvicorn")

    # Initialize API
    api = NeuronMapAPI(config_path)

    # Create FastAPI app
    app = FastAPI(
        title="NeuronMap API",
        description="REST API for neural network activation analysis",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Authentication (optional)
    security = HTTPBearer(auto_error=False)

    def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Placeholder for authentication."""
        # Implement your authentication logic here
        return {"user_id": "anonymous"}

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
        user = Depends(get_current_user)
    ):
        """Create a new analysis job."""
        try:
            job_id = api.create_job("analysis", request=request.dict())

            # Start analysis in background
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

    @app.post("/visualize")
    async def create_visualization(
        request: VisualizationRequest,
        background_tasks: BackgroundTasks
    ):
        """Create visualization from analysis results."""
        if request.analysis_id not in api.results:
            raise HTTPException(status_code=404, detail="Analysis not found")

        try:
            job_id = api.create_job("visualization", request=request.dict())

            # Create visualization in background
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

    @app.get("/visualizations/{viz_id}")
    async def get_visualization(viz_id: str):
        """Get visualization file."""
        # This would serve the actual visualization file
        viz_path = Path(f"./outputs/visualizations/{viz_id}.html")
        if not viz_path.exists():
            raise HTTPException(status_code=404, detail="Visualization not found")

        return FileResponse(viz_path, media_type="text/html")

    @app.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        """Upload file for analysis."""
        try:
            # Save uploaded file
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
                # Handle WebSocket communication
                data = await websocket.receive_text()
                message = json.loads(data)

                # Echo back for now
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "client_id": client_id,
                    "message": f"Received: {message}"
                }))

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await websocket.close()

    # Add API instance to app state
    app.state.api = api

    return app

# Additional methods for NeuronMapAPI
def _add_api_methods():
    """Add additional methods to NeuronMapAPI class."""

    async def create_visualization_job(self, request: VisualizationRequest, job_id: str):
        """Create visualization in background."""
        try:
            self.update_job(job_id, "running", progress=0.1)

            # Get analysis results
            analysis_data = self.results[request.analysis_id]

            self.update_job(job_id, "running", progress=0.3)

            # Create visualization
            viz_config = request.config or {}

            if request.plot_type == "dashboard":
                viz_path = create_interactive_dashboard(
                    results=analysis_data['results'],
                    save_path=f"./outputs/visualizations/{job_id}.html",
                    **viz_config
                )
            else:
                # Create specific plot type
                viz_path = self.neuron_map.create_visualization(
                    results=analysis_data['results'],
                    plot_type=request.plot_type,
                    save_path=f"./outputs/visualizations/{job_id}.png",
                    **viz_config
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

    # Add method to class
    NeuronMapAPI.create_visualization_job = create_visualization_job

# Apply additional methods
_add_api_methods()

# CLI for running the API server
def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    config_path: Optional[str] = None,
    reload: bool = False
):
    """Run the API server."""
    app = create_app(config_path)

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuronMap API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        config_path=args.config,
        reload=args.reload
    )
