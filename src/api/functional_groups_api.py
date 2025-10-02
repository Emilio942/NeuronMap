"""
Web API for Functional Groups Discovery
=======================================

FastAPI endpoints for the Functional Groups Finder functionality.
Provides REST API access to functional neuron group discovery and analysis.

Author: GitHub Copilot
Date: July 29, 2025
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Any, Optional
import json
import numpy as np
import tempfile
import os
from pathlib import Path
import logging
from pydantic import BaseModel, Field

# Import our functional groups components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.functional_groups_finder import (
    FunctionalGroupsFinder,
    AnalysisTaskType,
    ClusteringMethod,
    NeuronGroup
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/groups", tags=["Functional Groups"])

# Global finder instance (in production, use proper state management)
_finders: Dict[str, FunctionalGroupsFinder] = {}


# Pydantic models for API requests and responses
class DiscoveryRequest(BaseModel):
    """Request model for functional groups discovery."""
    model_name: str = Field(..., description="Name of the transformer model")
    layer: int = Field(..., ge=0, description="Layer number to analyze")
    task_type: str = Field(..., description="Type of cognitive task")
    clustering_method: str = Field(default="kmeans", description="Clustering algorithm")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Correlation threshold")
    min_group_size: int = Field(default=3, ge=1, description="Minimum neurons per group")
    max_group_size: int = Field(default=50, ge=1, description="Maximum neurons per group")
    inputs: List[str] = Field(..., description="Input texts for analysis")


class GroupInfo(BaseModel):
    """Model for neuron group information."""
    group_id: str
    neurons: List[int]
    layer: int
    function: str
    activation_trigger: List[str]
    ablation_effect: str
    confidence: float
    task_type: str
    statistical_metrics: Dict[str, float]
    co_activation_strength: float
    cluster_coherence: float


class DiscoveryResponse(BaseModel):
    """Response model for groups discovery."""
    status: str
    message: str
    session_id: str
    num_groups: int
    groups: List[GroupInfo]
    execution_time: float


class AnalysisReport(BaseModel):
    """Model for analysis report."""
    session_id: str
    pattern_id: str
    summary: str
    detailed_report: str
    recommendations: List[str]


class TaskSpecificityRequest(BaseModel):
    """Request for task specificity analysis."""
    session_id: str
    pattern_id: str
    target_inputs: List[str]


class LayerComparisonRequest(BaseModel):
    """Request for layer comparison analysis."""
    model_name: str
    layers: List[int]
    task_type: str
    inputs: List[str]


# Helper functions
def _validate_task_type(task_type: str) -> AnalysisTaskType:
    """Validate and convert task type string."""
    try:
        return AnalysisTaskType(task_type)
    except ValueError:
        valid_types = [t.value for t in AnalysisTaskType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_type. Must be one of: {valid_types}"
        )


def _validate_clustering_method(method: str) -> ClusteringMethod:
    """Validate and convert clustering method string."""
    try:
        return ClusteringMethod(method)
    except ValueError:
        valid_methods = [m.value for m in ClusteringMethod]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid clustering_method. Must be one of: {valid_methods}"
        )


def _convert_group_to_dict(group: NeuronGroup) -> Dict[str, Any]:
    """Convert NeuronGroup to dictionary for JSON serialization."""
    return {
        "group_id": group.group_id,
        "neurons": group.neurons,
        "layer": group.layer,
        "function": group.function,
        "activation_trigger": group.activation_trigger,
        "ablation_effect": group.ablation_effect,
        "confidence": group.confidence,
        "task_type": group.task_type.value,
        "statistical_metrics": group.statistical_metrics,
        "co_activation_strength": group.co_activation_strength,
        "cluster_coherence": group.cluster_coherence
    }


# API Endpoints
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "functional_groups_api", "version": "1.0"}


@router.get("/task-types")
async def get_task_types():
    """Get available task types for analysis."""
    return {
        "task_types": [
            {
                "value": task.value,
                "description": task.value.replace("_", " ").title()
            }
            for task in AnalysisTaskType
        ]
    }


@router.get("/clustering-methods")
async def get_clustering_methods():
    """Get available clustering methods."""
    return {
        "clustering_methods": [
            {
                "value": method.value,
                "description": method.value.replace("_", " ").title()
            }
            for method in ClusteringMethod
        ]
    }


@router.post("/discover", response_model=DiscoveryResponse)
async def discover_functional_groups(request: DiscoveryRequest, background_tasks: BackgroundTasks):
    """
    Discover functional neuron groups in model activations.
    
    This endpoint analyzes activation patterns to identify groups of neurons
    that work together for specific cognitive tasks.
    """
    try:
        import time
        start_time = time.time()
        
        # Validate inputs
        task_type = _validate_task_type(request.task_type)
        clustering_method = _validate_clustering_method(request.clustering_method)
        
        if not request.inputs:
            raise HTTPException(status_code=400, detail="inputs cannot be empty")
        
        if len(request.inputs) < request.min_group_size:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {request.min_group_size} inputs for analysis"
            )
        
        # Generate session ID
        session_id = f"session_{int(time.time())}_{hash(str(request.inputs[:3]))}"
        
        # Create finder instance
        finder = FunctionalGroupsFinder(
            similarity_threshold=request.similarity_threshold,
            min_group_size=request.min_group_size,
            max_group_size=request.max_group_size,
            clustering_method=clustering_method
        )
        
        # Store finder for later use
        _finders[session_id] = finder
        
        # For demo purposes, simulate activation extraction
        # In production, this would interface with actual model
        logger.info(f"Simulating activation extraction for {request.model_name} layer {request.layer}")
        
        np.random.seed(42)  # For reproducible demo results
        n_samples = len(request.inputs)
        n_neurons = 768  # Typical transformer width
        
        # Generate synthetic activations with task-specific patterns
        activations = np.random.randn(n_samples, n_neurons) * 0.1
        
        # Add structured patterns based on task type
        if task_type == AnalysisTaskType.ARITHMETIC_OPERATIONS:
            # Arithmetic neurons: indices 0-19
            arithmetic_indices = [i for i, inp in enumerate(request.inputs) 
                                if any(op in inp for op in ['+', '-', '*', '/', 'calculate', 'add', 'subtract'])]
            if arithmetic_indices:
                activations[arithmetic_indices, 0:20] += np.random.randn(len(arithmetic_indices), 20) * 0.3 + 1.0
        
        elif task_type == AnalysisTaskType.CAUSAL_REASONING:
            # Causal reasoning neurons: indices 30-49
            causal_indices = [i for i, inp in enumerate(request.inputs)
                            if any(word in inp.lower() for word in ['because', 'therefore', 'since', 'thus', 'so'])]
            if causal_indices:
                activations[causal_indices, 30:50] += np.random.randn(len(causal_indices), 20) * 0.4 + 0.8
        
        elif task_type == AnalysisTaskType.TOKEN_CLASSIFICATION:
            # Token classification neurons: indices 60-79
            classification_indices = [i for i, inp in enumerate(request.inputs)
                                    if any(word in inp for word in ['John', 'Apple', 'New York', 'Monday'])]
            if classification_indices:
                activations[classification_indices, 60:80] += np.random.randn(len(classification_indices), 20) * 0.5 + 1.2
        
        # Add random co-activation patterns
        for group_start in range(0, n_neurons, 30):
            group_end = min(group_start + 10, n_neurons)
            random_samples = np.random.choice(n_samples, size=n_samples//3, replace=False)
            activations[random_samples, group_start:group_end] += np.random.randn(len(random_samples), group_end-group_start) * 0.2 + 0.3
        
        # Add activation pattern to finder
        pattern_id = f"{request.model_name}_layer{request.layer}_{task_type.value}"
        finder.add_activation_pattern(
            pattern_id=pattern_id,
            activations=activations,
            inputs=request.inputs,
            layer=request.layer,
            task_type=task_type
        )
        
        # Discover functional groups
        logger.info("Discovering functional groups...")
        groups = finder.discover_functional_groups(
            pattern_id=pattern_id,
            task_type=task_type,
            generate_visualizations=True
        )
        
        execution_time = time.time() - start_time
        
        # Convert groups to API format
        groups_data = [_convert_group_to_dict(group) for group in groups]
        
        logger.info(f"Discovery completed: {len(groups)} groups found in {execution_time:.2f}s")
        
        return DiscoveryResponse(
            status="success",
            message=f"Successfully discovered {len(groups)} functional groups",
            session_id=session_id,
            num_groups=len(groups),
            groups=groups_data,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error in groups discovery: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")


@router.get("/sessions/{session_id}/report", response_model=AnalysisReport)
async def get_analysis_report(session_id: str):
    """
    Generate and retrieve comprehensive analysis report for a session.
    """
    try:
        if session_id not in _finders:
            raise HTTPException(status_code=404, detail="Session not found")
        
        finder = _finders[session_id]
        
        # Find the pattern ID (assumes single pattern per session for simplicity)
        if not finder.discovered_groups:
            raise HTTPException(status_code=400, detail="No analysis results found for session")
        
        pattern_id = list(finder.discovered_groups.keys())[0]
        
        # Generate comprehensive report
        detailed_report = finder.generate_analysis_report(pattern_id)
        
        # Extract summary from report
        lines = detailed_report.split('\n')
        summary_lines = []
        for line in lines[1:10]:  # Take first few lines as summary
            if line.strip():
                summary_lines.append(line.strip())
        summary = ' '.join(summary_lines)
        
        # Generate recommendations
        groups = finder.discovered_groups[pattern_id]
        recommendations = []
        
        high_confidence_groups = [g for g in groups if g.confidence > 0.8]
        if high_confidence_groups:
            recommendations.append(f"Consider ablation studies on {len(high_confidence_groups)} high-confidence groups")
        
        large_groups = [g for g in groups if len(g.neurons) > 20]
        if large_groups:
            recommendations.append(f"Investigate sub-clustering for {len(large_groups)} large groups")
        
        if len(groups) > 5:
            recommendations.append("Consider cross-layer analysis to track group evolution")
        
        return AnalysisReport(
            session_id=session_id,
            pattern_id=pattern_id,
            summary=summary,
            detailed_report=detailed_report,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.post("/sessions/{session_id}/task-specificity")
async def analyze_task_specificity(session_id: str, request: TaskSpecificityRequest):
    """
    Analyze how specific neuron groups are to particular task inputs.
    """
    try:
        if session_id not in _finders:
            raise HTTPException(status_code=404, detail="Session not found")
        
        finder = _finders[session_id]
        
        # Validate pattern exists
        if request.pattern_id not in finder.activation_patterns:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        # Perform specificity analysis
        specificity_scores = finder.analyze_task_specificity(
            pattern_id=request.pattern_id,
            target_inputs=request.target_inputs
        )
        
        # Organize results by groups
        groups = finder.discovered_groups.get(request.pattern_id, [])
        group_specificity = {}
        
        for group in groups:
            group_scores = {str(neuron): specificity_scores.get(neuron, 0.0) 
                          for neuron in group.neurons}
            group_specificity[group.group_id] = {
                "neuron_scores": group_scores,
                "mean_specificity": np.mean(list(group_scores.values())),
                "max_specificity": max(group_scores.values()) if group_scores else 0,
                "highly_specific_count": sum(1 for score in group_scores.values() if score > 1.5)
            }
        
        return {
            "status": "success",
            "session_id": session_id,
            "pattern_id": request.pattern_id,
            "target_inputs": request.target_inputs,
            "group_specificity": group_specificity,
            "summary": {
                "total_groups_analyzed": len(group_specificity),
                "highly_specific_groups": len([g for g in group_specificity.values() 
                                             if g["mean_specificity"] > 1.5])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in specificity analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Specificity analysis failed: {str(e)}")


@router.post("/compare-layers")
async def compare_layers(request: LayerComparisonRequest):
    """
    Compare functional groups across different layers of a model.
    """
    try:
        import time
        start_time = time.time()
        
        # Validate inputs
        task_type = _validate_task_type(request.task_type)
        
        if len(request.layers) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 layers for comparison")
        
        # Initialize results storage
        layer_results = {}
        
        # Analyze each layer
        for layer in request.layers:
            logger.info(f"Analyzing layer {layer}...")
            
            # Create finder for this layer
            finder = FunctionalGroupsFinder(similarity_threshold=0.6)
            
            # Simulate activations (layer-dependent patterns)
            np.random.seed(42 + layer)
            n_samples, n_neurons = len(request.inputs), 768
            activations = np.random.randn(n_samples, n_neurons) * 0.1
            
            # Layer-specific patterns
            if task_type == AnalysisTaskType.ARITHMETIC_OPERATIONS:
                pattern_strength = 0.3 + layer * 0.1
                if layer <= 6:
                    # Early layers: basic pattern detection
                    activations[:, :30] += np.random.randn(n_samples, 30) * 0.2 + pattern_strength
                else:
                    # Later layers: complex processing
                    activations[:, 40:80] += np.random.randn(n_samples, 40) * 0.3 + pattern_strength
            
            # Add pattern and discover groups
            pattern_id = f"{request.model_name}_layer{layer}_{task_type.value}"
            finder.add_activation_pattern(
                pattern_id=pattern_id,
                activations=activations,
                inputs=request.inputs,
                layer=layer,
                task_type=task_type
            )
            
            groups = finder.discover_functional_groups(
                pattern_id=pattern_id,
                task_type=task_type
            )
            
            # Store results
            layer_results[layer] = {
                "num_groups": len(groups),
                "groups": [_convert_group_to_dict(g) for g in groups],
                "avg_group_size": float(np.mean([len(g.neurons) for g in groups]) if groups else 0),
                "avg_confidence": float(np.mean([g.confidence for g in groups]) if groups else 0),
                "total_neurons_in_groups": sum(len(g.neurons) for g in groups)
            }
        
        # Generate comparison insights
        insights = []
        group_counts = [layer_results[l]["num_groups"] for l in request.layers]
        confidences = [layer_results[l]["avg_confidence"] for l in request.layers]
        
        if group_counts[-1] > group_counts[0]:
            insights.append("Number of functional groups increases with depth")
        elif group_counts[-1] < group_counts[0]:
            insights.append("Number of functional groups decreases with depth")
        
        if confidences[-1] > confidences[0]:
            insights.append("Group specialization increases in deeper layers")
        
        execution_time = time.time() - start_time
        
        return {
            "status": "success",
            "model_name": request.model_name,
            "task_type": request.task_type,
            "layers_analyzed": request.layers,
            "layer_results": layer_results,
            "comparison_insights": insights,
            "execution_time": execution_time,
            "summary": {
                "total_layers": len(request.layers),
                "total_groups_found": sum(r["num_groups"] for r in layer_results.values()),
                "layer_with_most_groups": max(layer_results.keys(), 
                                           key=lambda l: layer_results[l]["num_groups"]),
                "highest_avg_confidence": max(confidences) if confidences else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in layer comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Layer comparison failed: {str(e)}")


@router.get("/sessions/{session_id}/export")
async def export_session_results(session_id: str, format: str = "json"):
    """
    Export session results in various formats.
    """
    try:
        if session_id not in _finders:
            raise HTTPException(status_code=404, detail="Session not found")
        
        finder = _finders[session_id]
        
        if not finder.discovered_groups:
            raise HTTPException(status_code=400, detail="No results to export")
        
        # Create temporary file for export
        temp_dir = Path(tempfile.gettempdir()) / "neuronmap_exports"
        temp_dir.mkdir(exist_ok=True)
        
        pattern_id = list(finder.discovered_groups.keys())[0]
        
        if format.lower() == "json":
            export_file = temp_dir / f"{session_id}_results.json"
            finder.export_groups_to_json(pattern_id, export_file)
            
            return FileResponse(
                path=str(export_file),
                filename=f"functional_groups_{session_id}.json",
                media_type="application/json"
            )
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and free up memory.
    """
    try:
        if session_id not in _finders:
            raise HTTPException(status_code=404, detail="Session not found")
        
        del _finders[session_id]
        
        return {
            "status": "success",
            "message": f"Session {session_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Session deletion failed: {str(e)}")


@router.get("/sessions")
async def list_sessions():
    """
    List all active sessions.
    """
    try:
        sessions = []
        for session_id, finder in _finders.items():
            pattern_count = len(finder.activation_patterns)
            group_count = sum(len(groups) for groups in finder.discovered_groups.values())
            
            sessions.append({
                "session_id": session_id,
                "patterns": pattern_count,
                "total_groups": group_count,
                "patterns_list": list(finder.activation_patterns.keys())
            })
        
        return {
            "status": "success",
            "active_sessions": len(sessions),
            "sessions": sessions
        }
        
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


# Demo endpoint for testing
@router.post("/demo")
async def run_demo():
    """
    Run a demonstration of functional groups discovery with sample data.
    """
    try:
        from src.analysis.functional_groups_finder import create_sample_data
        
        # Create sample data
        activations, inputs = create_sample_data()
        
        # Create finder
        finder = FunctionalGroupsFinder(
            similarity_threshold=0.6,
            min_group_size=3,
            max_group_size=20
        )
        
        # Add pattern
        pattern_id = "demo_mixed_tasks"
        finder.add_activation_pattern(
            pattern_id=pattern_id,
            activations=activations,
            inputs=inputs,
            layer=6,
            task_type=AnalysisTaskType.ARITHMETIC_OPERATIONS
        )
        
        # Discover groups
        groups = finder.discover_functional_groups(
            pattern_id=pattern_id,
            task_type=AnalysisTaskType.ARITHMETIC_OPERATIONS
        )
        
        # Generate session ID and store
        import time
        session_id = f"demo_{int(time.time())}"
        _finders[session_id] = finder
        
        # Convert groups to API format
        groups_data = [_convert_group_to_dict(group) for group in groups]
        
        return {
            "status": "success",
            "message": "Demo completed successfully",
            "session_id": session_id,
            "demo_data": {
                "num_samples": len(inputs),
                "num_neurons": activations.shape[1],
                "discovered_groups": len(groups),
                "groups": groups_data
            }
        }
        
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")


# Include this router in your main FastAPI app
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="NeuronMap Functional Groups API")
    app.include_router(router)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
