"""
Analysis Zoo API Server

FastAPI-based REST API server for the NeuronMap Analysis Zoo.
Provides endpoints for artifact management, search, and discovery.

Based on aufgabenliste_b.md Task B2: API-Server für den "Zoo"
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from .artifact_schema import (
    ArtifactSchema, 
    ArtifactType, 
    LicenseType,
    ArtifactSearchFilter,
    ArtifactSearchResult,
    AuthorInfo
)
from .artifact_manager import ArtifactManager, ArtifactValidationError, ArtifactStorageError
from .auth import get_auth_manager, UserInfo


logger = logging.getLogger(__name__)

# Configuration
class APIConfig:
    """API server configuration."""
    STORAGE_ROOT = Path(os.getenv("ZOO_STORAGE_ROOT", "./zoo_storage"))
    CACHE_DIR = Path(os.getenv("ZOO_CACHE_DIR", "./zoo_cache"))
    API_HOST = os.getenv("ZOO_API_HOST", "localhost")
    API_PORT = int(os.getenv("ZOO_API_PORT", "8001"))
    API_TITLE = "NeuronMap Analysis Zoo API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = """
    REST API for the NeuronMap Analysis Zoo - a collaborative platform for sharing
    and discovering ML interpretability artifacts including SAE models, circuits,
    configurations, and analysis results.
    """
    
    # CORS settings
    CORS_ORIGINS = os.getenv("ZOO_CORS_ORIGINS", "*").split(",")
    
    # Rate limiting (requests per minute per IP)
    RATE_LIMIT = int(os.getenv("ZOO_RATE_LIMIT", "100"))


# Global instances
config = APIConfig()
artifact_manager: Optional[ArtifactManager] = None
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    global artifact_manager
    artifact_manager = ArtifactManager(
        storage_root=config.STORAGE_ROOT,
        cache_dir=config.CACHE_DIR
    )
    logger.info(f"Analysis Zoo API Server starting up")
    logger.info(f"Storage root: {config.STORAGE_ROOT}")
    logger.info(f"Cache directory: {config.CACHE_DIR}")
    
    yield
    
    # Shutdown
    logger.info("Analysis Zoo API Server shutting down")


# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ArtifactUploadRequest(BaseModel):
    """Request model for artifact upload."""
    name: str
    version: str
    artifact_type: ArtifactType
    description: str
    authors: List[AuthorInfo]
    license: LicenseType
    tags: Optional[List[str]] = []
    long_description: Optional[str] = None
    model_compatibility: Optional[List[Dict[str, Any]]] = []
    dependencies: Optional[List[str]] = []


class ArtifactUpdateRequest(BaseModel):
    """Request model for artifact updates."""
    description: Optional[str] = None
    long_description: Optional[str] = None
    tags: Optional[List[str]] = None
    license: Optional[LicenseType] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseModel):
    """Standard success response."""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None


class LoginRequest(BaseModel):
    """Login request model."""
    api_key: str


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: Dict[str, Any]


# Dependency functions
async def get_artifact_manager() -> ArtifactManager:
    """Get the artifact manager instance."""
    if artifact_manager is None:
        raise HTTPException(status_code=500, detail="Artifact manager not initialized")
    return artifact_manager


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[UserInfo]:
    """
    Verify authentication token and return user info.
    
    Supports both JWT tokens and API keys.
    """
    if credentials is None:
        return None
    
    try:
        auth_manager = get_auth_manager()
        user = auth_manager.authenticate_bearer_token(credentials.credentials)
        return user
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        return None


# API Routes

@app.get("/", response_model=Dict[str, Any])
async def root():
    """API root endpoint with basic information."""
    return {
        "name": config.API_TITLE,
        "version": config.API_VERSION,
        "description": "Analysis Zoo API for sharing ML interpretability artifacts",
        "endpoints": {
            "artifacts": "/artifacts",
            "search": "/artifacts/search",
            "upload": "/artifacts/upload",
            "auth": "/auth",
            "stats": "/stats",
            "health": "/health"
        }
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": config.API_VERSION
    }


@app.get("/stats", response_model=Dict[str, Any])
async def get_statistics(manager: ArtifactManager = Depends(get_artifact_manager)):
    """Get collection statistics."""
    try:
        stats = manager.get_statistics()
        return {
            "success": True,
            "data": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/artifacts", response_model=ArtifactSearchResult)
async def list_artifacts(
    artifact_type: Optional[ArtifactType] = Query(None, description="Filter by artifact type"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by"),
    model_name: Optional[str] = Query(None, description="Filter by compatible model"),
    license: Optional[LicenseType] = Query(None, description="Filter by license"),
    author: Optional[str] = Query(None, description="Filter by author name"),
    verified_only: bool = Query(False, description="Only verified artifacts"),
    official_only: bool = Query(False, description="Only official artifacts"),
    featured_only: bool = Query(False, description="Only featured artifacts"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    limit: int = Query(50, ge=1, le=100, description="Results limit"),
    offset: int = Query(0, ge=0, description="Results offset"),
    manager: ArtifactManager = Depends(get_artifact_manager)
):
    """List artifacts with optional filtering."""
    try:
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
        
        # Create filter
        filter_params = ArtifactSearchFilter(
            artifact_type=artifact_type,
            tags=tag_list,
            model_name=model_name,
            license=license,
            author=author,
            verified_only=verified_only,
            official_only=official_only,
            featured_only=featured_only,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset
        )
        
        result = manager.list_artifacts(filter_params)
        return result
        
    except Exception as e:
        logger.error(f"Failed to list artifacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/artifacts/search", response_model=ArtifactSearchResult)
async def search_artifacts(
    query: str = Query(..., description="Search query"),
    artifact_type: Optional[ArtifactType] = Query(None, description="Filter by artifact type"),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    model_name: Optional[str] = Query(None, description="Filter by compatible model"),
    limit: int = Query(50, ge=1, le=100, description="Results limit"),
    offset: int = Query(0, ge=0, description="Results offset"),
    manager: ArtifactManager = Depends(get_artifact_manager)
):
    """Search artifacts by text query."""
    try:
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
        
        # Create filter with search query
        filter_params = ArtifactSearchFilter(
            query=query,
            artifact_type=artifact_type,
            tags=tag_list,
            model_name=model_name,
            limit=limit,
            offset=offset
        )
        
        result = manager.list_artifacts(filter_params)
        return result
        
    except Exception as e:
        logger.error(f"Failed to search artifacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/artifacts/{artifact_id}", response_model=ArtifactSchema)
async def get_artifact(
    artifact_id: str,
    manager: ArtifactManager = Depends(get_artifact_manager)
):
    """Get specific artifact by ID."""
    try:
        artifact = manager.get_artifact(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
        
        return artifact
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get artifact {artifact_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/artifacts/{artifact_id}/download")
async def download_artifact(
    artifact_id: str,
    manager: ArtifactManager = Depends(get_artifact_manager)
):
    """Download artifact as ZIP archive."""
    try:
        artifact = manager.get_artifact(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
        
        # Get artifact path
        artifact_path = manager.download_artifact(artifact_id)
        
        # For now, return the artifact.json file
        # TODO: Create ZIP archive and return it
        manifest_path = artifact_path / "artifact.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=500, detail="Artifact manifest not found")
        
        return FileResponse(
            path=manifest_path,
            filename=f"{artifact.name}-{artifact.version}.json",
            media_type="application/json"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download artifact {artifact_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/artifacts/{artifact_id}/star", response_model=SuccessResponse)
async def star_artifact(
    artifact_id: str,
    user: Optional[UserInfo] = Depends(verify_token),
    manager: ArtifactManager = Depends(get_artifact_manager)
):
    """Star an artifact (requires authentication)."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        artifact = manager.get_artifact(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
        
        # TODO: Implement starring logic with user tracking
        # For now, just increment the star count
        artifact.stars += 1
        
        # Save updated metadata
        metadata_file = manager.metadata_dir / f"{artifact_id}.json"
        with open(metadata_file, 'w') as f:
            import json
            json.dump(artifact.dict(exclude_none=True), f, indent=2, default=str)
        
        return SuccessResponse(
            message=f"Starred artifact {artifact.name}",
            data={"stars": artifact.stars}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to star artifact {artifact_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/artifacts/{artifact_id}", response_model=SuccessResponse)
async def update_artifact(
    artifact_id: str,
    update_data: ArtifactUpdateRequest,
    user: Optional[UserInfo] = Depends(verify_token),
    manager: ArtifactManager = Depends(get_artifact_manager)
):
    """Update artifact metadata (requires authentication)."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Check permissions
    auth_manager = get_auth_manager()
    if not auth_manager.check_permission(user, "push"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        artifact = manager.get_artifact(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
        
        # TODO: Check if user owns this artifact
        
        # Update fields
        if update_data.description is not None:
            artifact.description = update_data.description
        if update_data.long_description is not None:
            artifact.long_description = update_data.long_description
        if update_data.tags is not None:
            artifact.tags = update_data.tags
        if update_data.license is not None:
            artifact.license = update_data.license
        
        artifact.updated_at = datetime.utcnow()
        
        # Save updated metadata
        metadata_file = manager.metadata_dir / f"{artifact_id}.json"
        with open(metadata_file, 'w') as f:
            import json
            json.dump(artifact.dict(exclude_none=True), f, indent=2, default=str)
        
        return SuccessResponse(
            message=f"Updated artifact {artifact.name}",
            data={"artifact_id": artifact_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update artifact {artifact_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/artifacts/{artifact_id}", response_model=SuccessResponse)
async def delete_artifact(
    artifact_id: str,
    user: Optional[UserInfo] = Depends(verify_token),
    manager: ArtifactManager = Depends(get_artifact_manager)
):
    """Delete an artifact (requires authentication)."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Check permissions
    auth_manager = get_auth_manager()
    if not auth_manager.check_permission(user, "admin"):
        raise HTTPException(status_code=403, detail="Admin permissions required")
    
    try:
        artifact = manager.get_artifact(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
        
        success = manager.delete_artifact(artifact_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete artifact")
        
        return SuccessResponse(
            message=f"Deleted artifact {artifact.name}",
            data={"artifact_id": artifact_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete artifact {artifact_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Authentication endpoints

@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login with API key and get JWT token."""
    try:
        auth_manager = get_auth_manager()
        
        # Verify API key
        user = auth_manager.verify_api_key(request.api_key)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Generate JWT token
        jwt_token = auth_manager.create_jwt_token(user)
        
        return LoginResponse(
            access_token=jwt_token,
            expires_in=auth_manager.config.JWT_EXPIRATION_HOURS * 3600,  # seconds
            user_info={
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "display_name": user.display_name,
                "roles": user.roles
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@app.post("/auth/logout", response_model=SuccessResponse)
async def logout(user: Optional[UserInfo] = Depends(verify_token)):
    """Logout and revoke current token."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        # TODO: Revoke the current JWT token
        # For now, just return success
        return SuccessResponse(message="Logged out successfully")
        
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")


@app.get("/auth/me", response_model=Dict[str, Any])
async def get_current_user(user: Optional[UserInfo] = Depends(verify_token)):
    """Get current user information."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    return {
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "display_name": user.display_name,
        "roles": user.roles,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None,
        "is_active": user.is_active
    }


@app.post("/auth/api-key", response_model=SuccessResponse)
async def generate_api_key(user: Optional[UserInfo] = Depends(verify_token)):
    """Generate a new API key for the current user."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        auth_manager = get_auth_manager()
        api_key = auth_manager.generate_api_key(user.user_id)
        
        return SuccessResponse(
            message="API key generated successfully",
            data={"api_key": api_key}
        )
        
    except Exception as e:
        logger.error(f"API key generation failed: {e}")
        raise HTTPException(status_code=500, detail="API key generation failed")


# Protected admin endpoints

@app.post("/admin/users", response_model=SuccessResponse)
async def create_user(
    username: str = Form(...),
    email: str = Form(...),
    display_name: str = Form(...),
    roles: str = Form("read"),
    user: Optional[UserInfo] = Depends(verify_token)
):
    """Create a new user (admin only)."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    auth_manager = get_auth_manager()
    if not auth_manager.check_permission(user, "admin"):
        raise HTTPException(status_code=403, detail="Admin permissions required")
    
    try:
        role_list = [role.strip() for role in roles.split(",")]
        new_user = auth_manager.create_user(username, email, display_name, role_list)
        
        # Generate API key for new user
        api_key = auth_manager.generate_api_key(new_user.user_id)
        
        return SuccessResponse(
            message=f"User {username} created successfully",
            data={
                "user_id": new_user.user_id,
                "api_key": api_key
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"User creation failed: {e}")
        raise HTTPException(status_code=500, detail="User creation failed")


@app.get("/admin/users", response_model=List[Dict[str, Any]])
async def list_users(user: Optional[UserInfo] = Depends(verify_token)):
    """List all users (admin only)."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    auth_manager = get_auth_manager()
    if not auth_manager.check_permission(user, "admin"):
        raise HTTPException(status_code=403, detail="Admin permissions required")
    
    users = auth_manager.list_users()
    return [
        {
            "user_id": u.user_id,
            "username": u.username,
            "email": u.email,
            "display_name": u.display_name,
            "roles": u.roles,
            "created_at": u.created_at.isoformat(),
            "last_login": u.last_login.isoformat() if u.last_login else None,
            "is_active": u.is_active
        }
        for u in users
    ]


# Error handlers
@app.exception_handler(ArtifactValidationError)
async def validation_error_handler(request, exc: ArtifactValidationError):
    """Handle artifact validation errors."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="ValidationError",
            message=str(exc)
        ).dict()
    )


@app.exception_handler(ArtifactStorageError)
async def storage_error_handler(request, exc: ArtifactStorageError):
    """Handle artifact storage errors."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="StorageError",
            message=str(exc)
        ).dict()
    )


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


def run_server(
    host: str = None,
    port: int = None,
    reload: bool = False,
    log_level: str = "info"
):
    """Run the API server."""
    host = host or config.API_HOST
    port = port or config.API_PORT
    
    logger.info(f"Starting Analysis Zoo API server on {host}:{port}")
    
    uvicorn.run(
        "src.zoo.api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analysis Zoo API Server")
    parser.add_argument("--host", default=config.API_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.API_PORT, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )
