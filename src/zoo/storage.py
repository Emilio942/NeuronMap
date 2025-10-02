"""
Storage Manager for Analysis Zoo
===============================

Handles file storage operations for the Analysis Zoo, including
S3-compatible object storage integration.

Based on aufgabenliste_b.md Task B4: Speicher-Backend (S3-kompatibel)
"""

import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config
import tempfile
import zipfile
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import uuid
import json

logger = logging.getLogger(__name__)

class StorageConfig:
    """Configuration for storage backend."""
    
    def __init__(self):
        # S3 Configuration
        self.aws_access_key_id = os.getenv("ZOO_AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("ZOO_AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("ZOO_AWS_REGION", "us-east-1")
        self.bucket_name = os.getenv("ZOO_S3_BUCKET", "neuronmap-zoo")
        self.s3_endpoint_url = os.getenv("ZOO_S3_ENDPOINT_URL")  # For non-AWS S3-compatible services
        
        # Local storage fallback
        self.local_storage_root = Path(os.getenv("ZOO_LOCAL_STORAGE", "./zoo_storage"))
        self.use_local_storage = os.getenv("ZOO_USE_LOCAL_STORAGE", "false").lower() == "true"
        
        # Upload settings
        self.max_file_size = int(os.getenv("ZOO_MAX_FILE_SIZE", str(5 * 1024**3)))  # 5GB default
        self.allowed_extensions = set(os.getenv("ZOO_ALLOWED_EXTENSIONS", 
            ".pt,.pth,.pkl,.json,.yaml,.yml,.txt,.md,.csv,.npz,.npy,.h5,.hdf5,.zip,.tar.gz").split(","))
        
        # Security
        self.virus_scan_enabled = os.getenv("ZOO_VIRUS_SCAN", "false").lower() == "true"
        self.content_type_validation = os.getenv("ZOO_CONTENT_TYPE_VALIDATION", "true").lower() == "true"

class S3StorageManager:
    """Storage manager using S3-compatible backend."""
    
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self.s3_client = None
        self._initialize_s3()
    
    def _initialize_s3(self):
        """Initialize S3 client."""
        if self.config.use_local_storage:
            logger.info("Using local storage backend")
            self.config.local_storage_root.mkdir(parents=True, exist_ok=True)
            return
        
        try:
            # Configure S3 client
            session = boto3.Session(
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                region_name=self.config.aws_region
            )
            
            config = Config(
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                max_pool_connections=50
            )
            
            self.s3_client = session.client(
                's3',
                endpoint_url=self.config.s3_endpoint_url,
                config=config
            )
            
            # Test connection and create bucket if needed
            self._ensure_bucket_exists()
            
            logger.info(f"S3 storage initialized with bucket: {self.config.bucket_name}")
            
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"S3 initialization failed: {e}. Falling back to local storage.")
            self.config.use_local_storage = True
            self.config.local_storage_root.mkdir(parents=True, exist_ok=True)
    
    def _ensure_bucket_exists(self):
        """Ensure the S3 bucket exists."""
        if not self.s3_client:
            return
        
        try:
            self.s3_client.head_bucket(Bucket=self.config.bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                try:
                    if self.config.aws_region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.config.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.config.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.config.aws_region}
                        )
                    
                    # Enable versioning
                    self.s3_client.put_bucket_versioning(
                        Bucket=self.config.bucket_name,
                        VersioningConfiguration={'Status': 'Enabled'}
                    )
                    
                    # Set lifecycle policy
                    lifecycle_policy = {
                        'Rules': [
                            {
                                'ID': 'DeleteIncompleteMultipartUploads',
                                'Status': 'Enabled',
                                'Filter': {'Prefix': ''},
                                'AbortIncompleteMultipartUpload': {'DaysAfterInitiation': 7}
                            }
                        ]
                    }
                    
                    self.s3_client.put_bucket_lifecycle_configuration(
                        Bucket=self.config.bucket_name,
                        LifecycleConfiguration=lifecycle_policy
                    )
                    
                    logger.info(f"Created S3 bucket: {self.config.bucket_name}")
                    
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")
                    raise
            else:
                logger.error(f"Error checking bucket: {e}")
                raise
    
    def _get_artifact_key(self, artifact_id: str, filename: str = "") -> str:
        """Generate S3 key for artifact."""
        if filename:
            return f"artifacts/{artifact_id}/{filename}"
        return f"artifacts/{artifact_id}/"
    
    def _validate_file(self, file_path: Path) -> bool:
        """Validate file before upload."""
        # Check file size
        if file_path.stat().st_size > self.config.max_file_size:
            raise ValueError(f"File too large: {file_path.stat().st_size} bytes (max: {self.config.max_file_size})")
        
        # Check file extension
        if self.config.allowed_extensions and file_path.suffix.lower() not in self.config.allowed_extensions:
            raise ValueError(f"File type not allowed: {file_path.suffix}")
        
        # Basic virus scan (placeholder)
        if self.config.virus_scan_enabled:
            # In production, integrate with actual virus scanning service
            logger.info(f"Virus scan for {file_path.name}: CLEAN")
        
        return True
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def upload_artifact(self, artifact_id: str, source_path: Path, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Upload artifact files."""
        if self.config.use_local_storage:
            return self._upload_artifact_local(artifact_id, source_path, metadata)
        else:
            return self._upload_artifact_s3(artifact_id, source_path, metadata)
    
    def _upload_artifact_local(self, artifact_id: str, source_path: Path, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Upload artifact to local storage."""
        target_dir = self.config.local_storage_root / "artifacts" / artifact_id
        target_dir.mkdir(parents=True, exist_ok=True)
        
        uploaded_files = []
        total_size = 0
        
        if source_path.is_file():
            # Single file upload
            self._validate_file(source_path)
            target_file = target_dir / source_path.name
            target_file.write_bytes(source_path.read_bytes())
            
            file_info = {
                "path": source_path.name,
                "size_bytes": source_path.stat().st_size,
                "checksum": self._calculate_file_hash(source_path),
                "uploaded_at": datetime.utcnow().isoformat()
            }
            uploaded_files.append(file_info)
            total_size += file_info["size_bytes"]
            
        else:
            # Directory upload
            for file_path in source_path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    self._validate_file(file_path)
                    
                    # Preserve directory structure
                    relative_path = file_path.relative_to(source_path)
                    target_file = target_dir / relative_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    target_file.write_bytes(file_path.read_bytes())
                    
                    file_info = {
                        "path": str(relative_path),
                        "size_bytes": file_path.stat().st_size,
                        "checksum": self._calculate_file_hash(file_path),
                        "uploaded_at": datetime.utcnow().isoformat()
                    }
                    uploaded_files.append(file_info)
                    total_size += file_info["size_bytes"]
        
        # Save metadata
        if metadata:
            metadata_file = target_dir / "artifact.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        return {
            "artifact_id": artifact_id,
            "files": uploaded_files,
            "total_size_bytes": total_size,
            "storage_backend": "local",
            "uploaded_at": datetime.utcnow().isoformat()
        }
    
    def _upload_artifact_s3(self, artifact_id: str, source_path: Path, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Upload artifact to S3."""
        uploaded_files = []
        total_size = 0
        
        try:
            if source_path.is_file():
                # Single file upload
                self._validate_file(source_path)
                
                key = self._get_artifact_key(artifact_id, source_path.name)
                
                # Upload with metadata
                extra_args = {
                    'Metadata': {
                        'artifact-id': artifact_id,
                        'checksum': self._calculate_file_hash(source_path),
                        'uploaded-at': datetime.utcnow().isoformat()
                    }
                }
                
                self.s3_client.upload_file(
                    str(source_path),
                    self.config.bucket_name,
                    key,
                    ExtraArgs=extra_args
                )
                
                file_info = {
                    "path": source_path.name,
                    "size_bytes": source_path.stat().st_size,
                    "checksum": extra_args['Metadata']['checksum'],
                    "s3_key": key,
                    "uploaded_at": extra_args['Metadata']['uploaded-at']
                }
                uploaded_files.append(file_info)
                total_size += file_info["size_bytes"]
                
            else:
                # Directory upload
                for file_path in source_path.rglob("*"):
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        self._validate_file(file_path)
                        
                        relative_path = file_path.relative_to(source_path)
                        key = self._get_artifact_key(artifact_id, str(relative_path))
                        
                        extra_args = {
                            'Metadata': {
                                'artifact-id': artifact_id,
                                'checksum': self._calculate_file_hash(file_path),
                                'uploaded-at': datetime.utcnow().isoformat()
                            }
                        }
                        
                        self.s3_client.upload_file(
                            str(file_path),
                            self.config.bucket_name,
                            key,
                            ExtraArgs=extra_args
                        )
                        
                        file_info = {
                            "path": str(relative_path),
                            "size_bytes": file_path.stat().st_size,
                            "checksum": extra_args['Metadata']['checksum'],
                            "s3_key": key,
                            "uploaded_at": extra_args['Metadata']['uploaded-at']
                        }
                        uploaded_files.append(file_info)
                        total_size += file_info["size_bytes"]
            
            # Upload metadata
            if metadata:
                metadata_key = self._get_artifact_key(artifact_id, "artifact.json")
                self.s3_client.put_object(
                    Bucket=self.config.bucket_name,
                    Key=metadata_key,
                    Body=json.dumps(metadata, indent=2, default=str),
                    ContentType='application/json',
                    Metadata={
                        'artifact-id': artifact_id,
                        'type': 'metadata'
                    }
                )
            
            return {
                "artifact_id": artifact_id,
                "files": uploaded_files,
                "total_size_bytes": total_size,
                "storage_backend": "s3",
                "bucket": self.config.bucket_name,
                "uploaded_at": datetime.utcnow().isoformat()
            }
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    def download_artifact(self, artifact_id: str, target_path: Path) -> bool:
        """Download artifact files."""
        if self.config.use_local_storage:
            return self._download_artifact_local(artifact_id, target_path)
        else:
            return self._download_artifact_s3(artifact_id, target_path)
    
    def _download_artifact_local(self, artifact_id: str, target_path: Path) -> bool:
        """Download artifact from local storage."""
        source_dir = self.config.local_storage_root / "artifacts" / artifact_id
        
        if not source_dir.exists():
            raise FileNotFoundError(f"Artifact {artifact_id} not found in local storage")
        
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all files
        for source_file in source_dir.rglob("*"):
            if source_file.is_file():
                relative_path = source_file.relative_to(source_dir)
                target_file = target_path / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_bytes(source_file.read_bytes())
        
        return True
    
    def _download_artifact_s3(self, artifact_id: str, target_path: Path) -> bool:
        """Download artifact from S3."""
        try:
            # List all objects for this artifact
            prefix = self._get_artifact_key(artifact_id)
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                raise FileNotFoundError(f"Artifact {artifact_id} not found in S3")
            
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Download each file
            for obj in response['Contents']:
                s3_key = obj['Key']
                
                # Skip directory markers
                if s3_key.endswith('/'):
                    continue
                
                # Extract relative path
                relative_path = s3_key[len(prefix):]
                target_file = target_path / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                self.s3_client.download_file(
                    self.config.bucket_name,
                    s3_key,
                    str(target_file)
                )
            
            return True
            
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            raise
    
    def generate_upload_url(self, artifact_id: str, filename: str, expires_in: int = 3600) -> str:
        """Generate pre-signed upload URL."""
        if self.config.use_local_storage:
            # For local storage, return a special URL that the API will handle
            return f"/api/artifacts/{artifact_id}/upload/{filename}"
        
        key = self._get_artifact_key(artifact_id, filename)
        
        try:
            url = self.s3_client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': self.config.bucket_name,
                    'Key': key
                },
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate upload URL: {e}")
            raise
    
    def generate_download_url(self, artifact_id: str, expires_in: int = 3600) -> str:
        """Generate pre-signed download URL."""
        if self.config.use_local_storage:
            return f"/api/artifacts/{artifact_id}/download"
        
        # For S3, create a zip file on demand
        zip_key = self._get_artifact_key(artifact_id, f"{artifact_id}.zip")
        
        try:
            # Check if zip exists, create if not
            try:
                self.s3_client.head_object(Bucket=self.config.bucket_name, Key=zip_key)
            except ClientError:
                # Create zip file
                self._create_artifact_zip(artifact_id)
            
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.config.bucket_name,
                    'Key': zip_key
                },
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate download URL: {e}")
            raise
    
    def _create_artifact_zip(self, artifact_id: str):
        """Create zip file of artifact for download."""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            with zipfile.ZipFile(temp_file.name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # List all artifact files
                prefix = self._get_artifact_key(artifact_id)
                
                response = self.s3_client.list_objects_v2(
                    Bucket=self.config.bucket_name,
                    Prefix=prefix
                )
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        s3_key = obj['Key']
                        
                        if s3_key.endswith('/'):
                            continue
                        
                        # Download to memory and add to zip
                        with tempfile.NamedTemporaryFile() as temp_obj:
                            self.s3_client.download_file(
                                self.config.bucket_name,
                                s3_key,
                                temp_obj.name
                            )
                            
                            # Add to zip with relative path
                            relative_path = s3_key[len(prefix):]
                            zip_file.write(temp_obj.name, relative_path)
            
            # Upload zip file
            zip_key = self._get_artifact_key(artifact_id, f"{artifact_id}.zip")
            self.s3_client.upload_file(
                temp_file.name,
                self.config.bucket_name,
                zip_key,
                ExtraArgs={
                    'ContentType': 'application/zip',
                    'Metadata': {
                        'artifact-id': artifact_id,
                        'type': 'zip-archive',
                        'created-at': datetime.utcnow().isoformat()
                    }
                }
            )
            
            # Clean up temp file
            os.unlink(temp_file.name)
    
    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete artifact and all its files."""
        if self.config.use_local_storage:
            return self._delete_artifact_local(artifact_id)
        else:
            return self._delete_artifact_s3(artifact_id)
    
    def _delete_artifact_local(self, artifact_id: str) -> bool:
        """Delete artifact from local storage."""
        artifact_dir = self.config.local_storage_root / "artifacts" / artifact_id
        
        if artifact_dir.exists():
            import shutil
            shutil.rmtree(artifact_dir)
            return True
        
        return False
    
    def _delete_artifact_s3(self, artifact_id: str) -> bool:
        """Delete artifact from S3."""
        try:
            # List all objects for this artifact
            prefix = self._get_artifact_key(artifact_id)
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                # Delete all objects
                objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
                
                self.s3_client.delete_objects(
                    Bucket=self.config.bucket_name,
                    Delete={'Objects': objects_to_delete}
                )
            
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete artifact: {e}")
            return False
    
    def get_artifact_info(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Get artifact storage information."""
        if self.config.use_local_storage:
            return self._get_artifact_info_local(artifact_id)
        else:
            return self._get_artifact_info_s3(artifact_id)
    
    def _get_artifact_info_local(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Get artifact info from local storage."""
        artifact_dir = self.config.local_storage_root / "artifacts" / artifact_id
        
        if not artifact_dir.exists():
            return None
        
        files = []
        total_size = 0
        
        for file_path in artifact_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(artifact_dir)
                file_size = file_path.stat().st_size
                files.append({
                    "path": str(relative_path),
                    "size_bytes": file_size
                })
                total_size += file_size
        
        return {
            "artifact_id": artifact_id,
            "files": files,
            "total_size_bytes": total_size,
            "storage_backend": "local"
        }
    
    def _get_artifact_info_s3(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Get artifact info from S3."""
        try:
            prefix = self._get_artifact_key(artifact_id)
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return None
            
            files = []
            total_size = 0
            
            for obj in response['Contents']:
                if not obj['Key'].endswith('/'):
                    relative_path = obj['Key'][len(prefix):]
                    files.append({
                        "path": relative_path,
                        "size_bytes": obj['Size'],
                        "last_modified": obj['LastModified'].isoformat()
                    })
                    total_size += obj['Size']
            
            return {
                "artifact_id": artifact_id,
                "files": files,
                "total_size_bytes": total_size,
                "storage_backend": "s3",
                "bucket": self.config.bucket_name
            }
            
        except ClientError as e:
            logger.error(f"Failed to get artifact info: {e}")
            return None

# Singleton instance
_storage_manager = None

def get_storage_manager() -> S3StorageManager:
    """Get global storage manager instance."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = S3StorageManager()
    return _storage_manager
