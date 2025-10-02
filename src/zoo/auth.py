"""
Authentication System for Analysis Zoo API

Provides OAuth2-inspired authentication with JWT tokens for the Analysis Zoo.
Supports API key generation, token validation, and user management.

Based on aufgabenliste_b.md Task B3: Authentifizierungssystem (API-Keys)
"""

import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class UserInfo(BaseModel):
    """User information model."""
    user_id: str
    username: str
    email: str
    display_name: str
    roles: List[str] = ["read"]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    api_key_hash: Optional[str] = None


class TokenData(BaseModel):
    """JWT token data model."""
    user_id: str
    username: str
    roles: List[str]
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for token revocation


class AuthConfig:
    """Authentication configuration."""
    
    def __init__(self):
        self.JWT_SECRET = os.getenv("ZOO_JWT_SECRET", self._generate_secret())
        self.JWT_ALGORITHM = "HS256"
        self.JWT_EXPIRATION_HOURS = int(os.getenv("ZOO_JWT_EXPIRATION_HOURS", "24"))
        self.API_KEY_LENGTH = 32
        
        # User database file (JSON for simplicity)
        self.USER_DB_PATH = Path(os.getenv("ZOO_USER_DB", "./zoo_users.json"))
        
        # Revoked tokens file
        self.REVOKED_TOKENS_PATH = Path(os.getenv("ZOO_REVOKED_TOKENS", "./zoo_revoked_tokens.json"))
    
    def _generate_secret(self) -> str:
        """Generate a random JWT secret if not provided."""
        return secrets.token_urlsafe(32)


class AuthManager:
    """Authentication manager for the Analysis Zoo."""
    
    def __init__(self, config: AuthConfig = None):
        self.config = config or AuthConfig()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Load user database
        self.users: Dict[str, UserInfo] = self._load_users()
        self.revoked_tokens: set = self._load_revoked_tokens()
        
        # Create default admin user if no users exist
        if not self.users:
            self._create_default_admin()
    
    def _load_users(self) -> Dict[str, UserInfo]:
        """Load users from JSON file."""
        if not self.config.USER_DB_PATH.exists():
            return {}
        
        try:
            with open(self.config.USER_DB_PATH, 'r') as f:
                data = json.load(f)
            
            users = {}
            for user_id, user_data in data.items():
                # Convert datetime strings back to objects
                if 'created_at' in user_data:
                    user_data['created_at'] = datetime.fromisoformat(user_data['created_at'])
                if 'last_login' in user_data and user_data['last_login']:
                    user_data['last_login'] = datetime.fromisoformat(user_data['last_login'])
                
                users[user_id] = UserInfo(**user_data)
            
            return users
            
        except Exception as e:
            logger.error(f"Failed to load user database: {e}")
            return {}
    
    def _save_users(self):
        """Save users to JSON file."""
        try:
            data = {}
            for user_id, user in self.users.items():
                user_dict = user.dict()
                # Convert datetime objects to strings
                user_dict['created_at'] = user_dict['created_at'].isoformat()
                if user_dict['last_login']:
                    user_dict['last_login'] = user_dict['last_login'].isoformat()
                data[user_id] = user_dict
            
            with open(self.config.USER_DB_PATH, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save user database: {e}")
    
    def _load_revoked_tokens(self) -> set:
        """Load revoked tokens from JSON file."""
        if not self.config.REVOKED_TOKENS_PATH.exists():
            return set()
        
        try:
            with open(self.config.REVOKED_TOKENS_PATH, 'r') as f:
                return set(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load revoked tokens: {e}")
            return set()
    
    def _save_revoked_tokens(self):
        """Save revoked tokens to JSON file."""
        try:
            with open(self.config.REVOKED_TOKENS_PATH, 'w') as f:
                json.dump(list(self.revoked_tokens), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save revoked tokens: {e}")
    
    def _create_default_admin(self):
        """Create default admin user."""
        admin_user = UserInfo(
            user_id="admin",
            username="admin",
            email="admin@neuronmap.local",
            display_name="Administrator",
            roles=["read", "push", "admin"],
            created_at=datetime.utcnow()
        )
        
        self.users["admin"] = admin_user
        self._save_users()
        
        logger.info("Created default admin user")
    
    def create_user(
        self,
        username: str,
        email: str,
        display_name: str,
        roles: List[str] = None
    ) -> UserInfo:
        """Create a new user."""
        if roles is None:
            roles = ["read"]
        
        user_id = f"user_{secrets.token_urlsafe(8)}"
        
        # Check if username already exists
        for user in self.users.values():
            if user.username == username:
                raise ValueError(f"Username '{username}' already exists")
        
        user = UserInfo(
            user_id=user_id,
            username=username,
            email=email,
            display_name=display_name,
            roles=roles,
            created_at=datetime.utcnow()
        )
        
        self.users[user_id] = user
        self._save_users()
        
        logger.info(f"Created user: {username} ({user_id})")
        return user
    
    def generate_api_key(self, user_id: str) -> str:
        """Generate an API key for a user."""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        api_key = secrets.token_urlsafe(self.config.API_KEY_LENGTH)
        api_key_hash = self.pwd_context.hash(api_key)
        
        # Store hash in user record
        self.users[user_id].api_key_hash = api_key_hash
        self._save_users()
        
        logger.info(f"Generated API key for user {user_id}")
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[UserInfo]:
        """Verify an API key and return user info."""
        for user in self.users.values():
            if user.api_key_hash and self.pwd_context.verify(api_key, user.api_key_hash):
                if user.is_active:
                    return user
                break
        return None
    
    def create_jwt_token(self, user: UserInfo) -> str:
        """Create a JWT token for a user."""
        now = datetime.utcnow()
        exp = now + timedelta(hours=self.config.JWT_EXPIRATION_HOURS)
        jti = secrets.token_urlsafe(16)
        
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": user.roles,
            "exp": exp,
            "iat": now,
            "jti": jti
        }
        
        token = jwt.encode(payload, self.config.JWT_SECRET, algorithm=self.config.JWT_ALGORITHM)
        
        # Update last login
        user.last_login = now
        self._save_users()
        
        logger.info(f"Created JWT token for user {user.username}")
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[TokenData]:
        """Verify a JWT token and return token data."""
        try:
            payload = jwt.decode(
                token,
                self.config.JWT_SECRET,
                algorithms=[self.config.JWT_ALGORITHM]
            )
            
            # Check if token is revoked
            jti = payload.get("jti")
            if jti in self.revoked_tokens:
                logger.warning(f"Attempt to use revoked token: {jti}")
                return None
            
            # Convert datetime
            exp = datetime.fromtimestamp(payload["exp"])
            iat = datetime.fromtimestamp(payload["iat"])
            
            return TokenData(
                user_id=payload["user_id"],
                username=payload["username"],
                roles=payload["roles"],
                exp=exp,
                iat=iat,
                jti=jti
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.config.JWT_SECRET,
                algorithms=[self.config.JWT_ALGORITHM],
                options={"verify_exp": False}  # Don't verify expiration for revocation
            )
            
            jti = payload.get("jti")
            if jti:
                self.revoked_tokens.add(jti)
                self._save_revoked_tokens()
                logger.info(f"Revoked token: {jti}")
                return True
                
        except jwt.InvalidTokenError:
            pass
        
        return False
    
    def authenticate_bearer_token(self, token: str) -> Optional[UserInfo]:
        """Authenticate a bearer token (JWT or API key)."""
        # Try JWT first
        token_data = self.verify_jwt_token(token)
        if token_data:
            user = self.users.get(token_data.user_id)
            if user and user.is_active:
                return user
        
        # Try API key
        user = self.verify_api_key(token)
        if user:
            return user
        
        return None
    
    def check_permission(self, user: UserInfo, required_role: str) -> bool:
        """Check if user has required role/permission."""
        return required_role in user.roles
    
    def get_user_by_username(self, username: str) -> Optional[UserInfo]:
        """Get user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def list_users(self) -> List[UserInfo]:
        """List all users."""
        return list(self.users.values())
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        if user_id in self.users:
            del self.users[user_id]
            self._save_users()
            logger.info(f"Deleted user: {user_id}")
            return True
        return False
    
    def update_user_roles(self, user_id: str, roles: List[str]) -> bool:
        """Update user roles."""
        if user_id in self.users:
            self.users[user_id].roles = roles
            self._save_users()
            logger.info(f"Updated roles for user {user_id}: {roles}")
            return True
        return False


# Singleton instance
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get the global auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager
