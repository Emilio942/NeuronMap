"""Health Monitoring System for External Services in NeuronMap.

This module provides comprehensive health monitoring for external services with automatic
failover, service discovery, and connection pooling according to roadmap section 2.3.
"""

import asyncio
import aiohttp
import time
import threading
import logging
import json
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from contextlib import asynccontextmanager
import socket
import requests
from urllib.parse import urlparse

from .error_handling import NeuronMapException, ResourceError
from .robust_decorators import robust_execution

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


class ServiceType(Enum):
    """Types of external services."""
    OLLAMA = "ollama"
    HTTP_API = "http_api"
    DATABASE = "database"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    MODEL_SERVER = "model_server"
    CUSTOM = "custom"


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""
    name: str
    service_type: ServiceType
    url: str
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 60.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['service_type'] = self.service_type.value
        return data


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service_name: str
    status: ServiceStatus
    response_time_ms: float
    timestamp: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        return data


@dataclass
class ServiceStatistics:
    """Service statistics over time."""
    service_name: str
    avg_response_time: float
    p95_response_time: float
    uptime_percentage: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class CircuitBreaker:
    """Circuit breaker pattern implementation for service protection."""

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: tuple = (Exception,)):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception types that trigger circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self.lock = threading.Lock()

    def __call__(self, func):
        """Decorator to wrap function with circuit breaker."""
        def wrapper(*args, **kwargs):
            with self.lock:
                if self.state == 'open':
                    if (time.time() - self.last_failure_time) > self.recovery_timeout:
                        self.state = 'half-open'
                        logger.info(f"Circuit breaker entering half-open state")
                    else:
                        raise ResourceError("Circuit breaker is open")

                try:
                    result = func(*args, **kwargs)
                    self.on_success()
                    return result

                except self.expected_exception as e:
                    self.on_failure()
                    raise

        return wrapper

    def on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = 'closed'
        logger.debug("Circuit breaker reset to closed state")

    def on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class ConnectionPool:
    """Connection pool for HTTP services."""

    def __init__(self, max_connections: int = 10, timeout: float = 30.0):
        """Initialize connection pool.

        Args:
            max_connections: Maximum number of connections
            timeout: Connection timeout in seconds
        """
        self.max_connections = max_connections
        self.timeout = timeout
        self.session = None
        self.connector = None
        self._lock = threading.Lock()

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            with self._lock:
                if self.session is None or self.session.closed:
                    self.connector = aiohttp.TCPConnector(
                        limit=self.max_connections,
                        limit_per_host=self.max_connections,
                        ttl_dns_cache=300,
                        use_dns_cache=True
                    )

                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    self.session = aiohttp.ClientSession(
                        connector=self.connector,
                        timeout=timeout
                    )

        return self.session

    async def close(self):
        """Close connection pool."""
        if self.session and not self.session.closed:
            await self.session.close()
        if self.connector:
            await self.connector.close()


class ServiceHealthChecker:
    """Health checker for individual service types."""

    @staticmethod
    async def check_ollama_service(endpoint: ServiceEndpoint,
                                  session: aiohttp.ClientSession) -> HealthCheckResult:
        """Check Ollama service health."""
        start_time = time.time()

        try:
            # Try to list models as health check
            health_url = f"{endpoint.url.rstrip('/')}/api/tags"

            async with session.get(health_url) as response:
                response_time = (time.time() - start_time) * 1000

                if response.status == 200:
                    data = await response.json()
                    model_count = len(data.get('models', []))

                    return HealthCheckResult(
                        service_name=endpoint.name,
                        status=ServiceStatus.HEALTHY,
                        response_time_ms=response_time,
                        timestamp=time.time(),
                        metadata={
                            'model_count': model_count,
                            'status_code': response.status
                        }
                    )
                else:
                    return HealthCheckResult(
                        service_name=endpoint.name,
                        status=ServiceStatus.DEGRADED,
                        response_time_ms=response_time,
                        timestamp=time.time(),
                        error=f"HTTP {response.status}",
                        metadata={'status_code': response.status}
                    )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service_name=endpoint.name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                timestamp=time.time(),
                error=str(e)
            )

    @staticmethod
    async def check_http_api(endpoint: ServiceEndpoint,
                           session: aiohttp.ClientSession) -> HealthCheckResult:
        """Check HTTP API service health."""
        start_time = time.time()

        try:
            # Use custom health endpoint if specified, otherwise root
            health_url = endpoint.metadata.get('health_endpoint', endpoint.url)

            async with session.get(health_url) as response:
                response_time = (time.time() - start_time) * 1000

                if response.status == 200:
                    return HealthCheckResult(
                        service_name=endpoint.name,
                        status=ServiceStatus.HEALTHY,
                        response_time_ms=response_time,
                        timestamp=time.time(),
                        metadata={'status_code': response.status}
                    )
                else:
                    return HealthCheckResult(
                        service_name=endpoint.name,
                        status=ServiceStatus.DEGRADED,
                        response_time_ms=response_time,
                        timestamp=time.time(),
                        error=f"HTTP {response.status}",
                        metadata={'status_code': response.status}
                    )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service_name=endpoint.name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                timestamp=time.time(),
                error=str(e)
            )

    @staticmethod
    def check_tcp_service(endpoint: ServiceEndpoint) -> HealthCheckResult:
        """Check TCP service health."""
        start_time = time.time()

        try:
            parsed_url = urlparse(endpoint.url)
            host = parsed_url.hostname or 'localhost'
            port = parsed_url.port or 80

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(endpoint.timeout)

            result = sock.connect_ex((host, port))
            response_time = (time.time() - start_time) * 1000

            sock.close()

            if result == 0:
                return HealthCheckResult(
                    service_name=endpoint.name,
                    status=ServiceStatus.HEALTHY,
                    response_time_ms=response_time,
                    timestamp=time.time(),
                    metadata={'port_open': True}
                )
            else:
                return HealthCheckResult(
                    service_name=endpoint.name,
                    status=ServiceStatus.UNHEALTHY,
                    response_time_ms=response_time,
                    timestamp=time.time(),
                    error=f"Connection failed: {result}",
                    metadata={'port_open': False}
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service_name=endpoint.name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                timestamp=time.time(),
                error=str(e)
            )


class FailoverManager:
    """Manages automatic failover between service instances."""

    def __init__(self):
        self.primary_services: Dict[str, ServiceEndpoint] = {}
        self.fallback_services: Dict[str, List[ServiceEndpoint]] = defaultdict(list)
        self.current_services: Dict[str, ServiceEndpoint] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    def register_primary_service(self, service: ServiceEndpoint):
        """Register primary service."""
        self.primary_services[service.name] = service
        self.current_services[service.name] = service
        self.circuit_breakers[service.name] = CircuitBreaker()

    def register_fallback_service(self, service_name: str, fallback: ServiceEndpoint):
        """Register fallback service."""
        self.fallback_services[service_name].append(fallback)
        self.circuit_breakers[fallback.name] = CircuitBreaker()

    def get_active_service(self, service_name: str) -> Optional[ServiceEndpoint]:
        """Get currently active service endpoint."""
        return self.current_services.get(service_name)

    def trigger_failover(self, service_name: str) -> Optional[ServiceEndpoint]:
        """Trigger failover to next available service."""
        fallbacks = self.fallback_services.get(service_name, [])

        for fallback in fallbacks:
            if self.circuit_breakers[fallback.name].state != 'open':
                logger.warning(f"Triggering failover from {service_name} to {fallback.name}")
                self.current_services[service_name] = fallback
                return fallback

        logger.error(f"No available fallbacks for service {service_name}")
        return None

    def restore_primary(self, service_name: str):
        """Restore primary service if healthy."""
        primary = self.primary_services.get(service_name)
        if primary and self.circuit_breakers[primary.name].state == 'closed':
            logger.info(f"Restoring primary service {service_name}")
            self.current_services[service_name] = primary


class ServiceRegistry:
    """Registry for managing service endpoints."""

    def __init__(self):
        self.services: Dict[str, ServiceEndpoint] = {}
        self.service_groups: Dict[str, List[str]] = defaultdict(list)

    def register_service(self, endpoint: ServiceEndpoint, group: Optional[str] = None):
        """Register a service endpoint."""
        self.services[endpoint.name] = endpoint

        if group:
            self.service_groups[group].append(endpoint.name)

        logger.info(f"Registered service: {endpoint.name} ({endpoint.service_type.value})")

    def get_service(self, name: str) -> Optional[ServiceEndpoint]:
        """Get service by name."""
        return self.services.get(name)

    def get_services_by_type(self, service_type: ServiceType) -> List[ServiceEndpoint]:
        """Get all services of a specific type."""
        return [svc for svc in self.services.values() if svc.service_type == service_type]

    def get_services_by_group(self, group: str) -> List[ServiceEndpoint]:
        """Get all services in a group."""
        service_names = self.service_groups.get(group, [])
        return [self.services[name] for name in service_names if name in self.services]

    def list_services(self) -> List[ServiceEndpoint]:
        """List all registered services."""
        return list(self.services.values())

    def remove_service(self, name: str):
        """Remove a service."""
        if name in self.services:
            del self.services[name]

            # Remove from groups
            for group, service_names in self.service_groups.items():
                if name in service_names:
                    service_names.remove(name)

            logger.info(f"Removed service: {name}")


class HealthMonitor:
    """Main health monitoring system."""

    def __init__(self,
                 check_interval: float = 60.0,
                 history_size: int = 1000):
        """Initialize health monitor.

        Args:
            check_interval: Interval between health checks in seconds
            history_size: Number of health check results to keep in history
        """
        self.check_interval = check_interval
        self.history_size = history_size

        self.registry = ServiceRegistry()
        self.failover_manager = FailoverManager()
        self.connection_pool = ConnectionPool()
        self.health_checker = ServiceHealthChecker()

        # Health check history
        self.health_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()

        # Callbacks for health events
        self.health_callbacks: List[Callable] = []

        logger.info("Health monitor initialized")

    def register_ollama_service(self, name: str, url: str,
                               fallback_urls: Optional[List[str]] = None):
        """Register Ollama service with optional fallbacks."""
        primary = ServiceEndpoint(
            name=name,
            service_type=ServiceType.OLLAMA,
            url=url,
            health_check_interval=30.0  # Check Ollama more frequently
        )

        self.registry.register_service(primary, group='model_services')
        self.failover_manager.register_primary_service(primary)

        # Register fallbacks if provided
        if fallback_urls:
            for i, fallback_url in enumerate(fallback_urls):
                fallback = ServiceEndpoint(
                    name=f"{name}_fallback_{i}",
                    service_type=ServiceType.OLLAMA,
                    url=fallback_url
                )
                self.registry.register_service(fallback, group='model_services')
                self.failover_manager.register_fallback_service(name, fallback)

        logger.info(f"Registered Ollama service: {name} with {len(fallback_urls or [])} fallbacks")

    def register_custom_service(self, endpoint: ServiceEndpoint,
                               fallbacks: Optional[List[ServiceEndpoint]] = None):
        """Register custom service with optional fallbacks."""
        self.registry.register_service(endpoint)
        self.failover_manager.register_primary_service(endpoint)

        if fallbacks:
            for fallback in fallbacks:
                self.registry.register_service(fallback)
                self.failover_manager.register_fallback_service(endpoint.name, fallback)

    def add_health_callback(self, callback: Callable[[HealthCheckResult], None]):
        """Add callback for health check events."""
        self.health_callbacks.append(callback)

    async def check_service_health(self, service: ServiceEndpoint) -> HealthCheckResult:
        """Check health of a single service."""
        try:
            if service.service_type == ServiceType.OLLAMA:
                session = await self.connection_pool.get_session()
                result = await self.health_checker.check_ollama_service(service, session)
            elif service.service_type == ServiceType.HTTP_API:
                session = await self.connection_pool.get_session()
                result = await self.health_checker.check_http_api(service, session)
            else:
                # Fallback to TCP check
                result = self.health_checker.check_tcp_service(service)

            # Store in history
            self.health_history[service.name].append(result)

            # Trigger callbacks
            for callback in self.health_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in health callback: {e}")

            return result

        except Exception as e:
            logger.error(f"Error checking health of {service.name}: {e}")
            result = HealthCheckResult(
                service_name=service.name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=0,
                timestamp=time.time(),
                error=str(e)
            )
            self.health_history[service.name].append(result)
            return result

    async def check_all_services(self) -> Dict[str, HealthCheckResult]:
        """Check health of all registered services."""
        results = {}

        services = self.registry.list_services()
        if not services:
            return results

        # Check services concurrently
        tasks = []
        for service in services:
            if service.enabled:
                task = asyncio.create_task(self.check_service_health(service))
                tasks.append((service.name, task))

        # Wait for all checks to complete
        for service_name, task in tasks:
            try:
                result = await task
                results[service_name] = result

                # Handle failover if service is unhealthy
                if result.status == ServiceStatus.UNHEALTHY:
                    fallback = self.failover_manager.trigger_failover(service_name)
                    if fallback:
                        # Check fallback service too
                        fallback_result = await self.check_service_health(fallback)
                        results[f"{service_name}_fallback"] = fallback_result

            except Exception as e:
                logger.error(f"Error checking {service_name}: {e}")

        return results

    def get_service_statistics(self, service_name: str,
                             hours: float = 24.0) -> Optional[ServiceStatistics]:
        """Get statistics for a service over time window."""
        if service_name not in self.health_history:
            return None

        history = list(self.health_history[service_name])
        if not history:
            return None

        # Filter to time window
        cutoff_time = time.time() - (hours * 3600)
        recent_checks = [check for check in history if check.timestamp >= cutoff_time]

        if not recent_checks:
            return None

        # Calculate statistics
        response_times = [check.response_time_ms for check in recent_checks]
        successful_checks = [check for check in recent_checks
                           if check.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]]
        failed_checks = [check for check in recent_checks
                        if check.status == ServiceStatus.UNHEALTHY]

        # Calculate uptime
        total_time = recent_checks[-1].timestamp - recent_checks[0].timestamp
        downtime = sum(
            (recent_checks[i].timestamp - recent_checks[i-1].timestamp)
            for i in range(1, len(recent_checks))
            if recent_checks[i-1].status == ServiceStatus.UNHEALTHY
        )
        uptime_percentage = ((total_time - downtime) / total_time * 100) if total_time > 0 else 100

        # Calculate percentiles
        sorted_response_times = sorted(response_times)
        p95_index = int(0.95 * len(sorted_response_times))
        p95_response_time = sorted_response_times[p95_index] if sorted_response_times else 0

        return ServiceStatistics(
            service_name=service_name,
            avg_response_time=sum(response_times) / len(response_times) if response_times else 0,
            p95_response_time=p95_response_time,
            uptime_percentage=uptime_percentage,
            total_requests=len(recent_checks),
            successful_requests=len(successful_checks),
            failed_requests=len(failed_checks),
            last_failure_time=failed_checks[-1].timestamp if failed_checks else None,
            last_success_time=successful_checks[-1].timestamp if successful_checks else None
        )

    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        services = self.registry.list_services()
        if not services:
            return {
                'status': 'unknown',
                'message': 'No services registered',
                'services': {}
            }

        service_statuses = {}
        healthy_count = 0
        total_count = 0

        for service in services:
            history = self.health_history.get(service.name, deque())
            if history:
                latest = history[-1]
                service_statuses[service.name] = {
                    'status': latest.status.value,
                    'response_time': latest.response_time_ms,
                    'last_check': latest.timestamp,
                    'error': latest.error
                }

                if latest.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]:
                    healthy_count += 1
                total_count += 1
            else:
                service_statuses[service.name] = {
                    'status': 'unknown',
                    'message': 'No health checks performed'
                }
                total_count += 1

        # Determine overall status
        if total_count == 0:
            overall_status = 'unknown'
        elif healthy_count == total_count:
            overall_status = 'healthy'
        elif healthy_count > 0:
            overall_status = 'degraded'
        else:
            overall_status = 'unhealthy'

        return {
            'status': overall_status,
            'healthy_services': healthy_count,
            'total_services': total_count,
            'services': service_statuses,
            'timestamp': time.time()
        }

    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Health monitoring already started")
            return

        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop health monitoring."""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while not self.stop_monitoring.is_set():
                try:
                    # Run health checks
                    results = loop.run_until_complete(self.check_all_services())

                    # Log summary
                    healthy = sum(1 for r in results.values()
                                 if r.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED])
                    total = len(results)
                    logger.debug(f"Health check completed: {healthy}/{total} services healthy")

                    # Wait for next check
                    self.stop_monitoring.wait(self.check_interval)

                except Exception as e:
                    logger.error(f"Error in health monitoring loop: {e}")
                    self.stop_monitoring.wait(self.check_interval)

        finally:
            loop.run_until_complete(self.connection_pool.close())
            loop.close()

    def export_health_data(self, filename: str, hours: float = 24.0):
        """Export health check data to file."""
        cutoff_time = time.time() - (hours * 3600)

        export_data = {}
        for service_name, history in self.health_history.items():
            recent_checks = [
                check.to_dict() for check in history
                if check.timestamp >= cutoff_time
            ]
            if recent_checks:
                export_data[service_name] = recent_checks

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported health data to {filename}")


# Global health monitor instance
health_monitor = HealthMonitor()

def register_ollama_service(name: str, url: str, fallback_urls: Optional[List[str]] = None):
    """Register Ollama service for health monitoring."""
    health_monitor.register_ollama_service(name, url, fallback_urls)

def start_health_monitoring():
    """Start health monitoring."""
    health_monitor.start_monitoring()

def stop_health_monitoring():
    """Stop health monitoring."""
    health_monitor.stop_monitoring()

def get_service_health(service_name: Optional[str] = None):
    """Get health status for specific service or overall system."""
    if service_name:
        return health_monitor.get_service_statistics(service_name)
    else:
        return health_monitor.get_overall_health()

def check_ollama_connection(url: str = "http://localhost:11434") -> bool:
    """Quick check if Ollama is accessible."""
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False
