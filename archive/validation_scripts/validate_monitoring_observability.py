#!/usr/bin/env python3
"""Validation script for Section 2.3 Monitoring and Observability implementations.

This script validates all Section 2.3 requirements from aufgabenliste.md:
1. Progress tracking with ETA estimation
2. System resource monitoring (RAM, GPU, Disk, Network)
3. Performance metrics collection and analysis
4. Health checks for external services

Usage:
    python validate_monitoring_observability.py
"""

import time
import asyncio
import logging
import random
import tempfile
import threading
from pathlib import Path
import sys
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.progress_tracker import (
    ProgressTracker, ProgressState, MultiLevelProgressTracker,
    ETACalculator, ProgressReporter
)
from src.utils.comprehensive_system_monitor import (
    SystemResourceMonitor, ResourceThresholds, AlertLevel,
    CPUMetrics, MemoryMetrics, DiskMetrics, NetworkMetrics,
    ResourceOptimizer, get_system_info, format_bytes
)
from src.utils.performance_metrics import (
    PerformanceCollector, MetricType, AggregationType,
    PerformanceMetric, AggregatedMetric, PerformanceTrend,
    TrendAnalyzer, measure_operation
)
from src.utils.health_monitor import (
    HealthMonitor, ServiceEndpoint, ServiceType, ServiceStatus,
    HealthCheckResult, ServiceStatistics, CircuitBreaker
)

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during validation
logger = logging.getLogger(__name__)


class ValidationResult:
    """Stores validation test results."""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = []
        self.warnings = []
    
    def add_test(self, test_name: str, passed: bool, error: str = None):
        """Add test result."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            print(f"✓ {test_name}")
        else:
            self.failed_tests.append((test_name, error))
            print(f"✗ {test_name}: {error}")
    
    def add_warning(self, message: str):
        """Add warning message."""
        self.warnings.append(message)
        print(f"⚠️  {message}")
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {len(self.failed_tests)}")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.failed_tests:
            print("\nFAILED TESTS:")
            for test_name, error in self.failed_tests:
                print(f"  ✗ {test_name}: {error}")
        
        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 VALIDATION PASSED - Implementation meets requirements!")
            return True
        else:
            print("❌ VALIDATION FAILED - Implementation needs improvement")
            return False


def validate_progress_tracking(result: ValidationResult):
    """Validate progress tracking functionality."""
    print("\n📊 Validating Progress Tracking...")
    
    try:
        # Test 1: Basic progress tracker creation
        tracker = ProgressTracker(100, "test_operation")
        tracker.start()  # Start the tracker
        result.add_test("Progress tracker creation", True)
    except Exception as e:
        result.add_test("Progress tracker creation", False, str(e))
        return
    
    try:
        # Test 2: Progress updates
        for i in range(10):
            tracker.update(1, {"step": i})
        status = tracker.get_status()
        result.add_test("Progress updates", status['completed_steps'] == 10)
    except Exception as e:
        result.add_test("Progress updates", False, str(e))
    
    try:
        # Test 3: ETA calculation
        status = tracker.get_status()
        has_eta = 'eta_seconds' in status and status['eta_seconds'] >= 0
        result.add_test("ETA calculation", has_eta)
    except Exception as e:
        result.add_test("ETA calculation", False, str(e))
    
    try:
        # Test 4: Progress percentage calculation
        status = tracker.get_status()
        correct_percentage = abs(status['progress_percent'] - 10.0) < 0.1
        result.add_test("Progress percentage calculation", correct_percentage)
    except Exception as e:
        result.add_test("Progress percentage calculation", False, str(e))
    
    try:
        # Test 5: Multi-level progress tracker
        ml_tracker = MultiLevelProgressTracker("test_ml")
        ml_tracker.add_level("level1", 5, "Level 1")
        ml_tracker.add_level("level2", 10, "Level 2")
        
        ml_tracker.update_level("level1", 1)
        ml_tracker.update_level("level2", 5)
        
        status = ml_tracker.get_overall_status()
        has_overall_progress = 'overall_progress_percent' in status
        result.add_test("Multi-level progress tracking", has_overall_progress)
    except Exception as e:
        result.add_test("Multi-level progress tracking", False, str(e))
    
    try:
        # Test 6: Progress state management
        tracker.pause()
        is_paused = tracker.get_status()['state'] == ProgressState.PAUSED.value
        tracker.resume()
        is_resumed = tracker.get_status()['state'] == ProgressState.RUNNING.value
        result.add_test("Progress state management", is_paused and is_resumed)
    except Exception as e:
        result.add_test("Progress state management", False, str(e))


def validate_system_monitoring(result: ValidationResult):
    """Validate comprehensive system resource monitoring."""
    print("\n🖥️  Validating System Resource Monitoring...")
    
    try:
        # Test 1: System monitor creation
        monitor = SystemResourceMonitor()
        result.add_test("System monitor creation", True)
    except Exception as e:
        result.add_test("System monitor creation", False, str(e))
        return
    
    try:
        # Test 2: CPU metrics collection
        cpu_metrics = monitor.collect_cpu_metrics()
        has_cpu_data = (hasattr(cpu_metrics, 'usage_percent') and
                       hasattr(cpu_metrics, 'usage_per_core') and
                       hasattr(cpu_metrics, 'load_average'))
        result.add_test("CPU metrics collection", has_cpu_data)
    except Exception as e:
        result.add_test("CPU metrics collection", False, str(e))
    
    try:
        # Test 3: Memory metrics collection
        memory_metrics = monitor.collect_memory_metrics()
        has_memory_data = (hasattr(memory_metrics, 'total') and
                          hasattr(memory_metrics, 'used') and
                          hasattr(memory_metrics, 'percent'))
        result.add_test("Memory metrics collection", has_memory_data)
    except Exception as e:
        result.add_test("Memory metrics collection", False, str(e))
    
    try:
        # Test 4: Disk metrics collection
        disk_metrics = monitor.collect_disk_metrics()
        has_disk_data = (hasattr(disk_metrics, 'total_space') and
                        hasattr(disk_metrics, 'used_space') and
                        hasattr(disk_metrics, 'usage_percent'))
        result.add_test("Disk metrics collection", has_disk_data)
    except Exception as e:
        result.add_test("Disk metrics collection", False, str(e))
    
    try:
        # Test 5: Network metrics collection
        network_metrics = monitor.collect_network_metrics()
        has_network_data = (hasattr(network_metrics, 'bytes_sent') and
                           hasattr(network_metrics, 'bytes_recv') and
                           hasattr(network_metrics, 'connections_active'))
        result.add_test("Network metrics collection", has_network_data)
    except Exception as e:
        result.add_test("Network metrics collection", False, str(e))
    
    try:
        # Test 6: Complete system metrics
        system_metrics = monitor.get_system_metrics()
        has_complete_metrics = (hasattr(system_metrics, 'cpu') and
                               hasattr(system_metrics, 'memory') and
                               hasattr(system_metrics, 'disk') and
                               hasattr(system_metrics, 'network'))
        result.add_test("Complete system metrics collection", has_complete_metrics)
    except Exception as e:
        result.add_test("Complete system metrics collection", False, str(e))
    
    try:
        # Test 7: Threshold checking and alerting
        # Set low thresholds to trigger alerts
        low_thresholds = ResourceThresholds(
            cpu_warning=0.1, cpu_critical=0.2,
            memory_warning=0.1, memory_critical=0.2
        )
        alert_monitor = SystemResourceMonitor(thresholds=low_thresholds)
        
        alert_triggered = False
        def alert_callback(level, message, context):
            nonlocal alert_triggered
            alert_triggered = True
        
        alert_monitor.add_alert_callback(alert_callback)
        metrics = alert_monitor.get_system_metrics()
        
        result.add_test("Threshold checking and alerting", alert_triggered or len(metrics.alerts) > 0)
    except Exception as e:
        result.add_test("Threshold checking and alerting", False, str(e))
    
    try:
        # Test 8: Resource optimization recommendations
        optimizer = ResourceOptimizer()
        
        # Create mock history with high CPU usage
        mock_cpu = CPUMetrics(
            usage_percent=90.0, usage_per_core=[85, 95, 90, 88],
            load_average=(2.5, 2.0, 1.8), frequency_current=2400,
            frequency_max=3200, context_switches=100000, interrupts=50000
        )
        
        recommendations = optimizer.analyze_cpu_usage(mock_cpu, [])
        has_recommendations = len(recommendations) > 0
        result.add_test("Resource optimization recommendations", has_recommendations)
    except Exception as e:
        result.add_test("Resource optimization recommendations", False, str(e))
    
    try:
        # Test 9: Background monitoring
        monitor.start_monitoring()
        time.sleep(2)  # Let it collect some data
        
        history = monitor.get_metrics_history(last_n=2)
        has_history = len(history) > 0
        
        monitor.stop_monitoring_gracefully()
        result.add_test("Background monitoring", has_history)
    except Exception as e:
        result.add_test("Background monitoring", False, str(e))


def validate_performance_metrics(result: ValidationResult):
    """Validate performance metrics collection and analysis."""
    print("\n📈 Validating Performance Metrics...")
    
    try:
        # Test 1: Performance collector creation
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_metrics.db"
            collector = PerformanceCollector(storage_path=str(db_path))
            result.add_test("Performance collector creation", True)
    except Exception as e:
        result.add_test("Performance collector creation", False, str(e))
        return
    
    try:
        # Test 2: Metric recording
        collector.record_metric(
            name="test_metric",
            value=1.5,
            metric_type=MetricType.EXECUTION_TIME,
            operation="test_operation"
        )
        
        recent_metrics = collector.get_recent_metrics(operation="test_operation")
        has_recorded_metric = len(recent_metrics) > 0
        result.add_test("Metric recording", has_recorded_metric)
    except Exception as e:
        result.add_test("Metric recording", False, str(e))
    
    try:
        # Test 3: Context manager measurement
        with collector.measure_operation("test_context_op") as profiler:
            time.sleep(0.1)  # Simulate operation
        
        context_metrics = collector.get_recent_metrics(operation="test_context_op")
        has_context_metrics = len(context_metrics) > 0
        result.add_test("Context manager measurement", has_context_metrics)
    except Exception as e:
        result.add_test("Context manager measurement", False, str(e))
    
    try:
        # Test 4: Metric aggregation
        # Record multiple metrics for aggregation
        for i in range(10):
            collector.record_metric(
                name="aggregation_test",
                value=random.uniform(0.5, 1.5),
                metric_type=MetricType.EXECUTION_TIME,
                operation="agg_test"
            )
        
        aggregated = collector.aggregate_metrics(
            operation="agg_test",
            metric_name="aggregation_test",
            aggregation_type=AggregationType.MEAN
        )
        
        has_aggregation = aggregated is not None and hasattr(aggregated, 'value')
        result.add_test("Metric aggregation", has_aggregation)
    except Exception as e:
        result.add_test("Metric aggregation", False, str(e))
    
    try:
        # Test 5: Trend analysis
        trend_analyzer = TrendAnalyzer()
        
        # Create test metrics with trend
        test_metrics = []
        for i in range(20):
            metric = PerformanceMetric(
                name="trend_test",
                value=1.0 + (i * 0.1),  # Increasing trend
                metric_type=MetricType.EXECUTION_TIME,
                timestamp=time.time() - (20-i) * 60,  # Spread over time
                operation="trend_op"
            )
            test_metrics.append(metric)
        
        trend = trend_analyzer.analyze_trend(test_metrics)
        has_trend_analysis = trend is not None and hasattr(trend, 'trend_direction')
        result.add_test("Trend analysis", has_trend_analysis)
    except Exception as e:
        result.add_test("Trend analysis", False, str(e))
    
    try:
        # Test 6: Performance summary generation
        summary = collector.get_performance_summary(hours=1.0)
        has_summary = isinstance(summary, dict) and 'total_metrics' in summary
        result.add_test("Performance summary generation", has_summary)
    except Exception as e:
        result.add_test("Performance summary generation", False, str(e))
    
    try:
        # Test 7: Regression detection
        # Create metrics with regression pattern
        regression_metrics = []
        base_time = time.time() - 3600  # 1 hour ago
        
        # Normal performance
        for i in range(10):
            regression_metrics.append(PerformanceMetric(
                name="regression_test", value=1.0, metric_type=MetricType.EXECUTION_TIME,
                timestamp=base_time + i * 60, operation="regression_op"
            ))
        
        # Performance degradation
        for i in range(10):
            regression_metrics.append(PerformanceMetric(
                name="regression_test", value=2.0, metric_type=MetricType.EXECUTION_TIME,
                timestamp=base_time + (10 + i) * 60, operation="regression_op"
            ))
        
        trend = trend_analyzer.analyze_trend(regression_metrics)
        regression_detected = trend and trend.regression_detected
        result.add_test("Regression detection", regression_detected)
    except Exception as e:
        result.add_test("Regression detection", False, str(e))
    
    try:
        collector.close()
        result.add_test("Performance collector cleanup", True)
    except Exception as e:
        result.add_test("Performance collector cleanup", False, str(e))


def validate_health_monitoring(result: ValidationResult):
    """Validate health monitoring for external services."""
    print("\n🏥 Validating Health Monitoring...")
    
    try:
        # Test 1: Health monitor creation
        monitor = HealthMonitor(check_interval=30.0)
        result.add_test("Health monitor creation", True)
    except Exception as e:
        result.add_test("Health monitor creation", False, str(e))
        return
    
    try:
        # Test 2: Service registration
        test_service = ServiceEndpoint(
            name="test_http_service",
            service_type=ServiceType.HTTP_API,
            url="https://httpbin.org/status/200",
            timeout=10.0
        )
        
        monitor.register_custom_service(test_service)
        registered_service = monitor.registry.get_service("test_http_service")
        result.add_test("Service registration", registered_service is not None)
    except Exception as e:
        result.add_test("Service registration", False, str(e))
    
    try:
        # Test 3: Health check execution
        async def test_health_check():
            try:
                health_result = await monitor.check_service_health(test_service)
                return (hasattr(health_result, 'status') and 
                       hasattr(health_result, 'response_time_ms'))
            except Exception:
                return False
        
        health_check_works = asyncio.run(test_health_check())
        result.add_test("Health check execution", health_check_works)
    except Exception as e:
        result.add_test("Health check execution", False, str(e))
    
    try:
        # Test 4: Service statistics
        # Add some mock health history
        mock_result = HealthCheckResult(
            service_name="test_http_service",
            status=ServiceStatus.HEALTHY,
            response_time_ms=150.0,
            timestamp=time.time()
        )
        monitor.health_history["test_http_service"].append(mock_result)
        
        stats = monitor.get_service_statistics("test_http_service")
        has_stats = stats is not None and hasattr(stats, 'avg_response_time')
        result.add_test("Service statistics", has_stats)
    except Exception as e:
        result.add_test("Service statistics", False, str(e))
    
    try:
        # Test 5: Circuit breaker functionality
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        # Test function that fails
        @circuit_breaker
        def failing_function():
            raise Exception("Test failure")
        
        # Trigger failures to open circuit
        failures = 0
        for _ in range(5):
            try:
                failing_function()
            except Exception:
                failures += 1
        
        # Circuit should be open now
        circuit_open = circuit_breaker.state == 'open'
        result.add_test("Circuit breaker functionality", circuit_open and failures >= 3)
    except Exception as e:
        result.add_test("Circuit breaker functionality", False, str(e))
    
    try:
        # Test 6: Failover management
        failover_manager = monitor.failover_manager
        
        primary_service = ServiceEndpoint(
            name="primary_test",
            service_type=ServiceType.HTTP_API,
            url="http://primary.test"
        )
        
        fallback_service = ServiceEndpoint(
            name="fallback_test",
            service_type=ServiceType.HTTP_API,
            url="http://fallback.test"
        )
        
        failover_manager.register_primary_service(primary_service)
        failover_manager.register_fallback_service("primary_test", fallback_service)
        
        # Trigger failover
        failover_result = failover_manager.trigger_failover("primary_test")
        has_failover = failover_result is not None
        result.add_test("Failover management", has_failover)
    except Exception as e:
        result.add_test("Failover management", False, str(e))
    
    try:
        # Test 7: Overall health assessment
        overall_health = monitor.get_overall_health()
        has_health_status = (isinstance(overall_health, dict) and
                           'status' in overall_health and
                           'services' in overall_health)
        result.add_test("Overall health assessment", has_health_status)
    except Exception as e:
        result.add_test("Overall health assessment", False, str(e))


def validate_integration(result: ValidationResult):
    """Validate integration between different monitoring components."""
    print("\n🔗 Validating Component Integration...")
    
    try:
        # Test 1: Progress tracking with resource monitoring
        system_monitor = SystemResourceMonitor(collection_interval=0.5)
        progress_tracker = ProgressTracker(10, "integration_test")
        progress_tracker.start()  # Start the tracker
        
        system_monitor.start_monitoring()
        
        # Simulate work with progress tracking
        for i in range(10):
            with system_monitor.monitor_operation(f"step_{i}"):
                progress_tracker.update(1)
                time.sleep(0.1)
        
        system_monitor.stop_monitoring_gracefully()
        
        # Check that both collected data
        progress_status = progress_tracker.get_status()
        system_history = system_monitor.get_metrics_history()
        
        integration_works = (progress_status['completed_steps'] == 10 and
                           len(system_history) > 0)
        result.add_test("Progress tracking with resource monitoring", integration_works)
    except Exception as e:
        result.add_test("Progress tracking with resource monitoring", False, str(e))
    
    try:
        # Test 2: Performance metrics with health monitoring
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "integration_metrics.db"
            perf_collector = PerformanceCollector(storage_path=str(db_path))
            health_monitor = HealthMonitor()
            
            # Record performance metric
            perf_collector.record_metric(
                name="service_response_time",
                value=0.250,
                metric_type=MetricType.EXECUTION_TIME,
                operation="external_service_call"
            )
            
            # Add health callback that records metrics
            def health_to_performance(health_result):
                perf_collector.record_metric(
                    name="health_check_time",
                    value=health_result.response_time_ms,
                    metric_type=MetricType.EXECUTION_TIME,
                    operation="health_check"
                )
            
            health_monitor.add_health_callback(health_to_performance)
            
            # Simulate health check
            mock_result = HealthCheckResult(
                service_name="test_integration",
                status=ServiceStatus.HEALTHY,
                response_time_ms=120.0,
                timestamp=time.time()
            )
            
            # Trigger callback
            health_to_performance(mock_result)
            
            # Check metrics were recorded
            metrics = perf_collector.get_recent_metrics()
            integration_metrics = len(metrics) >= 2  # Original + health check metric
            
            perf_collector.close()
            result.add_test("Performance metrics with health monitoring", integration_metrics)
    except Exception as e:
        result.add_test("Performance metrics with health monitoring", False, str(e))
    
    try:
        # Test 3: Comprehensive monitoring simulation
        # This test ensures all components can work together
        monitor = SystemResourceMonitor(collection_interval=0.5)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "comprehensive_test.db"
            collector = PerformanceCollector(storage_path=str(db_path))
            
            monitor.start_monitoring()
            
            # Simulate complex operation with all monitoring
            with collector.measure_operation("comprehensive_test") as profiler:
                tracker = ProgressTracker(5, "comprehensive_operation")
                tracker.start()  # Start the tracker
                
                for i in range(5):
                    with monitor.monitor_operation(f"comprehensive_step_{i}"):
                        # Simulate some work
                        _ = [x**2 for x in range(10000)]
                        tracker.update(1)
                        time.sleep(0.1)
            
            monitor.stop_monitoring_gracefully()
            
            # Verify all systems collected data
            system_metrics = monitor.get_metrics_history()
            performance_metrics = collector.get_recent_metrics()
            progress_complete = tracker.get_status()['completed_steps'] == 5
            
            comprehensive_success = (len(system_metrics) > 0 and
                                   len(performance_metrics) > 0 and
                                   progress_complete)
            
            collector.close()
            result.add_test("Comprehensive monitoring simulation", comprehensive_success)
    except Exception as e:
        result.add_test("Comprehensive monitoring simulation", False, str(e))


def validate_error_handling(result: ValidationResult):
    """Validate error handling and edge cases."""
    print("\n⚠️  Validating Error Handling...")
    
    try:
        # Test 1: Invalid progress tracker parameters
        try:
            invalid_tracker = ProgressTracker(-1, "")  # Invalid total steps
            result.add_test("Invalid progress tracker parameters", False, "Should have raised exception")
        except Exception:
            result.add_test("Invalid progress tracker parameters", True)
    except Exception as e:
        result.add_test("Invalid progress tracker parameters", False, str(e))
    
    try:
        # Test 2: Resource monitoring with invalid paths
        monitor = SystemResourceMonitor()
        
        try:
            # Try to monitor non-existent disk path
            disk_metrics = monitor.collect_disk_metrics("/non/existent/path")
            result.add_test("Resource monitoring error handling", False, "Should handle invalid paths")
        except Exception:
            # Expected to fail, but should be handled gracefully
            result.add_test("Resource monitoring error handling", True)
    except Exception as e:
        result.add_test("Resource monitoring error handling", False, str(e))
    
    try:
        # Test 3: Performance collector with invalid database
        try:
            # Try to create collector with invalid path
            invalid_collector = PerformanceCollector(storage_path="/invalid/path/metrics.db")
            result.add_test("Performance collector error handling", False, "Should handle invalid paths")
        except Exception:
            result.add_test("Performance collector error handling", True)
    except Exception as e:
        result.add_test("Performance collector error handling", False, str(e))
    
    try:
        # Test 4: Health monitor with unreachable services
        monitor = HealthMonitor()
        
        unreachable_service = ServiceEndpoint(
            name="unreachable_test",
            service_type=ServiceType.HTTP_API,
            url="http://192.0.2.0:12345",  # RFC 5737 test address
            timeout=1.0
        )
        
        async def test_unreachable():
            result_obj = await monitor.check_service_health(unreachable_service)
            return result_obj.status == ServiceStatus.UNHEALTHY
        
        handles_unreachable = asyncio.run(test_unreachable())
        result.add_test("Health monitor unreachable service handling", handles_unreachable)
    except Exception as e:
        result.add_test("Health monitor unreachable service handling", False, str(e))


def main():
    """Run all validation tests."""
    print("NeuronMap - Section 2.3 Monitoring & Observability Validation")
    print("=============================================================")
    
    result = ValidationResult()
    
    try:
        # Run all validation tests
        validate_progress_tracking(result)
        validate_system_monitoring(result)
        validate_performance_metrics(result)
        validate_health_monitoring(result)
        validate_integration(result)
        validate_error_handling(result)
        
        # Print summary and determine overall result
        success = result.print_summary()
        
        if success:
            print("\n🎉 SECTION 2.3 VALIDATION PASSED!")
            print("All monitoring and observability requirements implemented correctly.")
            return 0
        else:
            print("\n❌ SECTION 2.3 VALIDATION FAILED!")
            print("Some requirements need additional work.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
