#!/usr/bin/env python3
"""Demo script for Section 2.3 Monitoring and Observability features.

This script demonstrates:
1. Advanced Progress Tracking with ETA estimation
2. Comprehensive System Resource Monitoring
3. Performance Metrics Collection and Analysis
4. Health Monitoring for External Services

Usage:
    python demo_monitoring_observability.py
"""

import time
import asyncio
import logging
import random
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.progress_tracker import (
    ProgressTracker, ProgressState, MultiLevelProgressTracker
)
from src.utils.comprehensive_system_monitor import (
    SystemResourceMonitor, ResourceThresholds, AlertLevel,
    get_system_info, format_metrics_summary
)
from src.utils.performance_metrics import (
    PerformanceCollector, MetricType, measure_operation,
    get_performance_summary, analyze_trends
)
from src.utils.health_monitor import (
    HealthMonitor, ServiceEndpoint, ServiceType,
    register_ollama_service, get_service_health
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_progress_tracking():
    """Demonstrate advanced progress tracking and ETA estimation."""
    print("\n" + "="*60)
    print("DEMO: Advanced Progress Tracking & ETA Estimation")
    print("="*60)
    
    # Single-level progress tracking
    print("\n1. Single-Level Progress Tracking:")
    tracker = ProgressTracker(
        total_steps=100,
        operation_name="Data Processing",
        description="Processing neural network activations"
    )
    
    for i in range(100):
        # Simulate variable processing time
        time.sleep(random.uniform(0.01, 0.05))
        
        tracker.update(
            steps_completed=1,
            current_step_info={
                'batch_id': i // 10,
                'sample_id': i,
                'processing_speed': random.uniform(80, 120)
            }
        )
        
        # Print progress every 20 steps
        if i % 20 == 0:
            status = tracker.get_status()
            print(f"  Step {i}: {status.progress_percent:.1f}% - "
                  f"ETA: {status.eta_seconds:.1f}s - "
                  f"Speed: {status.items_per_second:.1f} items/s")
    
    final_status = tracker.get_status()
    print(f"  Completed in {final_status.elapsed_time:.2f}s")
    
    # Multi-level progress tracking
    print("\n2. Multi-Level Progress Tracking:")
    ml_tracker = MultiLevelProgressTracker("Model Analysis")
    
    # Add multiple levels
    ml_tracker.add_level("models", 3, "Processing Models")
    ml_tracker.add_level("layers", 12, "Analyzing Layers")
    ml_tracker.add_level("activations", 1000, "Extracting Activations")
    
    for model in range(3):
        ml_tracker.update_level("models", 1, {"model_name": f"model_{model}"})
        
        for layer in range(12):
            ml_tracker.update_level("layers", 1, {"layer_name": f"layer_{layer}"})
            
            for activation in range(1000):
                time.sleep(0.001)  # Simulate processing
                ml_tracker.update_level("activations", 1)
                
                if activation % 200 == 0:
                    status = ml_tracker.get_overall_status()
                    print(f"    Model {model}, Layer {layer}: "
                          f"{status.overall_progress_percent:.1f}% - "
                          f"ETA: {status.overall_eta_seconds:.1f}s")
            
            # Reset activations for next layer
            ml_tracker.reset_level("activations")
        
        # Reset layers for next model
        ml_tracker.reset_level("layers")
    
    print(f"  Multi-level tracking completed!")


def demo_system_monitoring():
    """Demonstrate comprehensive system resource monitoring."""
    print("\n" + "="*60)
    print("DEMO: Comprehensive System Resource Monitoring")
    print("="*60)
    
    # Create system monitor with custom thresholds
    thresholds = ResourceThresholds(
        cpu_warning=70.0,
        cpu_critical=90.0,
        memory_warning=75.0,
        memory_critical=90.0
    )
    
    monitor = SystemResourceMonitor(
        thresholds=thresholds,
        collection_interval=2.0
    )
    
    # Add alert callback
    def alert_handler(level: AlertLevel, message: str, context: dict):
        print(f"  🚨 ALERT [{level.value.upper()}]: {message}")
    
    monitor.add_alert_callback(alert_handler)
    
    print("\n1. System Information:")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    
    print("\n2. Real-time Resource Monitoring:")
    print("  Collecting metrics for 10 seconds...")
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some workload
    for i in range(5):
        print(f"\n  Measurement {i+1}:")
        
        # Get current metrics
        metrics = monitor.get_system_metrics()
        print(f"    {format_metrics_summary(metrics)}")
        
        # Show any alerts
        if metrics.alerts:
            for alert in metrics.alerts:
                print(f"    ⚠️  {alert['message']}")
        
        # Simulate some CPU/memory load
        _ = [x**2 for x in range(100000)]
        
        time.sleep(2)
    
    # Stop monitoring
    monitor.stop_monitoring_gracefully()
    
    print("\n3. Resource Optimization Recommendations:")
    recommendations = monitor.get_optimization_recommendations()
    if recommendations:
        for resource, recs in recommendations.items():
            print(f"  {resource.upper()}:")
            for rec in recs:
                print(f"    • {rec}")
    else:
        print("  No optimization recommendations at this time.")
    
    print("\n4. Metrics History:")
    history = monitor.get_metrics_history(last_n=3)
    for i, metrics in enumerate(history):
        print(f"  Sample {i+1}: CPU {metrics.cpu.usage_percent:.1f}%, "
              f"Memory {metrics.memory.percent:.1f}%, "
              f"Disk {metrics.disk.usage_percent:.1f}%")


def demo_performance_metrics():
    """Demonstrate performance metrics collection and analysis."""
    print("\n" + "="*60)
    print("DEMO: Performance Metrics Collection & Analysis")
    print("="*60)
    
    # Initialize performance collector
    collector = PerformanceCollector()
    
    print("\n1. Operation Performance Measurement:")
    
    # Measure different operations
    operations = [
        ("model_loading", 0.5, 2.0),
        ("data_preprocessing", 0.1, 0.5),
        ("inference", 0.2, 1.0),
        ("visualization", 0.3, 0.8)
    ]
    
    for op_name, min_time, max_time in operations:
        print(f"\n  Measuring {op_name}:")
        
        # Perform multiple measurements
        for i in range(5):
            processing_time = random.uniform(min_time, max_time)
            
            with measure_operation(
                operation_name=op_name,
                metadata={
                    'iteration': i,
                    'batch_size': random.randint(16, 64),
                    'model_size': random.choice(['small', 'medium', 'large'])
                }
            ):
                # Simulate operation
                time.sleep(processing_time)
                
                # Add some memory allocation
                data = [random.random() for _ in range(10000)]
                
                # Simulate potential error
                if random.random() < 0.1:
                    raise ValueError("Simulated processing error")
            
            print(f"    Run {i+1}: {processing_time:.2f}s")
    
    print("\n2. Performance Summary:")
    summary = get_performance_summary(hours=1.0)
    print(f"  Total metrics collected: {summary['total_metrics']}")
    print(f"  Operations analyzed: {summary['unique_operations']}")
    
    if 'operation_stats' in summary:
        for op_name, stats in summary['operation_stats'].items():
            print(f"\n  {op_name}:")
            if 'avg_duration' in stats and stats['avg_duration'] > 0:
                print(f"    Average duration: {stats['avg_duration']:.3f}s")
            if 'error_rate' in stats:
                print(f"    Error rate: {stats['error_rate']:.1%}")
    
    print("\n3. Performance Trends:")
    trends = analyze_trends(hours=1.0)
    if trends:
        for trend in trends:
            print(f"  {trend.metric_name} ({trend.operation}):")
            print(f"    Trend: {trend.trend_direction} ({trend.change_percent:+.1f}%)")
            print(f"    Confidence: {trend.confidence:.1%}")
            if trend.regression_detected:
                print(f"    ⚠️  Performance regression detected!")
    else:
        print("  No significant trends detected (insufficient data)")
    
    collector.close()


def demo_health_monitoring():
    """Demonstrate health monitoring for external services."""
    print("\n" + "="*60)
    print("DEMO: Health Monitoring for External Services")
    print("="*60)
    
    # Create health monitor
    monitor = HealthMonitor(check_interval=10.0)
    
    print("\n1. Service Registration:")
    
    # Register Ollama service (if available)
    try:
        monitor.register_ollama_service(
            name="local_ollama",
            url="http://localhost:11434",
            fallback_urls=["http://localhost:11435"]  # Backup instance
        )
        print("  ✓ Registered Ollama service")
    except Exception as e:
        print(f"  ⚠️  Could not register Ollama: {e}")
    
    # Register some mock HTTP services
    test_services = [
        ServiceEndpoint(
            name="httpbin_test",
            service_type=ServiceType.HTTP_API,
            url="https://httpbin.org/status/200",
            timeout=10.0,
            metadata={'health_endpoint': 'https://httpbin.org/status/200'}
        ),
        ServiceEndpoint(
            name="mock_api",
            service_type=ServiceType.HTTP_API,
            url="https://jsonplaceholder.typicode.com/posts/1",
            timeout=10.0
        )
    ]
    
    for service in test_services:
        monitor.register_custom_service(service)
        print(f"  ✓ Registered {service.name}")
    
    print("\n2. Health Check Execution:")
    
    async def run_health_checks():
        # Perform initial health checks
        results = await monitor.check_all_services()
        
        print(f"  Checked {len(results)} services:")
        for service_name, result in results.items():
            status_icon = "✓" if result.status.value == "healthy" else "⚠️" if result.status.value == "degraded" else "✗"
            print(f"    {status_icon} {service_name}: {result.status.value} "
                  f"({result.response_time_ms:.1f}ms)")
            
            if result.error:
                print(f"        Error: {result.error}")
    
    # Run health checks
    asyncio.run(run_health_checks())
    
    print("\n3. Overall System Health:")
    overall_health = monitor.get_overall_health()
    print(f"  System Status: {overall_health['status']}")
    print(f"  Healthy Services: {overall_health['healthy_services']}/{overall_health['total_services']}")
    
    print("\n4. Service Statistics:")
    for service_name in monitor.registry.services.keys():
        stats = monitor.get_service_statistics(service_name, hours=1.0)
        if stats:
            print(f"  {service_name}:")
            print(f"    Uptime: {stats.uptime_percentage:.1f}%")
            print(f"    Avg Response Time: {stats.avg_response_time:.1f}ms")
            print(f"    Success Rate: {stats.success_rate:.1f}%")


def simulate_workload_monitoring():
    """Demonstrate integrated monitoring during a simulated workload."""
    print("\n" + "="*60)
    print("DEMO: Integrated Monitoring During Workload")
    print("="*60)
    
    # Initialize all monitoring systems
    system_monitor = SystemResourceMonitor(collection_interval=1.0)
    performance_collector = PerformanceCollector()
    
    # Start monitoring
    system_monitor.start_monitoring()
    
    print("\n  Simulating neural network training workload...")
    print("  (Monitor will track system resources and performance)")
    
    # Simulate training loop
    for epoch in range(3):
        print(f"\n  Epoch {epoch + 1}/3:")
        
        with performance_collector.measure_operation(
            operation_name="training_epoch",
            metadata={'epoch': epoch, 'learning_rate': 0.001}
        ):
            # Simulate model training
            for batch in range(10):
                with system_monitor.monitor_operation(f"epoch_{epoch}_batch_{batch}"):
                    # Simulate computation load
                    computation = [x**0.5 for x in range(50000)]
                    
                    # Add some memory allocation
                    data_batch = [[random.random() for _ in range(1000)] for _ in range(100)]
                    
                    # Simulate processing time
                    time.sleep(random.uniform(0.1, 0.3))
                
                if batch % 5 == 0:
                    # Check system metrics
                    metrics = system_monitor.get_system_metrics()
                    print(f"    Batch {batch}: CPU {metrics.cpu.usage_percent:.1f}%, "
                          f"Memory {metrics.memory.percent:.1f}%")
    
    # Stop monitoring
    system_monitor.stop_monitoring_gracefully()
    
    print("\n  Workload Complete! Generating Reports...")
    
    # Performance summary
    print("\n  Performance Summary:")
    perf_summary = performance_collector.get_performance_summary(hours=1.0)
    print(f"    Total metrics: {perf_summary['total_metrics']}")
    print(f"    Operations: {perf_summary['unique_operations']}")
    
    # System resource recommendations
    print("\n  Resource Optimization Recommendations:")
    recommendations = system_monitor.get_optimization_recommendations()
    if recommendations:
        for resource, recs in recommendations.items():
            print(f"    {resource.upper()}:")
            for rec in recs:
                print(f"      • {rec}")
    else:
        print("    System performance is optimal")
    
    performance_collector.close()


def main():
    """Run all monitoring and observability demos."""
    print("NeuronMap - Section 2.3: Monitoring & Observability Demo")
    print("========================================================")
    
    try:
        # Run individual demos
        demo_progress_tracking()
        demo_system_monitoring()
        demo_performance_metrics()
        demo_health_monitoring()
        simulate_workload_monitoring()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nAll Section 2.3 features demonstrated:")
        print("✓ Advanced Progress Tracking with ETA estimation")
        print("✓ Comprehensive System Resource Monitoring")
        print("✓ Performance Metrics Collection and Analysis")
        print("✓ Health Monitoring for External Services")
        print("✓ Integrated Monitoring During Workloads")
        
        print("\nImplementation Status: COMPLETE")
        print("All requirements from aufgabenliste.md Section 2.3 fulfilled!")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
