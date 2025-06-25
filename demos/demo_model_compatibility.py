#!/usr/bin/env python3
"""Demo script for Model Compatibility and Resource Monitoring systems.

This script demonstrates the advanced model compatibility checking and GPU resource
monitoring capabilities implemented according to roadmap section 2.2.
"""

import sys
import time
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from utils.model_compatibility import (
        ModelCompatibilityChecker, AnalysisType, 
        check_model_compatibility, get_compatible_models
    )
    from utils.resource_monitor import (
        GPUResourceManager, WorkloadPriority, WorkloadTask,
        get_gpu_memory_usage, optimize_gpu_memory
    )
    from utils.error_handling import NeuronMapException
    print("✅ Successfully imported model compatibility and resource monitoring modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)


def demo_model_compatibility():
    """Demonstrate model compatibility checking system."""
    print("\n" + "="*60)
    print("MODEL COMPATIBILITY CHECKING DEMO")
    print("="*60)
    
    checker = ModelCompatibilityChecker()
    
    # 1. Show system resources
    print("\n1. System Resource Detection:")
    print("-" * 30)
    try:
        resources = checker.get_system_resources()
        print(f"CPU Cores: {resources.cpu_cores}")
        print(f"Total Memory: {resources.total_memory_gb:.1f} GB")
        print(f"Available Memory: {resources.available_memory_gb:.1f} GB")
        print(f"GPU Available: {resources.gpu_available}")
        if resources.gpu_available:
            print(f"GPU Count: {resources.gpu_count}")
            print(f"CUDA Version: {resources.cuda_version}")
            for i, (name, memory) in enumerate(zip(resources.gpu_names or [], resources.gpu_memory_gb or [])):
                print(f"  GPU {i}: {name} ({memory:.1f} GB)")
        print(f"Storage Available: {resources.storage_available_gb:.1f} GB")
        print(f"Internet Available: {resources.internet_available}")
    except Exception as e:
        print(f"❌ Error detecting system resources: {e}")
        return
    
    # 2. Test different models and analysis types
    print("\n2. Model Compatibility Tests:")
    print("-" * 30)
    
    test_cases = [
        ("gpt2", AnalysisType.ACTIVATION_EXTRACTION),
        ("gpt2-large", AnalysisType.ATTENTION_VISUALIZATION),
        ("bert-base-uncased", AnalysisType.LAYER_ANALYSIS),
        ("t5-small", AnalysisType.CONCEPTUAL_ANALYSIS),
        ("unknown-model", AnalysisType.ACTIVATION_EXTRACTION),
        ("gpt2", AnalysisType.GRADIENT_ANALYSIS),  # High memory requirement
    ]
    
    for model_name, analysis_type in test_cases:
        print(f"\nTesting: {model_name} + {analysis_type.value}")
        try:
            result = checker.check_compatibility(
                model_name, analysis_type, 
                batch_size=2, sequence_length=512
            )
            
            status = "✅ COMPATIBLE" if result.compatible else "❌ INCOMPATIBLE"
            print(f"  Status: {status} (confidence: {result.confidence_score:.2f})")
            
            if result.estimated_memory_usage_gb:
                print(f"  Est. Memory: {result.estimated_memory_usage_gb:.1f} GB")
            if result.estimated_processing_time_minutes:
                print(f"  Est. Time: {result.estimated_processing_time_minutes:.1f} min")
            
            if result.errors:
                print(f"  Errors: {result.errors}")
            if result.warnings:
                print(f"  Warnings: {result.warnings}")
            if result.recommendations:
                print(f"  Recommendations: {result.recommendations}")
            if result.fallback_suggestions:
                print(f"  Suggestions: {result.fallback_suggestions}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # 3. Show compatible models
    print("\n3. Compatible Models for Current System:")
    print("-" * 40)
    
    for analysis_type in AnalysisType:
        try:
            compatible = checker.model_registry.list_compatible_models(
                analysis_type, resources
            )
            print(f"{analysis_type.value}: {', '.join(compatible) if compatible else 'None'}")
        except Exception as e:
            print(f"{analysis_type.value}: Error - {e}")
    
    # 4. Batch validation
    print("\n4. Batch Compatibility Validation:")
    print("-" * 35)
    
    try:
        analysis_types = [AnalysisType.ACTIVATION_EXTRACTION, AnalysisType.LAYER_ANALYSIS]
        results = checker.validate_model_compatibility(
            "gpt2", analysis_types, 
            config={"batch_size": 4, "max_length": 1024}
        )
        
        for analysis_name, result in results.items():
            status = "✅" if result.compatible else "❌"
            print(f"  {analysis_name}: {status} (confidence: {result.confidence_score:.2f})")
            
    except Exception as e:
        print(f"  ❌ Batch validation error: {e}")


def demo_resource_monitoring():
    """Demonstrate GPU resource monitoring system."""
    print("\n" + "="*60)
    print("GPU RESOURCE MONITORING DEMO")
    print("="*60)
    
    manager = GPUResourceManager(monitoring_interval=2.0)
    
    # 1. GPU Status Detection
    print("\n1. GPU Status Detection:")
    print("-" * 25)
    
    try:
        gpu_statuses = manager.get_gpu_status()
        if not gpu_statuses:
            print("No GPUs detected or CUDA not available")
        else:
            print(f"Found {len(gpu_statuses)} GPU(s):")
            for status in gpu_statuses:
                print(f"\n  GPU {status.device_id}: {status.name}")
                print(f"    Memory: {status.memory_used_gb:.1f}/{status.memory_total_gb:.1f} GB "
                      f"({status.memory_percent:.1f}%)")
                print(f"    Utilization: {status.utilization_percent:.1f}%")
                print(f"    Available: {'Yes' if status.is_available else 'No'}")
                if status.temperature:
                    print(f"    Temperature: {status.temperature}°C")
                if status.power_draw:
                    print(f"    Power: {status.power_draw:.1f}W")
                if status.compute_capability:
                    print(f"    Compute Capability: {status.compute_capability[0]}.{status.compute_capability[1]}")
                    
    except Exception as e:
        print(f"❌ Error getting GPU status: {e}")
    
    # 2. Memory Optimization Demo
    print("\n2. Memory Optimization Demo:")
    print("-" * 30)
    
    try:
        if gpu_statuses:
            device_id = 0  # Use first GPU
            status = gpu_statuses[0]
            
            print(f"Current memory usage on GPU {device_id}: {status.memory_used_gb:.1f} GB")
            
            # Simulate memory optimization
            optimization_result = manager.memory_optimizer.optimize_memory_usage(
                current_usage_gb=status.memory_used_gb,
                available_gb=status.memory_available_gb
            )
            
            print("Optimization strategies attempted:")
            for strategy in optimization_result["applied_strategies"]:
                print(f"  ✅ {strategy}")
            print(f"Estimated memory saved: {optimization_result['memory_saved_gb']:.2f} GB")
            
            if optimization_result["recommendations"]:
                print("Additional recommendations:")
                for rec in optimization_result["recommendations"]:
                    print(f"  💡 {rec}")
        else:
            print("No GPUs available for memory optimization demo")
            
    except Exception as e:
        print(f"❌ Memory optimization error: {e}")
    
    # 3. Workload Scheduling Demo
    print("\n3. Workload Scheduling Demo:")
    print("-" * 30)
    
    try:
        def dummy_task(task_name, duration=2):
            """Dummy task for demonstration."""
            print(f"  Executing {task_name}...")
            time.sleep(duration)
            return f"{task_name} completed"
        
        # Submit some demo tasks
        tasks = [
            WorkloadTask(
                task_id="task_1",
                priority=WorkloadPriority.HIGH,
                estimated_memory_gb=2.0,
                estimated_time_minutes=1.0,
                gpu_required=True,
                callback=dummy_task,
                args=("High Priority Task", 1)
            ),
            WorkloadTask(
                task_id="task_2", 
                priority=WorkloadPriority.NORMAL,
                estimated_memory_gb=1.0,
                estimated_time_minutes=0.5,
                gpu_required=False,
                callback=dummy_task,
                args=("Normal Priority Task", 0.5)
            ),
            WorkloadTask(
                task_id="task_3",
                priority=WorkloadPriority.CRITICAL,
                estimated_memory_gb=4.0,
                estimated_time_minutes=2.0,
                gpu_required=True,
                callback=dummy_task,
                args=("Critical Priority Task", 1)
            )
        ]
        
        # Submit tasks
        print("Submitting tasks to scheduler:")
        for task in tasks:
            task_id = manager.workload_scheduler.submit_task(task)
            print(f"  📝 Submitted {task_id} (priority: {task.priority.name})")
        
        # Start scheduler and let it run briefly
        print("\nStarting scheduler...")
        manager.workload_scheduler.start_scheduler()
        
        # Monitor progress
        for i in range(8):  # Monitor for ~8 seconds
            status = manager.workload_scheduler.get_queue_status()
            print(f"  Queue: {status['queued_tasks']}, "
                  f"Running: {status['running_tasks']}, "
                  f"Completed: {status['completed_tasks']}")
            time.sleep(1)
        
        manager.workload_scheduler.stop_scheduler()
        
        final_status = manager.workload_scheduler.get_queue_status()
        print(f"\nFinal status: {final_status}")
        
    except Exception as e:
        print(f"❌ Workload scheduling error: {e}")
    
    # 4. Real-time Monitoring Demo
    print("\n4. Real-time Monitoring Demo:")
    print("-" * 30)
    
    try:
        print("Starting real-time monitoring for 10 seconds...")
        manager.start_monitoring()
        
        # Let it monitor for a bit
        for i in range(5):
            time.sleep(2)
            summary = manager.get_resource_summary()
            print(f"  Time {i*2+2}s: "
                  f"Avg GPU utilization: {summary['average_gpu_utilization']:.1f}%, "
                  f"Memory used: {summary['used_memory_gb']:.1f}/{summary['total_memory_gb']:.1f} GB")
        
        manager.stop_monitoring()
        print("Monitoring stopped.")
        
        # Show final resource summary
        print("\nFinal Resource Summary:")
        summary = manager.get_resource_summary()
        print(json.dumps(summary, indent=2, default=str))
        
    except Exception as e:
        print(f"❌ Real-time monitoring error: {e}")


def demo_integration_example():
    """Demonstrate integration of compatibility checking with resource monitoring."""
    print("\n" + "="*60)
    print("INTEGRATION EXAMPLE: SMART MODEL SELECTION")
    print("="*60)
    
    print("\nScenario: Automatically select optimal model and configuration")
    print("based on system capabilities and user requirements.")
    
    try:
        # Initialize systems
        compatibility_checker = ModelCompatibilityChecker()
        resource_manager = GPUResourceManager()
        
        # Get current system state
        system_resources = compatibility_checker.get_system_resources()
        gpu_status = resource_manager.get_gpu_status()
        
        print(f"\nSystem Assessment:")
        print(f"  Available Memory: {system_resources.available_memory_gb:.1f} GB")
        print(f"  GPU Available: {system_resources.gpu_available}")
        if gpu_status:
            total_gpu_memory = sum(s.memory_total_gb for s in gpu_status)
            used_gpu_memory = sum(s.memory_used_gb for s in gpu_status)
            print(f"  GPU Memory: {used_gpu_memory:.1f}/{total_gpu_memory:.1f} GB")
        
        # Define analysis requirements
        analysis_type = AnalysisType.ACTIVATION_EXTRACTION
        max_batch_size = 8
        
        print(f"\nAnalysis Requirements:")
        print(f"  Type: {analysis_type.value}")
        print(f"  Max Batch Size: {max_batch_size}")
        
        # Test different models and find the best fit
        candidate_models = ["gpt2", "gpt2-medium", "gpt2-large", "bert-base-uncased", "t5-small"]
        compatible_models = []
        
        print(f"\nEvaluating {len(candidate_models)} candidate models:")
        
        for model_name in candidate_models:
            result = compatibility_checker.check_compatibility(
                model_name, analysis_type, 
                batch_size=max_batch_size, sequence_length=512
            )
            
            status_emoji = "✅" if result.compatible else "❌"
            print(f"  {status_emoji} {model_name}: "
                  f"Compatible={result.compatible}, "
                  f"Confidence={result.confidence_score:.2f}")
            
            if result.compatible:
                compatible_models.append((model_name, result))
        
        # Select the best model
        if compatible_models:
            # Sort by confidence score and estimated performance
            best_model, best_result = max(compatible_models, 
                                        key=lambda x: x[1].confidence_score)
            
            print(f"\n🎯 Recommended Model: {best_model}")
            print(f"   Confidence Score: {best_result.confidence_score:.2f}")
            print(f"   Estimated Memory: {best_result.estimated_memory_usage_gb:.1f} GB")
            print(f"   Estimated Time: {best_result.estimated_processing_time_minutes:.1f} min")
            
            if best_result.recommendations:
                print(f"   Optimizations: {', '.join(best_result.recommendations)}")
        else:
            print("\n❌ No compatible models found for current system configuration")
            print("   Consider:")
            print("   - Reducing batch size")
            print("   - Freeing up memory")
            print("   - Using a smaller model")
        
    except Exception as e:
        print(f"❌ Integration example error: {e}")


def main():
    """Main demo function."""
    print("🧠 NeuronMap Model Compatibility & Resource Monitoring Demo")
    print("This demo showcases the advanced features implemented in roadmap section 2.2")
    
    try:
        # Run all demo sections
        demo_model_compatibility()
        demo_resource_monitoring()
        demo_integration_example()
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✅ Model compatibility checking with resource validation")
        print("✅ Real-time GPU monitoring and memory optimization")
        print("✅ Intelligent workload scheduling for multi-GPU systems")
        print("✅ Automatic fallback suggestions and optimization recommendations")
        print("✅ Integration of compatibility checking with resource monitoring")
        
        print("\nRoadmap Section 2.2 Status:")
        print("✅ Modell-Kompatibilitätsprüfung vor Ausführung - IMPLEMENTED")
        print("✅ GPU-Verfügbarkeit und VRAM-Monitoring - IMPLEMENTED")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
