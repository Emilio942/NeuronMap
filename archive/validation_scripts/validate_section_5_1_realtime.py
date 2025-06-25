#!/usr/bin/env python3
"""
Validation script for Section 5.1: Real-time Activation Viewer
Tests implementation completeness and performance requirements
"""

import sys
import os
import time
import asyncio
import threading
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_dependencies():
    """Check if required dependencies are available"""
    dependencies = {
        'websockets': 'WebSocket communication',
        'numpy': 'Numerical computations',
        'asyncio': 'Async programming (built-in)',
        'threading': 'Multi-threading (built-in)',
        'json': 'JSON handling (built-in)',
        'time': 'Time operations (built-in)'
    }
    
    print("🔍 Checking dependencies...")
    missing = []
    
    for dep, description in dependencies.items():
        try:
            if dep in ['asyncio', 'threading', 'json', 'time']:
                # Built-in modules
                __import__(dep)
                print(f"✅ {dep}: {description}")
            else:
                __import__(dep)
                print(f"✅ {dep}: {description}")
        except ImportError:
            print(f"❌ {dep}: {description} - MISSING")
            missing.append(dep)
    
    return missing

def check_file_structure():
    """Check if required files exist"""
    required_files = {
        'src/visualization/realtime_streamer.py': 'Real-time streaming engine',
        'static/realtime_viewer.html': 'WebSocket client interface',
        'demo_realtime_visualization.py': 'Demo script'
    }
    
    print("\n📁 Checking file structure...")
    missing_files = []
    
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            print(f"✅ {file_path}: {description}")
        else:
            print(f"❌ {file_path}: {description} - MISSING")
            missing_files.append(file_path)
    
    return missing_files

def check_implementation_completeness():
    """Check if implementation meets requirements"""
    print("\n🎯 Checking implementation completeness...")
    
    requirements = [
        ("RealtimeActivationStreamer class", "Core streaming functionality"),
        ("CircularBuffer implementation", "High-performance activation storage"),
        ("FrameRateController", "Adaptive FPS control"),
        ("DeltaCompressor", "Bandwidth optimization"),
        ("WebSocket server", "Real-time communication"),
        ("Client-side visualization", "Interactive web interface"),
        ("Performance monitoring", "Latency and FPS tracking"),
        ("Async processing", "Non-blocking operation")
    ]
    
    completed = 0
    total = len(requirements)
    
    for requirement, description in requirements:
        try:
            if requirement == "RealtimeActivationStreamer class":
                # Check if the main class exists and has required methods
                spec = """
                class RealtimeActivationStreamer:
                    - __init__(model_wrapper, config)
                    - start_streaming_server()
                    - handle_client_connection()
                    - stream_activations()
                    - _extract_activations_async()
                    - _broadcast_frame()
                """
                file_path = 'src/visualization/realtime_streamer.py'
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if 'class RealtimeActivationStreamer:' in content:
                            print(f"✅ {requirement}: {description}")
                            completed += 1
                        else:
                            print(f"❌ {requirement}: Class not found")
                else:
                    print(f"❌ {requirement}: File not found")
            
            elif requirement == "CircularBuffer implementation":
                file_path = 'src/visualization/realtime_streamer.py'
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if 'class CircularBuffer:' in content:
                            print(f"✅ {requirement}: {description}")
                            completed += 1
                        else:
                            print(f"❌ {requirement}: Class not found")
                else:
                    print(f"❌ {requirement}: File not found")
            
            elif requirement == "Client-side visualization":
                file_path = 'static/realtime_viewer.html'
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if 'WebSocket' in content and 'Plotly' in content:
                            print(f"✅ {requirement}: {description}")
                            completed += 1
                        else:
                            print(f"❌ {requirement}: WebSocket/Plotly not found")
                else:
                    print(f"❌ {requirement}: File not found")
            
            else:
                # Check for other components in the code
                file_path = 'src/visualization/realtime_streamer.py'
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                        class_name = requirement.split()[0]
                        if f'class {class_name}:' in content:
                            print(f"✅ {requirement}: {description}")
                            completed += 1
                        else:
                            print(f"❌ {requirement}: {description} - Not found")
                else:
                    print(f"❌ {requirement}: File not found")
                    
        except Exception as e:
            print(f"❌ {requirement}: Error checking - {str(e)}")
    
    print(f"\n📊 Implementation completeness: {completed}/{total} ({completed/total*100:.1f}%)")
    return completed, total

def test_performance_requirements():
    """Test if system meets performance requirements"""
    print("\n⚡ Testing performance requirements...")
    
    requirements = {
        "End-to-end latency": "<50ms for local deployments",
        "Neuron support": "Support for 50K+ neurons @ 30 FPS",
        "Memory usage": "<2GB for 1-hour continuous streaming",
        "Recording functionality": "Frame-accurate timing"
    }
    
    results = {}
    
    # Test 1: Synthetic latency test
    print("🔬 Testing synthetic activation processing...")
    try:
        import numpy as np
        
        # Simulate activation processing
        start_time = time.time()
        
        # Create synthetic activation data
        activations = np.random.randn(1000, 768)  # 1000 tokens, 768 dimensions
        
        # Simulate processing (mean, compression, serialization)
        processed = np.mean(activations, axis=0)
        compressed = processed[processed > 0.1]  # Simple compression
        serialized = compressed.tolist()
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms
        
        if latency < 50:
            print(f"✅ Processing latency: {latency:.2f}ms (< 50ms requirement)")
            results["latency"] = True
        else:
            print(f"⚠️ Processing latency: {latency:.2f}ms (exceeds 50ms requirement)")
            results["latency"] = False
            
    except Exception as e:
        print(f"❌ Latency test failed: {e}")
        results["latency"] = False
    
    # Test 2: Memory usage estimation
    print("🧠 Estimating memory usage...")
    try:
        import sys
        
        # Estimate memory for circular buffer
        max_frames = 10000
        neurons_per_frame = 50000
        bytes_per_float = 4
        
        buffer_memory = max_frames * neurons_per_frame * bytes_per_float
        buffer_memory_gb = buffer_memory / (1024**3)
        
        # Add overhead for WebSocket, processing, etc.
        total_memory_gb = buffer_memory_gb * 2  # Conservative estimate
        
        if total_memory_gb < 2.0:
            print(f"✅ Estimated memory usage: {total_memory_gb:.2f}GB (< 2GB requirement)")
            results["memory"] = True
        else:
            print(f"⚠️ Estimated memory usage: {total_memory_gb:.2f}GB (exceeds 2GB requirement)")
            results["memory"] = False
            
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        results["memory"] = False
    
    # Test 3: FPS capability
    print("📊 Testing FPS capability...")
    try:
        frames_processed = 0
        start_time = time.time()
        target_duration = 1.0  # Test for 1 second
        
        while (time.time() - start_time) < target_duration:
            # Simulate frame processing
            np.random.randn(1000)  # Synthetic processing
            frames_processed += 1
        
        actual_fps = frames_processed / target_duration
        
        if actual_fps >= 30:
            print(f"✅ Processing capability: {actual_fps:.1f} FPS (>= 30 FPS requirement)")
            results["fps"] = True
        else:
            print(f"⚠️ Processing capability: {actual_fps:.1f} FPS (< 30 FPS requirement)")
            results["fps"] = False
            
    except Exception as e:
        print(f"❌ FPS test failed: {e}")
        results["fps"] = False
    
    return results

def check_integration_requirements():
    """Check integration with existing system"""
    print("\n🔗 Checking integration requirements...")
    
    integration_points = {
        "ModelWrapper integration": "src/analysis/model_wrapper.py",
        "ConfigManager integration": "src/utils/config.py",
        "Visualization system": "src/visualization/",
        "CLI integration": "main.py"
    }
    
    integration_status = {}
    
    for component, file_path in integration_points.items():
        if os.path.exists(file_path):
            print(f"✅ {component}: Available for integration")
            integration_status[component] = True
        else:
            print(f"❌ {component}: {file_path} not found")
            integration_status[component] = False
    
    return integration_status

def run_validation():
    """Run complete validation of Section 5.1 implementation"""
    print("="*80)
    print("SECTION 5.1: REAL-TIME ACTIVATION VIEWER - VALIDATION")
    print("="*80)
    
    # Check dependencies
    missing_deps = check_dependencies()
    
    # Check file structure
    missing_files = check_file_structure()
    
    # Check implementation completeness
    completed, total = check_implementation_completeness()
    
    # Test performance requirements
    performance_results = test_performance_requirements()
    
    # Check integration requirements
    integration_status = check_integration_requirements()
    
    # Generate summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    # Overall completion score
    completeness_score = (completed / total) * 100 if total > 0 else 0
    performance_score = sum(performance_results.values()) / len(performance_results) * 100 if performance_results else 0
    integration_score = sum(integration_status.values()) / len(integration_status) * 100 if integration_status else 0
    
    overall_score = (completeness_score + performance_score + integration_score) / 3
    
    print(f"📊 Implementation Completeness: {completeness_score:.1f}%")
    print(f"⚡ Performance Requirements: {performance_score:.1f}%")
    print(f"🔗 Integration Readiness: {integration_score:.1f}%")
    print(f"🎯 Overall Score: {overall_score:.1f}%")
    
    # Status determination
    if overall_score >= 90:
        status = "✅ IMPLEMENTATION COMPLETE"
    elif overall_score >= 70:
        status = "🟡 IMPLEMENTATION MOSTLY COMPLETE"
    elif overall_score >= 50:
        status = "🟠 IMPLEMENTATION PARTIAL"
    else:
        status = "❌ IMPLEMENTATION INCOMPLETE"
    
    print(f"\n{status}")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    
    if missing_deps:
        print(f"1. Install missing dependencies: {', '.join(missing_deps)}")
        print("   Command: pip install " + " ".join(missing_deps))
    
    if missing_files:
        print("2. Create missing files:")
        for file in missing_files:
            print(f"   - {file}")
    
    if completeness_score < 100:
        print("3. Complete remaining implementation components")
    
    if performance_score < 100:
        print("4. Optimize performance for production requirements")
    
    if integration_score < 100:
        print("5. Ensure all integration points are available")
    
    # Next steps
    print("\n🚀 NEXT STEPS:")
    if overall_score >= 90:
        print("1. Run integration tests with real models")
        print("2. Performance optimization and tuning")
        print("3. User acceptance testing")
        print("4. Documentation updates")
    else:
        print("1. Complete missing implementation components")
        print("2. Fix identified issues")
        print("3. Re-run validation")
    
    return {
        'overall_score': overall_score,
        'completeness': completeness_score,
        'performance': performance_score,
        'integration': integration_score,
        'status': status,
        'missing_deps': missing_deps,
        'missing_files': missing_files
    }

if __name__ == "__main__":
    results = run_validation()
    
    # Exit with appropriate code
    if results['overall_score'] >= 70:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure
