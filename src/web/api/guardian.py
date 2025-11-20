from flask import Blueprint, jsonify, request
import yaml
import logging
from pathlib import Path
from src.utils.config_manager import get_config
import glob
import json
import os
from datetime import datetime
from src.analysis.overthinking import OverthinkingDetector

logger = logging.getLogger(__name__)

guardian_bp = Blueprint('guardian', __name__, url_prefix='/api/guardian')

@guardian_bp.route('/config', methods=['GET'])
def get_config_route():
    """Get current Guardian configuration."""
    try:
        config = get_config()
        if hasattr(config, 'guardian'):
            # Use model_dump() for Pydantic v2 or dict() for v1
            guardian_data = config.guardian.model_dump() if hasattr(config.guardian, 'model_dump') else config.guardian.dict()
            return jsonify(guardian_data)
        return jsonify({'enabled': False, 'mode': 'monitoring'})
    except Exception as e:
        logger.error(f"Error getting guardian config: {e}")
        return jsonify({'error': str(e)}), 500

@guardian_bp.route('/config', methods=['POST'])
def update_config_route():
    """Update Guardian configuration."""
    try:
        data = request.json
        config = get_config()
        
        if not hasattr(config, 'guardian'):
            return jsonify({'error': 'Guardian not configured'}), 404
            
        # Update in-memory config
        guardian_config = config.guardian
        for key, value in data.items():
            if hasattr(guardian_config, key):
                setattr(guardian_config, key, value)
                
        # Save to runtime config file
        # This allows the streaming process to potentially reload it if it watches this file
        runtime_config_path = Path("configs/guardian_runtime.yaml")
        runtime_config_path.parent.mkdir(exist_ok=True)
        
        guardian_data = guardian_config.model_dump() if hasattr(guardian_config, 'model_dump') else guardian_config.dict()
        
        with open(runtime_config_path, 'w') as f:
            yaml.dump({'guardian': guardian_data}, f)
            
        logger.info(f"Guardian config updated: {guardian_data}")
            
        return jsonify({'status': 'updated', 'config': guardian_data})
    except Exception as e:
        logger.error(f"Error updating guardian config: {e}")
        return jsonify({'error': str(e)}), 500

@guardian_bp.route('/traces', methods=['GET'])
def list_traces():
    """List available SwiReasoning trace files."""
    try:
        trace_dir = Path("outputs/benchmarks")
        if not trace_dir.exists():
            return jsonify([])
            
        files = glob.glob(str(trace_dir / "swireasoning_traces_*.json"))
        traces = []
        for f in files:
            path = Path(f)
            traces.append({
                'filename': path.name,
                'timestamp': int(path.stem.split('_')[-1]),
                'date': datetime.fromtimestamp(int(path.stem.split('_')[-1])).strftime('%Y-%m-%d %H:%M:%S'),
                'size': path.stat().st_size
            })
            
        # Sort by timestamp desc
        traces.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify(traces)
    except Exception as e:
        logger.error(f"Error listing traces: {e}")
        return jsonify({'error': str(e)}), 500

@guardian_bp.route('/traces/<filename>', methods=['GET'])
def get_trace(filename):
    """Get content of a specific trace file."""
    try:
        # Security check: ensure filename is just a filename, not a path
        if '/' in filename or '\\' in filename:
            return jsonify({'error': 'Invalid filename'}), 400
            
        file_path = Path("outputs/benchmarks") / filename
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error reading trace: {e}")
        return jsonify({'error': str(e)}), 500

@guardian_bp.route('/analyze_trace', methods=['POST'])
def analyze_trace():
    """Analyze a trace for overthinking."""
    try:
        trace_data = request.json
        detector = OverthinkingDetector()
        analysis = detector.analyze_trace(trace_data)
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Error analyzing trace: {e}")
        return jsonify({'error': str(e)}), 500
