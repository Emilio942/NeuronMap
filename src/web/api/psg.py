"""
PSG Analysis API Blueprint
==========================

API endpoints for Parameter Sparsity Gate analysis.
"""

from flask import Blueprint, request, jsonify, url_for
import logging
from pathlib import Path
import threading
import uuid

# Import analysis tools
from analysis.model_integration import get_model_manager
from analysis.psg_detector import PSGDetector
from visualization.psg_visualizer import PSGVisualizer

logger = logging.getLogger(__name__)

psg_bp = Blueprint('psg', __name__, url_prefix='/api/psg')

@psg_bp.route('/analyze', methods=['POST'])
def run_psg_analysis():
    """Run PSG analysis and return visualization URLs."""
    try:
        data = request.json
        model_name = data.get('model', 'distilgpt2')
        prompt = data.get('prompt', 'The quick brown fox jumps over the lazy dog.')
        threshold = float(data.get('threshold', 0.01))
        
        # Generate unique ID for this run
        run_id = str(uuid.uuid4())[:8]
        output_dir = Path(f"web/static/psg_outputs/{run_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        model_manager = get_model_manager()
        adapter = model_manager.load_model(model_name)
        
        # Run Detection
        detector = PSGDetector(adapter.model, adapter.tokenizer, device=model_manager.device)
        psgs = detector.detect([prompt], weight_threshold=threshold)
        
        # Generate Visualizations
        viz = PSGVisualizer(output_dir=str(output_dir))
        viz.visualize_structure(psgs, filename="structure.html")
        viz.visualize_reaction(psgs, prompt, filename="reaction.html")
        
        return jsonify({
            'success': True,
            'structure_url': url_for('static', filename=f'psg_outputs/{run_id}/structure.html'),
            'reaction_url': url_for('static', filename=f'psg_outputs/{run_id}/reaction.html'),
            'psg_count': len(psgs)
        })
        
    except Exception as e:
        logger.error(f"PSG Analysis failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
