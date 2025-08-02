"""
Interpretability Analysis Module
==============================

Advanced interpretability tools for neural network analysis including
attribution methods, concept analysis, and mechanistic interpretability.
"""

# Attribution methods
try:
    from .ig_explainer import IntegratedGradientsExplainer, create_ig_explainer
    IG_AVAILABLE = True
except ImportError:
    IG_AVAILABLE = False

try:
    from .shap_explainer import DeepSHAPExplainer, create_shap_explainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from .semantic_labeling import SemanticLabeler, create_semantic_labeler
    SEMANTIC_LABELING_AVAILABLE = True
except ImportError:
    SEMANTIC_LABELING_AVAILABLE = False

# Export available components
__all__ = []

if IG_AVAILABLE:
    __all__.extend(['IntegratedGradientsExplainer', 'create_ig_explainer'])

if SHAP_AVAILABLE:
    __all__.extend(['DeepSHAPExplainer', 'create_shap_explainer'])

if SEMANTIC_LABELING_AVAILABLE:
    __all__.extend(['SemanticLabeler', 'create_semantic_labeler'])

# Factory function for creating interpretability tools
def create_interpretability_tool(tool_name: str, config=None):
    """Factory function to create interpretability tools."""
    if tool_name == "integrated_gradients" and IG_AVAILABLE:
        return create_ig_explainer(config)
    elif tool_name == "deep_shap" and SHAP_AVAILABLE:
        return create_shap_explainer(config)
    elif tool_name == "semantic_labeling" and SEMANTIC_LABELING_AVAILABLE:
        return create_semantic_labeler(config)
    else:
        raise ValueError(f"Unknown or unavailable interpretability tool: {tool_name}")

def get_available_tools():
    """Get list of available interpretability tools."""
    tools = []
    if IG_AVAILABLE:
        tools.append("integrated_gradients")
    if SHAP_AVAILABLE:
        tools.append("deep_shap")
    if SEMANTIC_LABELING_AVAILABLE:
        tools.append("semantic_labeling")
    return tools
