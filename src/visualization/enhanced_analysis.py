"""
Integration Script for Neuron Group Analysis
===========================================

This script integrates the new neuron group visualization capabilities
into the existing NeuronMap analysis workflow.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

try:
    from .neuron_group_visualizer import NeuronGroupVisualizer, create_neuron_group_analysis
    NEURON_GROUP_AVAILABLE = True
except ImportError:
    NEURON_GROUP_AVAILABLE = False
    logger.warning("Neuron group visualizer not available")

try:
    import numpy as np
    import pandas as pd
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    logger.warning("NumPy/Pandas not available for neuron group analysis")


class EnhancedAnalysisWorkflow:
    """Enhanced analysis workflow including neuron group analysis."""
    
    def __init__(self, config=None):
        """Initialize enhanced analysis workflow.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def run_complete_analysis(self, 
                            activation_data: Dict[str, Any],
                            include_neuron_groups: bool = True,
                            output_dir: str = "data/outputs") -> Dict[str, Any]:
        """Run complete analysis including neuron group identification.
        
        Args:
            activation_data: Dictionary containing activation matrices and metadata
            include_neuron_groups: Whether to include neuron group analysis
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing all analysis results
        """
        results = {
            'traditional_analysis': {},
            'neuron_group_analysis': {},
            'combined_insights': {},
            'output_paths': []
        }
        
        try:
            # Run traditional analysis first
            self.logger.info("Running traditional activation analysis...")
            traditional_results = self._run_traditional_analysis(activation_data, output_dir)
            results['traditional_analysis'] = traditional_results
            
            # Run neuron group analysis if available and requested
            if include_neuron_groups and NEURON_GROUP_AVAILABLE and DEPENDENCIES_AVAILABLE:
                self.logger.info("Running neuron group analysis...")
                group_results = self._run_neuron_group_analysis(activation_data, output_dir)
                results['neuron_group_analysis'] = group_results
                
                # Generate combined insights
                combined_insights = self._generate_combined_insights(
                    traditional_results, group_results
                )
                results['combined_insights'] = combined_insights
            else:
                self.logger.warning("Skipping neuron group analysis (dependencies not available)")
            
            # Generate comprehensive report
            report_path = self._generate_comprehensive_report(results, output_dir)
            results['comprehensive_report'] = report_path
            
            self.logger.info("Complete analysis finished successfully")
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _run_traditional_analysis(self, activation_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Run traditional NeuronMap analysis."""
        try:
            # Import existing analysis components
            from ..analysis.advanced_analysis import AdvancedAnalyzer
            from ..visualization.activation_visualizer import ActivationVisualizer
            
            results = {}
            
            # Run advanced analysis
            analyzer = AdvancedAnalyzer()
            
            for layer_name, layer_data in activation_data.get('activations', {}).items():
                if isinstance(layer_data, np.ndarray):
                    # Clustering analysis
                    clustering_results = analyzer.perform_clustering_analysis(
                        layer_data, n_clusters=5, methods=['kmeans', 'hierarchical']
                    )
                    results[f'{layer_name}_clustering'] = clustering_results
                    
                    # Statistical analysis
                    stats = analyzer.compute_activation_statistics(layer_data)
                    results[f'{layer_name}_statistics'] = stats
            
            # Create traditional visualizations
            if self.config:
                visualizer = ActivationVisualizer(self.config)
                visualizer.generate_all_visualizations()
                results['visualizations_created'] = True
            
            return results
            
        except Exception as e:
            self.logger.error(f"Traditional analysis failed: {e}")
            return {'error': str(e)}
    
    def _run_neuron_group_analysis(self, activation_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Run neuron group analysis."""
        group_output_dir = Path(output_dir) / "neuron_groups"
        
        results = {}
        
        for layer_name, layer_data in activation_data.get('activations', {}).items():
            if isinstance(layer_data, np.ndarray) and layer_data.ndim == 2:
                self.logger.info(f"Analyzing neuron groups for layer: {layer_name}")
                
                # Get question metadata if available
                question_metadata = activation_data.get('metadata')
                
                # Run neuron group analysis
                layer_output_dir = group_output_dir / layer_name.replace('.', '_')
                
                try:
                    group_results = create_neuron_group_analysis(
                        activation_matrix=layer_data,
                        question_metadata=question_metadata,
                        output_dir=str(layer_output_dir),
                        config=self.config
                    )
                    
                    results[layer_name] = group_results
                    self.logger.info(f"Neuron group analysis completed for {layer_name}")
                    
                except Exception as e:
                    self.logger.error(f"Neuron group analysis failed for {layer_name}: {e}")
                    results[layer_name] = {'error': str(e)}
        
        return results
    
    def _generate_combined_insights(self, 
                                  traditional_results: Dict[str, Any],
                                  group_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights by combining traditional and group analysis results."""
        insights = {
            'cross_analysis_findings': [],
            'layer_comparisons': {},
            'learning_progression': {},
            'functional_specialization': {}
        }
        
        try:
            # Compare clustering results from both approaches
            for layer_name in group_results.keys():
                if 'error' not in group_results[layer_name]:
                    layer_insights = self._analyze_layer_insights(
                        layer_name, traditional_results, group_results[layer_name]
                    )
                    insights['layer_comparisons'][layer_name] = layer_insights
            
            # Analyze functional specialization
            insights['functional_specialization'] = self._analyze_functional_specialization(
                group_results
            )
            
            # Analyze learning progression patterns
            insights['learning_progression'] = self._analyze_learning_progression(
                group_results
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate combined insights: {e}")
            insights['error'] = str(e)
        
        return insights
    
    def _analyze_layer_insights(self, 
                              layer_name: str,
                              traditional_results: Dict[str, Any],
                              group_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze insights for a specific layer."""
        insights = {
            'traditional_clusters': 0,
            'neuron_groups': len(group_results.get('neuron_groups', [])),
            'learning_events': len(group_results.get('learning_events', [])),
            'grouping_efficiency': 0.0,
            'specialization_level': 0.0
        }
        
        # Get traditional clustering info
        clustering_key = f'{layer_name}_clustering'
        if clustering_key in traditional_results:
            clustering_data = traditional_results[clustering_key]
            if isinstance(clustering_data, dict) and 'kmeans' in clustering_data:
                insights['traditional_clusters'] = clustering_data['kmeans'].get('n_clusters', 0)
        
        # Calculate grouping efficiency
        if 'summary' in group_results:
            summary = group_results['summary']
            if 'total_groups' in summary:
                insights['grouping_efficiency'] = summary.get('grouping_efficiency', 0.0)
        
        # Analyze specialization level based on learning events
        learning_events = group_results.get('learning_events', [])
        if learning_events:
            skill_types = set()
            for event in learning_events:
                if hasattr(event, 'skill_type'):
                    skill_types.add(event.skill_type)
            
            insights['specialization_level'] = len(skill_types) / max(len(learning_events), 1)
        
        return insights
    
    def _analyze_functional_specialization(self, group_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze functional specialization across layers."""
        specialization = {
            'layer_specialization_scores': {},
            'skill_distribution': {},
            'cross_layer_patterns': []
        }
        
        all_skills = set()
        layer_skills = {}
        
        for layer_name, layer_data in group_results.items():
            if 'error' in layer_data:
                continue
                
            learning_events = layer_data.get('learning_events', [])
            layer_skill_counts = {}
            
            for event in learning_events:
                if hasattr(event, 'skill_type'):
                    skill = event.skill_type
                    all_skills.add(skill)
                    layer_skill_counts[skill] = layer_skill_counts.get(skill, 0) + 1
            
            layer_skills[layer_name] = layer_skill_counts
            
            # Calculate specialization score (entropy-based)
            if layer_skill_counts:
                total_events = sum(layer_skill_counts.values())
                entropy = 0.0
                for count in layer_skill_counts.values():
                    p = count / total_events
                    if p > 0:
                        entropy -= p * np.log2(p)
                
                # Lower entropy = higher specialization
                max_entropy = np.log2(len(layer_skill_counts)) if len(layer_skill_counts) > 1 else 1
                specialization_score = 1.0 - (entropy / max_entropy)
                specialization['layer_specialization_scores'][layer_name] = specialization_score
        
        # Overall skill distribution
        specialization['skill_distribution'] = dict(layer_skills)
        
        return specialization
    
    def _analyze_learning_progression(self, group_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze learning progression patterns across layers."""
        progression = {
            'temporal_patterns': {},
            'skill_emergence': {},
            'learning_efficiency': {}
        }
        
        for layer_name, layer_data in group_results.items():
            if 'error' in layer_data:
                continue
            
            learning_events = layer_data.get('learning_events', [])
            if not learning_events:
                continue
            
            # Analyze temporal patterns
            positions = [getattr(event, 'temporal_position', 0) for event in learning_events]
            strengths = [getattr(event, 'learning_strength', 0) for event in learning_events]
            
            if positions and strengths:
                # Calculate learning trend
                if len(positions) > 1:
                    correlation = np.corrcoef(positions, strengths)[0, 1]
                    progression['temporal_patterns'][layer_name] = {
                        'trend': 'increasing' if correlation > 0.1 else 'decreasing' if correlation < -0.1 else 'stable',
                        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                        'peak_position': positions[np.argmax(strengths)],
                        'average_strength': np.mean(strengths)
                    }
        
        return progression
    
    def _generate_comprehensive_report(self, results: Dict[str, Any], output_dir: str) -> str:
        """Generate a comprehensive analysis report."""
        report_path = Path(output_dir) / "comprehensive_analysis_report.md"
        
        try:
            with open(report_path, 'w') as f:
                f.write("# Comprehensive Neural Network Analysis Report\n\n")
                f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Executive Summary
                f.write("## Executive Summary\n\n")
                self._write_executive_summary(f, results)
                
                # Traditional Analysis Results
                f.write("## Traditional Analysis Results\n\n")
                self._write_traditional_results(f, results.get('traditional_analysis', {}))
                
                # Neuron Group Analysis Results
                if results.get('neuron_group_analysis'):
                    f.write("## Neuron Group Analysis Results\n\n")
                    self._write_neuron_group_results(f, results['neuron_group_analysis'])
                
                # Combined Insights
                if results.get('combined_insights'):
                    f.write("## Combined Insights\n\n")
                    self._write_combined_insights(f, results['combined_insights'])
                
                # Recommendations
                f.write("## Recommendations\n\n")
                self._write_recommendations(f, results)
                
            self.logger.info(f"Comprehensive report saved to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive report: {e}")
            return ""
    
    def _write_executive_summary(self, f, results: Dict[str, Any]):
        """Write executive summary section."""
        f.write("This report presents a comprehensive analysis of neural network activations ")
        f.write("using both traditional clustering methods and advanced neuron group identification.\n\n")
        
        # Count key metrics
        total_groups = 0
        total_events = 0
        analyzed_layers = 0
        
        for layer_data in results.get('neuron_group_analysis', {}).values():
            if isinstance(layer_data, dict) and 'summary' in layer_data:
                total_groups += layer_data['summary'].get('total_groups', 0)
                total_events += layer_data['summary'].get('total_learning_events', 0)
                analyzed_layers += 1
        
        f.write(f"**Key Findings:**\n")
        f.write(f"- Analyzed {analyzed_layers} neural network layers\n")
        f.write(f"- Identified {total_groups} distinct neuron groups\n")
        f.write(f"- Detected {total_events} learning events\n")
        f.write(f"- Generated comprehensive visualizations and interactive dashboards\n\n")
    
    def _write_traditional_results(self, f, traditional_results: Dict[str, Any]):
        """Write traditional analysis results section."""
        if not traditional_results:
            f.write("Traditional analysis results not available.\n\n")
            return
        
        f.write("Traditional clustering and statistical analysis provided baseline insights ")
        f.write("into activation patterns and neuron behavior.\n\n")
        
        for key, value in traditional_results.items():
            if isinstance(value, dict) and 'error' not in value:
                f.write(f"### {key}\n\n")
                # Write summary of traditional results
                if 'n_clusters' in value:
                    f.write(f"- Identified {value['n_clusters']} clusters\n")
                if 'inertia' in value:
                    f.write(f"- Clustering inertia: {value['inertia']:.3f}\n")
                f.write("\n")
    
    def _write_neuron_group_results(self, f, neuron_group_results: Dict[str, Any]):
        """Write neuron group analysis results section."""
        f.write("Advanced neuron group analysis revealed functional organization ")
        f.write("and learning patterns within the neural network.\n\n")
        
        for layer_name, layer_data in neuron_group_results.items():
            if isinstance(layer_data, dict) and 'error' not in layer_data:
                f.write(f"### Layer: {layer_name}\n\n")
                
                summary = layer_data.get('summary', {})
                f.write(f"- **Neuron Groups:** {summary.get('total_groups', 0)}\n")
                f.write(f"- **Learning Events:** {summary.get('total_learning_events', 0)}\n")
                
                # Write about visualizations
                viz_paths = layer_data.get('visualizations', {})
                if viz_paths:
                    f.write(f"- **Generated Visualizations:**\n")
                    for viz_type, path in viz_paths.items():
                        if path:
                            f.write(f"  - {viz_type.title()}: `{Path(path).name}`\n")
                
                f.write("\n")
    
    def _write_combined_insights(self, f, combined_insights: Dict[str, Any]):
        """Write combined insights section."""
        f.write("Combined analysis reveals deeper patterns by integrating traditional ")
        f.write("and neuron group analysis approaches.\n\n")
        
        # Layer comparisons
        layer_comparisons = combined_insights.get('layer_comparisons', {})
        if layer_comparisons:
            f.write("### Layer-wise Analysis\n\n")
            for layer_name, insights in layer_comparisons.items():
                f.write(f"**{layer_name}:**\n")
                f.write(f"- Traditional clusters: {insights.get('traditional_clusters', 0)}\n")
                f.write(f"- Neuron groups: {insights.get('neuron_groups', 0)}\n")
                f.write(f"- Learning events: {insights.get('learning_events', 0)}\n")
                f.write(f"- Grouping efficiency: {insights.get('grouping_efficiency', 0):.2%}\n\n")
        
        # Functional specialization
        specialization = combined_insights.get('functional_specialization', {})
        if specialization:
            f.write("### Functional Specialization\n\n")
            spec_scores = specialization.get('layer_specialization_scores', {})
            if spec_scores:
                f.write("Layer specialization scores:\n")
                for layer, score in spec_scores.items():
                    f.write(f"- {layer}: {score:.3f}\n")
                f.write("\n")
    
    def _write_recommendations(self, f, results: Dict[str, Any]):
        """Write recommendations section."""
        f.write("Based on the comprehensive analysis, the following recommendations are provided:\n\n")
        
        # Analyze results to generate recommendations
        recommendations = []
        
        # Check if neuron group analysis was successful
        neuron_groups_available = bool(results.get('neuron_group_analysis'))
        
        if neuron_groups_available:
            recommendations.append(
                "✅ **Neuron Group Analysis**: Successfully identified functional neuron groups. "
                "Consider using these insights for model optimization and interpretability."
            )
        else:
            recommendations.append(
                "⚠️ **Missing Dependencies**: Install required libraries (scikit-learn, plotly, networkx) "
                "to enable advanced neuron group analysis."
            )
        
        # Check for learning patterns
        total_events = sum(
            layer_data.get('summary', {}).get('total_learning_events', 0)
            for layer_data in results.get('neuron_group_analysis', {}).values()
            if isinstance(layer_data, dict)
        )
        
        if total_events > 0:
            recommendations.append(
                f"📈 **Learning Patterns**: Detected {total_events} learning events. "
                "Explore the interactive dashboard to understand temporal learning patterns."
            )
        
        recommendations.extend([
            "📊 **Visualization**: Review generated heatmaps and network visualizations "
            "to understand neuron group organization.",
            "🔍 **Further Analysis**: Consider running analysis on different model layers "
            "or with different hyperparameters to explore robustness.",
            "🎯 **Model Optimization**: Use identified neuron groups to guide model "
            "pruning or architecture modifications."
        ])
        
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n\n")


def integrate_neuron_group_analysis(existing_config=None) -> bool:
    """Integrate neuron group analysis into existing NeuronMap workflow.
    
    Args:
        existing_config: Existing NeuronMap configuration
        
    Returns:
        True if integration successful, False otherwise
    """
    logger.info("Integrating neuron group analysis into NeuronMap workflow")
    
    try:
        # Check dependencies
        if not NEURON_GROUP_AVAILABLE:
            logger.error("Neuron group visualizer not available")
            return False
        
        if not DEPENDENCIES_AVAILABLE:
            logger.error("Required dependencies (NumPy/Pandas) not available")
            return False
        
        # Create enhanced workflow
        workflow = EnhancedAnalysisWorkflow(config=existing_config)
        
        logger.info("Neuron group analysis integration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        return False


# Export main classes and functions
__all__ = [
    'EnhancedAnalysisWorkflow',
    'integrate_neuron_group_analysis',
    'NEURON_GROUP_AVAILABLE',
    'DEPENDENCIES_AVAILABLE'
]
