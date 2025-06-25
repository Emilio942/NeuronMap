"""
Advanced Reporting System
========================

Generate comprehensive PDF reports, data exports, and documentation
for neural network analysis results.
"""

import io
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import base64

# Try to import reporting dependencies
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdvancedReporter:
    """Advanced reporting system for neural network analysis."""

    def __init__(self, output_dir: str = "data/outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Report templates
        self.templates = {
            'analysis_summary': {
                'title': 'Neural Network Analysis Summary',
                'sections': ['overview', 'methodology', 'results', 'visualizations', 'conclusions']
            },
            'model_comparison': {
                'title': 'Multi-Model Comparison Report',
                'sections': ['executive_summary', 'models_tested', 'performance_metrics', 'detailed_analysis', 'recommendations']
            },
            'performance_audit': {
                'title': 'System Performance Audit',
                'sections': ['system_overview', 'resource_utilization', 'bottlenecks', 'optimization_recommendations']
            }
        }

        # Report styles
        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self.title_style = ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1,  # Center
                textColor=colors.HexColor('#2c3e50')
            )
            self.heading_style = ParagraphStyle(
                'CustomHeading',
                parent=self.styles['Heading2'],
                fontSize=16,
                spaceBefore=20,
                spaceAfter=12,
                textColor=colors.HexColor('#34495e')
            )

    def generate_comprehensive_report(self,
                                    analysis_results: Dict[str, Any],
                                    template: str = 'analysis_summary',
                                    include_visualizations: bool = True) -> str:
        """Generate comprehensive PDF report from analysis results."""
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available. Generating text report instead.")
            return self._generate_text_report(analysis_results, template)

        try:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{template}_report_{timestamp}.pdf"
            output_file = self.output_dir / filename

            # Create document
            doc = SimpleDocTemplate(
                str(output_file),
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )

            # Build story
            story = []
            template_config = self.templates.get(template, self.templates['analysis_summary'])

            # Title
            story.append(Paragraph(template_config['title'], self.title_style))
            story.append(Spacer(1, 12))

            # Metadata table
            metadata = [
                ['Generated', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ['Model', analysis_results.get('model_name', 'Unknown')],
                ['Questions Analyzed', str(analysis_results.get('num_questions', 0))],
                ['Layers Analyzed', str(len(analysis_results.get('layer_names', [])))]
            ]

            metadata_table = Table(metadata)
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(metadata_table)
            story.append(Spacer(1, 20))

            # Add sections based on template
            for section in template_config['sections']:
                story.extend(self._generate_section(section, analysis_results))

            # Add visualizations if requested
            if include_visualizations and 'visualizations' in analysis_results:
                story.extend(self._add_visualizations_to_report(analysis_results['visualizations']))

            # Build PDF
            doc.build(story)

            logger.info(f"Comprehensive report generated: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return ""

    def _generate_section(self, section_name: str, data: Dict[str, Any]) -> List[Any]:
        """Generate a specific section of the report."""
        section_content = []

        # Section title
        section_content.append(Paragraph(section_name.replace('_', ' ').title(), self.heading_style))

        if section_name == 'overview':
            content = f"""
            This report presents a comprehensive analysis of neural network activations.
            The analysis was performed on {data.get('num_questions', 0)} questions using the
            {data.get('model_name', 'specified')} model.

            Key findings include:
            • Average activation level: {data.get('avg_activation', 0):.4f}
            • Maximum activation: {data.get('max_activation', 0):.4f}
            • Number of layers analyzed: {len(data.get('layer_names', []))}
            """

        elif section_name == 'methodology':
            content = f"""
            Analysis Methodology:

            Model Configuration:
            • Model: {data.get('model_name', 'Not specified')}
            • Device: {data.get('device', 'Not specified')}
            • Precision: {data.get('precision', 'float32')}

            Analysis Parameters:
            • Target layers: {', '.join(data.get('target_layers', ['All']))}
            • Analysis type: {data.get('analysis_type', 'Standard activation analysis')}
            • Aggregation method: {data.get('aggregation', 'mean')}
            """

        elif section_name == 'results':
            # Create results table
            if 'layer_stats' in data:
                layer_stats = data['layer_stats']
                results_data = [['Layer', 'Mean Activation', 'Std Deviation', 'Max Activation']]

                for layer, stats in layer_stats.items():
                    results_data.append([
                        layer,
                        f"{stats.get('mean', 0):.4f}",
                        f"{stats.get('std', 0):.4f}",
                        f"{stats.get('max', 0):.4f}"
                    ])

                results_table = Table(results_data)
                results_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))

                section_content.append(results_table)
                section_content.append(Spacer(1, 12))

            content = "Detailed numerical results are presented in the table above."

        elif section_name == 'conclusions':
            content = f"""
            Analysis Conclusions:

            Based on the activation analysis, we observe:
            1. The model shows {data.get('activation_pattern', 'varied')} activation patterns across layers
            2. {'High' if data.get('max_activation', 0) > 1.0 else 'Moderate'} maximum activation values
            3. {'Consistent' if data.get('activation_variance', 1) < 0.5 else 'Variable'} behavior across questions

            Recommendations:
            • Consider layer-specific optimization for improved performance
            • Monitor activation levels for potential saturation issues
            • Evaluate model behavior on diverse question types
            """

        else:
            content = f"Content for {section_name} section would be generated based on specific analysis results."

        section_content.append(Paragraph(content, self.styles['Normal']))
        section_content.append(Spacer(1, 12))

        return section_content

    def _add_visualizations_to_report(self, visualizations: List[str]) -> List[Any]:
        """Add visualizations to the report."""
        viz_content = []

        viz_content.append(Paragraph("Visualizations", self.heading_style))

        for viz_path in visualizations:
            if Path(viz_path).exists() and viz_path.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = Image(viz_path, width=6*inch, height=4*inch)
                    viz_content.append(img)
                    viz_content.append(Spacer(1, 12))
                except Exception as e:
                    logger.warning(f"Could not add visualization {viz_path}: {e}")

        return viz_content

    def _generate_text_report(self, analysis_results: Dict[str, Any], template: str) -> str:
        """Generate text-based report when PDF generation is not available."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{template}_report_{timestamp}.txt"
            output_file = self.output_dir / filename

            template_config = self.templates.get(template, self.templates['analysis_summary'])

            with open(output_file, 'w') as f:
                f.write(f"{template_config['title']}\n")
                f.write("=" * len(template_config['title']) + "\n\n")

                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {analysis_results.get('model_name', 'Unknown')}\n")
                f.write(f"Questions: {analysis_results.get('num_questions', 0)}\n")
                f.write(f"Layers: {len(analysis_results.get('layer_names', []))}\n\n")

                for section in template_config['sections']:
                    f.write(f"{section.replace('_', ' ').title()}\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"[Content for {section} section]\n\n")

            logger.info(f"Text report generated: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Error generating text report: {e}")
            return ""

    def export_data_to_csv(self, analysis_results: Dict[str, Any], filename_prefix: str = "analysis") -> List[str]:
        """Export analysis data to CSV files."""
        exported_files = []

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Export activations
            if 'activations' in analysis_results:
                activations_file = self.output_dir / f"{filename_prefix}_activations_{timestamp}.csv"
                activations_df = pd.DataFrame(analysis_results['activations'])
                activations_df.to_csv(activations_file, index=False)
                exported_files.append(str(activations_file))

            # Export layer statistics
            if 'layer_stats' in analysis_results:
                stats_file = self.output_dir / f"{filename_prefix}_layer_stats_{timestamp}.csv"
                stats_df = pd.DataFrame(analysis_results['layer_stats']).T
                stats_df.to_csv(stats_file)
                exported_files.append(str(stats_file))

            # Export metadata
            metadata_file = self.output_dir / f"{filename_prefix}_metadata_{timestamp}.json"
            metadata = {
                'model_name': analysis_results.get('model_name'),
                'num_questions': analysis_results.get('num_questions'),
                'layer_names': analysis_results.get('layer_names'),
                'timestamp': datetime.now().isoformat(),
                'device': analysis_results.get('device'),
                'analysis_type': analysis_results.get('analysis_type')
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            exported_files.append(str(metadata_file))

            logger.info(f"Exported {len(exported_files)} data files")
            return exported_files

        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return []

    def generate_executive_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate executive summary for stakeholders."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"executive_summary_{timestamp}.md"
            output_file = self.output_dir / filename

            with open(output_file, 'w') as f:
                f.write("# Neural Network Analysis - Executive Summary\n\n")

                f.write("## Key Findings\n\n")
                f.write(f"- **Model Analyzed**: {analysis_results.get('model_name', 'Unknown')}\n")
                f.write(f"- **Questions Processed**: {analysis_results.get('num_questions', 0)}\n")
                f.write(f"- **Layers Analyzed**: {len(analysis_results.get('layer_names', []))}\n")
                f.write(f"- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d')}\n\n")

                f.write("## Performance Metrics\n\n")
                if 'performance_metrics' in analysis_results:
                    metrics = analysis_results['performance_metrics']
                    f.write(f"- **Processing Time**: {metrics.get('total_time', 0):.2f} seconds\n")
                    f.write(f"- **Memory Usage**: {metrics.get('peak_memory', 0):.2f} MB\n")
                    f.write(f"- **GPU Utilization**: {metrics.get('gpu_utilization', 0):.1f}%\n\n")

                f.write("## Recommendations\n\n")
                f.write("1. **Performance Optimization**: Consider model pruning for faster inference\n")
                f.write("2. **Resource Management**: Monitor memory usage for large batch processing\n")
                f.write("3. **Analysis Depth**: Expand analysis to include attention mechanisms\n\n")

                f.write("## Next Steps\n\n")
                f.write("- Conduct comparative analysis with alternative models\n")
                f.write("- Implement real-time monitoring for production deployment\n")
                f.write("- Develop custom visualization dashboards for stakeholders\n")

            logger.info(f"Executive summary generated: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return ""

    def create_analysis_dashboard_data(self, analysis_results: Dict[str, Any]) -> str:
        """Create JSON data file for web dashboard consumption."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_data_{timestamp}.json"
            output_file = self.output_dir / filename

            dashboard_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'model_name': analysis_results.get('model_name'),
                    'num_questions': analysis_results.get('num_questions'),
                    'num_layers': len(analysis_results.get('layer_names', []))
                },
                'summary_stats': {
                    'avg_activation': float(np.mean(analysis_results.get('activations', [0]))),
                    'max_activation': float(np.max(analysis_results.get('activations', [0]))),
                    'min_activation': float(np.min(analysis_results.get('activations', [0]))),
                    'activation_std': float(np.std(analysis_results.get('activations', [0])))
                },
                'layer_data': [],
                'visualizations': analysis_results.get('visualizations', [])
            }

            # Add layer-specific data
            if 'layer_stats' in analysis_results:
                for layer, stats in analysis_results['layer_stats'].items():
                    dashboard_data['layer_data'].append({
                        'layer_name': layer,
                        'mean_activation': float(stats.get('mean', 0)),
                        'std_activation': float(stats.get('std', 0)),
                        'max_activation': float(stats.get('max', 0))
                    })

            with open(output_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2)

            logger.info(f"Dashboard data created: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Error creating dashboard data: {e}")
            return ""

# Global instance
advanced_reporter = AdvancedReporter()

def generate_report(analysis_results: Dict[str, Any], template: str = "analysis_summary") -> str:
    """Generate comprehensive report."""
    return advanced_reporter.generate_comprehensive_report(analysis_results, template)

def export_data(analysis_results: Dict[str, Any], prefix: str = "analysis") -> List[str]:
    """Export analysis data."""
    return advanced_reporter.export_data_to_csv(analysis_results, prefix)

def create_executive_summary(analysis_results: Dict[str, Any]) -> str:
    """Create executive summary."""
    return advanced_reporter.generate_executive_summary(analysis_results)
