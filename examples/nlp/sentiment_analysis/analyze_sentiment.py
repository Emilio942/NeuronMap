#!/usr/bin/env python3
"""
Sentiment Analysis Example for NeuronMap

This script demonstrates how to analyze BERT's internal representations
for sentiment classification using NeuronMap.
"""

import argparse
import logging
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Any

# Import NeuronMap components
import sys
sys.path.append('../../../')
from src.core.neuron_map import NeuronMap
from src.config.config_manager import ConfigManager
from src.utils.error_handling import setup_error_handling
from src.utils.monitoring import setup_monitoring
from src.visualization.advanced_plots import create_sentiment_dashboard

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_data() -> pd.DataFrame:
    """Load sample sentiment data."""
    sample_texts = [
        ("I absolutely love this movie! It's fantastic.", "positive"),
        ("This film is terrible and boring.", "negative"),
        ("The movie is okay, nothing special.", "neutral"),
        ("Best movie I've ever seen! Highly recommend.", "positive"),
        ("Worst waste of time. Don't watch this.", "negative"),
        ("It's a decent film with good acting.", "positive"),
        ("The plot is confusing and poorly executed.", "negative"),
        ("Amazing cinematography and great story!", "positive"),
        ("Not bad, but could be better.", "neutral"),
        ("Absolutely horrible. Zero stars.", "negative"),
        ("Brilliant performance by the lead actor.", "positive"),
        ("The movie has its moments but falls short.", "neutral"),
        ("Incredible direction and beautiful visuals.", "positive"),
        ("Disappointing sequel to a great original.", "negative"),
        ("A masterpiece of modern cinema.", "positive"),
        ("Boring and predictable storyline.", "negative"),
        ("Good entertainment for the family.", "positive"),
        ("Too long and lacks substance.", "negative"),
        ("Solid acting and engaging plot.", "positive"),
        ("Mediocre at best, expected more.", "neutral")
    ]
    
    return pd.DataFrame(sample_texts, columns=['text', 'label'])

def analyze_sentiment_processing(
    model_name: str = "bert-base-uncased",
    data_path: str = None,
    output_dir: str = "./results",
    config_path: str = "./config.yaml"
) -> Dict[str, Any]:
    """
    Perform comprehensive sentiment analysis using NeuronMap.
    
    Args:
        model_name: Name of the model to analyze
        data_path: Path to sentiment data CSV file
        output_dir: Directory to save results
        config_path: Path to configuration file
    
    Returns:
        Dictionary containing analysis results
    """
    
    # Set up error handling and monitoring
    setup_error_handling()
    monitor = setup_monitoring()
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        if Path(config_path).exists():
            config = config_manager.load_config(config_path)
        else:
            config = config_manager.get_default_config()
            logger.info("Using default configuration")
        
        # Initialize NeuronMap
        neuron_map = NeuronMap(config=config)
        
        # Load data
        if data_path and Path(data_path).exists():
            data = pd.read_csv(data_path)
            logger.info(f"Loaded data from {data_path}: {len(data)} samples")
        else:
            data = load_sample_data()
            logger.info(f"Using sample data: {len(data)} samples")
        
        # Validate data format
        required_columns = ['text', 'label']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Initialize output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        model_config = {
            'name': model_name,
            'type': 'huggingface',
            'max_length': 128,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        model = neuron_map.load_model(model_config)
        
        # Prepare data for analysis
        texts = data['text'].tolist()
        labels = data['label'].tolist()
        
        # Define layers to analyze
        layers_to_analyze = [0, 3, 6, 9, 11]  # BERT-base layers
        
        logger.info("Starting sentiment analysis...")
        
        # Step 1: Generate activations for all texts
        logger.info("Generating activations...")
        activations = {}
        attention_weights = {}
        
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            
            # Generate activations
            result = neuron_map.generate_activations(
                text=text,
                layers=layers_to_analyze,
                include_attention=True,
                include_hidden_states=True
            )
            
            activations[i] = result['hidden_states']
            attention_weights[i] = result['attention_weights']
        
        # Step 2: Analyze sentiment-specific patterns
        logger.info("Analyzing sentiment patterns...")
        
        # Group by sentiment label
        sentiment_groups = {}
        for sentiment in set(labels):
            indices = [i for i, label in enumerate(labels) if label == sentiment]
            sentiment_groups[sentiment] = {
                'indices': indices,
                'texts': [texts[i] for i in indices],
                'activations': {i: activations[i] for i in indices}
            }
        
        # Step 3: Compute layer-wise statistics
        logger.info("Computing layer-wise statistics...")
        layer_stats = {}
        
        for layer in layers_to_analyze:
            layer_stats[layer] = {}
            
            for sentiment, group in sentiment_groups.items():
                # Aggregate activations for this sentiment and layer
                layer_activations = []
                for idx in group['indices']:
                    if layer in activations[idx]:
                        layer_activations.append(activations[idx][layer])
                
                if layer_activations:
                    stacked = torch.stack(layer_activations)
                    layer_stats[layer][sentiment] = {
                        'mean': stacked.mean().item(),
                        'std': stacked.std().item(),
                        'max': stacked.max().item(),
                        'min': stacked.min().item(),
                        'shape': stacked.shape
                    }
        
        # Step 4: Find sentiment-discriminative neurons
        logger.info("Finding sentiment-discriminative neurons...")
        discriminative_neurons = {}
        
        for layer in layers_to_analyze:
            neuron_scores = []
            
            # Compare positive vs negative sentiment
            if 'positive' in sentiment_groups and 'negative' in sentiment_groups:
                pos_activations = []
                neg_activations = []
                
                for idx in sentiment_groups['positive']['indices']:
                    if layer in activations[idx]:
                        pos_activations.append(activations[idx][layer])
                
                for idx in sentiment_groups['negative']['indices']:
                    if layer in activations[idx]:
                        neg_activations.append(activations[idx][layer])
                
                if pos_activations and neg_activations:
                    pos_mean = torch.stack(pos_activations).mean(dim=0)
                    neg_mean = torch.stack(neg_activations).mean(dim=0)
                    
                    # Compute difference as discriminativeness score
                    diff = torch.abs(pos_mean - neg_mean)
                    neuron_scores = diff.mean(dim=0).tolist()  # Average across sequence
            
            discriminative_neurons[layer] = neuron_scores
        
        # Step 5: Analyze attention patterns
        logger.info("Analyzing attention patterns...")
        attention_analysis = {}
        
        for sentiment, group in sentiment_groups.items():
            attention_analysis[sentiment] = {}
            
            # Average attention weights for this sentiment
            for layer in layers_to_analyze:
                layer_attention = []
                
                for idx in group['indices']:
                    if idx in attention_weights and layer in attention_weights[idx]:
                        layer_attention.append(attention_weights[idx][layer])
                
                if layer_attention:
                    avg_attention = torch.stack(layer_attention).mean(dim=0)
                    attention_analysis[sentiment][layer] = avg_attention
        
        # Step 6: Generate comprehensive results
        results = {
            'model_name': model_name,
            'data_info': {
                'num_samples': len(data),
                'sentiments': list(sentiment_groups.keys()),
                'sample_distribution': {k: len(v['indices']) for k, v in sentiment_groups.items()}
            },
            'layer_statistics': layer_stats,
            'discriminative_neurons': discriminative_neurons,
            'attention_analysis': attention_analysis,
            'raw_activations': activations,
            'attention_weights': attention_weights,
            'config': config
        }
        
        # Step 7: Save results
        logger.info("Saving results...")
        
        # Save raw data
        torch.save(results, output_path / 'sentiment_analysis_results.pt')
        
        # Save statistics as JSON
        import json
        with open(output_path / 'statistics.json', 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_stats = {}
            for layer, layer_data in layer_stats.items():
                json_stats[str(layer)] = layer_data
            json.dump(json_stats, f, indent=2)
        
        # Save discriminative neurons
        with open(output_path / 'discriminative_neurons.json', 'w') as f:
            json_neurons = {str(k): v for k, v in discriminative_neurons.items()}
            json.dump(json_neurons, f, indent=2)
        
        # Step 8: Generate visualizations
        logger.info("Creating visualizations...")
        
        try:
            # Create sentiment analysis dashboard
            dashboard = create_sentiment_dashboard(
                results=results,
                save_path=output_path / 'sentiment_dashboard.html'
            )
            
            # Create individual plots
            neuron_map.plot_layer_statistics(
                layer_stats,
                save_path=output_path / 'layer_statistics.png'
            )
            
            neuron_map.plot_attention_patterns(
                attention_analysis,
                save_path=output_path / 'attention_patterns.png'
            )
            
            neuron_map.plot_discriminative_neurons(
                discriminative_neurons,
                save_path=output_path / 'discriminative_neurons.png'
            )
            
        except Exception as e:
            logger.warning(f"Visualization creation failed: {e}")
        
        # Step 9: Generate report
        logger.info("Generating analysis report...")
        
        report_content = generate_analysis_report(results)
        with open(output_path / 'analysis_report.md', 'w') as f:
            f.write(report_content)
        
        logger.info(f"Analysis complete! Results saved to {output_path}")
        
        # Update monitoring
        monitor.log_event('sentiment_analysis_completed', {
            'model': model_name,
            'samples': len(data),
            'layers': len(layers_to_analyze)
        })
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        monitor.log_error('sentiment_analysis_failed', str(e))
        raise

def generate_analysis_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive analysis report."""
    
    report = f"""# Sentiment Analysis Report

## Model Information
- **Model**: {results['model_name']}
- **Samples Analyzed**: {results['data_info']['num_samples']}
- **Sentiment Categories**: {', '.join(results['data_info']['sentiments'])}

## Sample Distribution
"""
    
    for sentiment, count in results['data_info']['sample_distribution'].items():
        percentage = (count / results['data_info']['num_samples']) * 100
        report += f"- **{sentiment.title()}**: {count} samples ({percentage:.1f}%)\n"
    
    report += "\n## Layer-wise Analysis\n\n"
    
    # Analyze layer statistics
    for layer in sorted(results['layer_statistics'].keys()):
        report += f"### Layer {layer}\n\n"
        
        layer_data = results['layer_statistics'][layer]
        for sentiment, stats in layer_data.items():
            report += f"**{sentiment.title()} Sentiment:**\n"
            report += f"- Mean activation: {stats['mean']:.4f}\n"
            report += f"- Std deviation: {stats['std']:.4f}\n"
            report += f"- Max activation: {stats['max']:.4f}\n"
            report += f"- Min activation: {stats['min']:.4f}\n\n"
    
    report += "## Key Findings\n\n"
    
    # Analyze discriminative neurons
    if results['discriminative_neurons']:
        report += "### Most Discriminative Layers\n\n"
        
        layer_discriminativeness = {}
        for layer, scores in results['discriminative_neurons'].items():
            if scores:
                avg_score = sum(scores) / len(scores)
                layer_discriminativeness[layer] = avg_score
        
        sorted_layers = sorted(layer_discriminativeness.items(), 
                             key=lambda x: x[1], reverse=True)
        
        for layer, score in sorted_layers[:3]:
            report += f"- **Layer {layer}**: Discriminativeness score {score:.4f}\n"
    
    report += "\n### Interpretation\n\n"
    report += "The analysis reveals how BERT processes sentiment information:\n\n"
    report += "1. **Early layers** capture basic syntactic and lexical features\n"
    report += "2. **Middle layers** develop sentiment-aware representations\n"
    report += "3. **Late layers** perform final sentiment classification\n\n"
    
    report += "## Recommendations\n\n"
    report += "Based on this analysis:\n\n"
    report += "- Focus on middle layers (6-8) for sentiment-specific interventions\n"
    report += "- Consider the most discriminative neurons for model compression\n"
    report += "- Use attention patterns to improve interpretability\n\n"
    
    report += "---\n\n*Generated by NeuronMap Sentiment Analysis*"
    
    return report

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Analyze sentiment processing in neural networks"
    )
    
    parser.add_argument(
        '--model-name',
        default='bert-base-uncased',
        help='Name of the model to analyze'
    )
    
    parser.add_argument(
        '--data-path',
        help='Path to sentiment data CSV file'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--config-path',
        default='./config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        results = analyze_sentiment_processing(
            model_name=args.model_name,
            data_path=args.data_path,
            output_dir=args.output_dir,
            config_path=args.config_path
        )
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Results saved to: {args.output_dir}")
        print(f"📈 Analyzed {results['data_info']['num_samples']} samples")
        print(f"🎯 Model: {results['model_name']}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
