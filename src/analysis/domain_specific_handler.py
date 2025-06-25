"""
Domain-Specific Model Handlers for NeuronMap
Supports CodeBERT, SciBERT, BioBERT with specialized domain analysis.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    RobertaModel,
    RobertaTokenizer
)
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
import re
from collections import defaultdict

from ..utils.config import get_config_manager
from .base_model_handler import BaseModelHandler, ModelConfig, ActivationResult

logger = logging.getLogger(__name__)


@dataclass
class DomainActivationResult(ActivationResult):
    """Extended activation result for domain-specific models."""
    domain_specific_features: Dict[str, Any]
    specialized_tokens: Dict[str, List[str]]
    domain_patterns: Dict[str, Any]
    cross_domain_analysis: Optional[Dict[str, Any]] = None
    evaluation_metrics: Optional[Dict[str, float]] = None


class DomainSpecificBERTHandler(BaseModelHandler):
    """
    Specialized handler for domain-specific BERT variants.

    Supports:
    - CodeBERT (programming language understanding)
    - SciBERT (scientific text processing)
    - BioBERT (biomedical text comprehension)

    Features:
    - Domain-specific tokenization and analysis
    - Specialized evaluation metrics
    - Cross-domain transfer analysis
    - Domain-aware attention pattern extraction
    """

    DOMAIN_MODELS = {
        'codebert-base': {
            'domain': 'programming',
            'architecture': 'roberta',
            'special_tokens': ['<s>', '</s>', '<unk>', '<pad>', '<mask>'],
            'code_specific_features': True,
            'bimodal_training': True,  # Code + Natural Language
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'max_length': 512,
            'vocab_size': 50265,
            'model_name_hf': 'microsoft/codebert-base'
        },
        'codebert-base-mlm': {
            'domain': 'programming',
            'architecture': 'roberta',
            'special_tokens': ['<s>', '</s>', '<unk>', '<pad>', '<mask>'],
            'code_specific_features': True,
            'bimodal_training': True,
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'max_length': 512,
            'vocab_size': 50265,
            'model_name_hf': 'microsoft/codebert-base-mlm'
        },
        'scibert-scivocab-uncased': {
            'domain': 'scientific',
            'architecture': 'bert',
            'vocabulary_domain': 'scientific_papers',
            'citation_handling': True,
            'formula_tokenization': True,
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'max_length': 512,
            'vocab_size': 31090,
            'model_name_hf': 'allenai/scibert_scivocab_uncased'
        },
        'scibert-scivocab-cased': {
            'domain': 'scientific',
            'architecture': 'bert',
            'vocabulary_domain': 'scientific_papers',
            'citation_handling': True,
            'formula_tokenization': True,
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'max_length': 512,
            'vocab_size': 31090,
            'model_name_hf': 'allenai/scibert_scivocab_cased'
        },
        'biobert-base-cased-v1.1': {
            'domain': 'biomedical',
            'architecture': 'bert',
            'vocabulary_domain': 'pubmed_pmc',
            'entity_recognition_optimized': True,
            'drug_protein_relationships': True,
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'max_length': 512,
            'vocab_size': 28996,
            'model_name_hf': 'dmis-lab/biobert-base-cased-v1.1'
        },
        'biobert-large-cased-v1.1': {
            'domain': 'biomedical',
            'architecture': 'bert',
            'vocabulary_domain': 'pubmed_pmc',
            'entity_recognition_optimized': True,
            'drug_protein_relationships': True,
            'hidden_size': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'max_length': 512,
            'vocab_size': 28996,
            'model_name_hf': 'dmis-lab/biobert-large-cased-v1.1'
        }
    }

    # Domain-specific patterns for analysis
    DOMAIN_PATTERNS = {
        'programming': {
            'keywords': ['function', 'class', 'return', 'import', 'def', 'if', 'else', 'for', 'while'],
            'operators': ['=', '+', '-', '*', '/', '==', '!=', '&&', '||'],
            'syntax_patterns': [
                r'\bdef\s+\w+\s*\(',  # Python functions
                r'\bclass\s+\w+',     # Class definitions
                r'\bimport\s+\w+',    # Import statements
                r'\w+\.\w+\(',        # Method calls
            ]
        },
        'scientific': {
            'keywords': ['method', 'result', 'conclusion', 'experiment', 'hypothesis', 'analysis'],
            'citation_patterns': [
                r'\([A-Za-z\s]+,?\s*\d{4}\)',  # (Author, Year)
                r'\[\d+\]',                     # [1]
                r'et\s+al\.',                   # et al.
            ],
            'terminology': ['significant', 'correlation', 'p-value', 'dataset', 'methodology']
        },
        'biomedical': {
            'entities': ['protein', 'gene', 'drug', 'disease', 'symptom', 'treatment'],
            'drug_patterns': [
                r'\b[A-Z]{2,}[-]?\d*\b',       # Drug codes
                r'\w+mab\b',                    # Monoclonal antibodies
                r'\w+ine\b',                    # Common drug suffix
            ],
            'protein_patterns': [
                r'\b[A-Z]+\d+\b',              # Protein codes
                r'\bp\d+\b',                    # p-proteins
            ]
        }
    }

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize domain-specific BERT handler."""
        super().__init__(model_name, config)
        self.domain_analyzer = DomainAnalyzer()
        self.cross_domain_analyzer = CrossDomainAnalyzer()

    def _get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for domain-specific model variant."""
        # Normalize model name
        normalized_name = self._normalize_model_name(model_name)

        if normalized_name not in self.DOMAIN_MODELS:
            logger.warning(f"Unknown domain-specific model: {model_name}, using codebert-base config")
            normalized_name = 'codebert-base'

        variant_config = self.DOMAIN_MODELS[normalized_name]

        return ModelConfig(
            model_name=model_name,
            d_model=variant_config['hidden_size'],
            num_layers=variant_config['num_layers'],
            num_heads=variant_config['num_heads'],
            max_length=variant_config['max_length'],
            vocab_size=variant_config.get('vocab_size'),
            architecture_type=f"domain-specific-{variant_config['domain']}",
            special_features={
                'domain': variant_config['domain'],
                'architecture': variant_config['architecture'],
                'special_tokens': variant_config.get('special_tokens', []),
                'code_specific_features': variant_config.get('code_specific_features', False),
                'bimodal_training': variant_config.get('bimodal_training', False),
                'citation_handling': variant_config.get('citation_handling', False),
                'formula_tokenization': variant_config.get('formula_tokenization', False),
                'entity_recognition_optimized': variant_config.get('entity_recognition_optimized', False),
                'drug_protein_relationships': variant_config.get('drug_protein_relationships', False),
                'vocabulary_domain': variant_config.get('vocabulary_domain'),
                'model_name_hf': variant_config.get('model_name_hf')
            }
        )

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name for config lookup."""
        # Handle HuggingFace model names
        model_name = model_name.lower()

        # Map common variations to standard names
        name_mappings = {
            'microsoft/codebert-base': 'codebert-base',
            'microsoft/codebert-base-mlm': 'codebert-base-mlm',
            'allenai/scibert_scivocab_uncased': 'scibert-scivocab-uncased',
            'allenai/scibert_scivocab_cased': 'scibert-scivocab-cased',
            'dmis-lab/biobert-base-cased-v1.1': 'biobert-base-cased-v1.1',
            'dmis-lab/biobert-large-cased-v1.1': 'biobert-large-cased-v1.1',
            'codebert': 'codebert-base',
            'scibert': 'scibert-scivocab-uncased',
            'biobert': 'biobert-base-cased-v1.1'
        }

        return name_mappings.get(model_name, model_name)

    def load_model(self,
                   trust_remote_code: bool = False,
                   torch_dtype: torch.dtype = torch.float32,
                   **kwargs) -> bool:
        """
        Load domain-specific model and tokenizer.

        Args:
            trust_remote_code: Whether to trust remote code
            torch_dtype: Data type for model weights
            **kwargs: Additional arguments for model loading

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            model_name_hf = self.model_config.special_features.get('model_name_hf', self.model_name)
            logger.info(f"Loading domain-specific model: {model_name_hf}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_hf,
                trust_remote_code=trust_remote_code
            )

            # Load model
            model_kwargs = {
                'torch_dtype': torch_dtype,
                'trust_remote_code': trust_remote_code,
                **kwargs
            }

            self.model = AutoModel.from_pretrained(
                model_name_hf,
                **model_kwargs
            )

            # Move to device
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True

            logger.info(f"Successfully loaded domain-specific model: {model_name_hf}")
            logger.info(f"Domain: {self.model_config.special_features['domain']}")

            return True

        except Exception as e:
            logger.error(f"Failed to load domain-specific model {self.model_name}: {str(e)}")
            return False

    def extract_activations(
        self,
        input_text: str,
        target_text: Optional[str] = None,
        layer_indices: Optional[List[int]] = None,
        return_attention: bool = True,
        analyze_domain_patterns: bool = True
    ) -> DomainActivationResult:
        """
        Extract activations with domain-specific analysis.

        Args:
            input_text: Input text for analysis
            target_text: Optional target text (not used for BERT)
            layer_indices: Specific layers to analyze
            return_attention: Whether to return attention weights
            analyze_domain_patterns: Whether to analyze domain-specific patterns

        Returns:
            DomainActivationResult with comprehensive domain analysis
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded first")

        self._validate_input(input_text)

        domain = self.model_config.special_features['domain']

        # Prepare inputs
        inputs = self._prepare_inputs(input_text)

        # Storage for activations
        layer_activations = {}
        attention_weights = {}
        hidden_states = []

        # Register hooks for activation extraction
        hooks = self._register_domain_hooks(
            layer_activations,
            hidden_states,
            attention_weights,
            layer_indices,
            return_attention
        )

        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    output_attentions=return_attention,
                    output_hidden_states=True,
                    return_dict=True
                )

        finally:
            # Clean up hooks
            for hook in hooks.values():
                hook.remove()

        # Analyze domain-specific patterns
        domain_specific_features = {}
        specialized_tokens = {}
        domain_patterns = {}

        if analyze_domain_patterns:
            domain_analysis = self.domain_analyzer.analyze_domain_patterns(
                input_text,
                domain,
                attention_weights,
                self.tokenizer
            )

            domain_specific_features = domain_analysis['features']
            specialized_tokens = domain_analysis['tokens']
            domain_patterns = domain_analysis['patterns']

        # Calculate evaluation metrics
        evaluation_metrics = self._calculate_domain_metrics(
            input_text,
            domain,
            attention_weights,
            hidden_states
        )

        # Create result
        result = DomainActivationResult(
            layer_activations=layer_activations,
            attention_weights=attention_weights,
            hidden_states={'layers': hidden_states},
            metadata={
                'model_name': self.model_name,
                'domain': domain,
                'num_layers': len(hidden_states),
                'input_length': inputs['input_ids'].size(1),
                'architecture': self.model_config.special_features['architecture'],
                'vocabulary_domain': self.model_config.special_features.get('vocabulary_domain'),
                'domain_features_detected': len(domain_specific_features)
            },
            input_ids=inputs['input_ids'],
            input_text=input_text,
            domain_specific_features=domain_specific_features,
            specialized_tokens=specialized_tokens,
            domain_patterns=domain_patterns,
            evaluation_metrics=evaluation_metrics
        )

        return result

    def _register_domain_hooks(
        self,
        activations_dict: Dict[str, torch.Tensor],
        hidden_states_list: List[torch.Tensor],
        attention_weights: Dict[str, torch.Tensor],
        layer_indices: Optional[List[int]] = None,
        return_attention: bool = True
    ) -> Dict[str, Any]:
        """Register hooks for domain-specific analysis."""
        hooks = {}

        def layer_hook(name, layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_state = output[0]
                    if return_attention and len(output) > 1:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            attention_weights[f"layer_{layer_idx}_attention"] = attn_weights.detach().cpu()
                else:
                    hidden_state = output

                activations_dict[f"layer_{layer_idx}"] = hidden_state.detach().cpu()

                # Store in sequential list
                if len(hidden_states_list) <= layer_idx:
                    hidden_states_list.extend([None] * (layer_idx - len(hidden_states_list) + 1))
                hidden_states_list[layer_idx] = hidden_state.detach().cpu()

            return hook

        # Register on encoder layers
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            layers = self.model.encoder.layer
        elif hasattr(self.model, 'roberta') and hasattr(self.model.roberta.encoder, 'layer'):
            layers = self.model.roberta.encoder.layer
        elif hasattr(self.model, 'bert') and hasattr(self.model.bert.encoder, 'layer'):
            layers = self.model.bert.encoder.layer
        else:
            logger.warning("Could not find model layers for hook registration")
            return hooks

        target_indices = layer_indices if layer_indices else range(len(layers))

        for i in target_indices:
            if i < len(layers):
                hook_handle = layers[i].register_forward_hook(layer_hook(f"layer_{i}", i))
                hooks[f"layer_{i}"] = hook_handle

        return hooks

    def _calculate_domain_metrics(
        self,
        input_text: str,
        domain: str,
        attention_weights: Dict[str, torch.Tensor],
        hidden_states: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Calculate domain-specific evaluation metrics."""
        metrics = {}

        if domain == 'programming':
            metrics.update(self._calculate_code_metrics(input_text, attention_weights))
        elif domain == 'scientific':
            metrics.update(self._calculate_scientific_metrics(input_text, attention_weights))
        elif domain == 'biomedical':
            metrics.update(self._calculate_biomedical_metrics(input_text, attention_weights))

        return metrics

    def _calculate_code_metrics(
        self,
        input_text: str,
        attention_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Calculate code understanding metrics."""
        metrics = {}

        # Syntax pattern detection
        syntax_patterns = self.DOMAIN_PATTERNS['programming']['syntax_patterns']
        syntax_matches = sum(1 for pattern in syntax_patterns if re.search(pattern, input_text))
        metrics['syntax_pattern_coverage'] = syntax_matches / max(len(syntax_patterns), 1)

        # Keyword frequency
        keywords = self.DOMAIN_PATTERNS['programming']['keywords']
        keyword_count = sum(1 for word in input_text.lower().split() if word in keywords)
        total_words = len(input_text.split())
        metrics['keyword_density'] = keyword_count / max(total_words, 1)

        # Code structure complexity (simple heuristic)
        metrics['code_complexity'] = min(1.0, (input_text.count('(') + input_text.count('{')) / 10)

        return metrics

    def _calculate_scientific_metrics(
        self,
        input_text: str,
        attention_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Calculate scientific text understanding metrics."""
        metrics = {}

        # Citation pattern detection
        citation_patterns = self.DOMAIN_PATTERNS['scientific']['citation_patterns']
        citation_matches = sum(1 for pattern in citation_patterns if re.search(pattern, input_text))
        metrics['citation_density'] = citation_matches / max(len(input_text.split()), 1) * 100

        # Scientific terminology
        terminology = self.DOMAIN_PATTERNS['scientific']['terminology']
        term_count = sum(1 for term in terminology if term.lower() in input_text.lower())
        metrics['scientific_terminology_coverage'] = term_count / max(len(terminology), 1)

        # Methodology indicators
        method_indicators = ['method', 'approach', 'technique', 'procedure']
        method_count = sum(1 for indicator in method_indicators if indicator in input_text.lower())
        metrics['methodology_indicators'] = method_count / max(len(method_indicators), 1)

        return metrics

    def _calculate_biomedical_metrics(
        self,
        input_text: str,
        attention_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Calculate biomedical text understanding metrics."""
        metrics = {}

        # Drug pattern detection
        drug_patterns = self.DOMAIN_PATTERNS['biomedical']['drug_patterns']
        drug_matches = sum(1 for pattern in drug_patterns if re.search(pattern, input_text))
        metrics['drug_pattern_detection'] = drug_matches / max(len(input_text.split()), 1) * 100

        # Protein pattern detection
        protein_patterns = self.DOMAIN_PATTERNS['biomedical']['protein_patterns']
        protein_matches = sum(1 for pattern in protein_patterns if re.search(pattern, input_text))
        metrics['protein_pattern_detection'] = protein_matches / max(len(input_text.split()), 1) * 100

        # Biomedical entity coverage
        entities = self.DOMAIN_PATTERNS['biomedical']['entities']
        entity_count = sum(1 for entity in entities if entity.lower() in input_text.lower())
        metrics['biomedical_entity_coverage'] = entity_count / max(len(entities), 1)

        return metrics

    def analyze_cross_domain_transfer(
        self,
        input_texts: Dict[str, str],
        compare_domains: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze cross-domain transfer learning capabilities.

        Args:
            input_texts: Dict mapping domain names to input texts
            compare_domains: List of domains to compare

        Returns:
            Cross-domain analysis results
        """
        results = {}
        domain_activations = {}

        # Extract activations for each domain
        for domain, text in input_texts.items():
            if domain in compare_domains:
                activation_result = self.extract_activations(text, analyze_domain_patterns=True)
                domain_activations[domain] = activation_result

        # Perform cross-domain comparison
        results = self.cross_domain_analyzer.compare_domains(
            domain_activations,
            self.model_config.special_features['domain']
        )

        return results


class DomainAnalyzer:
    """Analyzer for domain-specific patterns and features."""

    def analyze_domain_patterns(
        self,
        input_text: str,
        domain: str,
        attention_weights: Dict[str, torch.Tensor],
        tokenizer
    ) -> Dict[str, Any]:
        """Analyze domain-specific patterns in the input text."""
        analysis = {
            'features': {},
            'tokens': {},
            'patterns': {}
        }

        if domain == 'programming':
            analysis.update(self._analyze_programming_patterns(input_text, attention_weights, tokenizer))
        elif domain == 'scientific':
            analysis.update(self._analyze_scientific_patterns(input_text, attention_weights, tokenizer))
        elif domain == 'biomedical':
            analysis.update(self._analyze_biomedical_patterns(input_text, attention_weights, tokenizer))

        return analysis

    def _analyze_programming_patterns(
        self,
        input_text: str,
        attention_weights: Dict[str, torch.Tensor],
        tokenizer
    ) -> Dict[str, Any]:
        """Analyze programming-specific patterns."""
        patterns = DomainSpecificBERTHandler.DOMAIN_PATTERNS['programming']

        features = {}
        tokens = {}
        pattern_analysis = {}

        # Tokenize and find special tokens
        token_ids = tokenizer.encode(input_text)
        token_texts = tokenizer.convert_ids_to_tokens(token_ids)

        # Find programming keywords
        programming_tokens = []
        for i, token in enumerate(token_texts):
            clean_token = token.replace('Ġ', '').lower()  # Remove RoBERTa prefix
            if clean_token in patterns['keywords']:
                programming_tokens.append((i, token, clean_token))

        tokens['programming_keywords'] = programming_tokens

        # Analyze syntax patterns
        syntax_matches = {}
        for pattern_name, pattern in enumerate(patterns['syntax_patterns']):
            matches = list(re.finditer(pattern, input_text))
            syntax_matches[f'pattern_{pattern_name}'] = [
                {'start': m.start(), 'end': m.end(), 'text': m.group()}
                for m in matches
            ]

        pattern_analysis['syntax_patterns'] = syntax_matches

        # Calculate code complexity features
        features['function_count'] = len(re.findall(r'\bdef\s+\w+', input_text))
        features['class_count'] = len(re.findall(r'\bclass\s+\w+', input_text))
        features['import_count'] = len(re.findall(r'\bimport\s+\w+', input_text))
        features['indentation_levels'] = len(set(len(line) - len(line.lstrip()) for line in input_text.split('\n')))

        return {
            'features': features,
            'tokens': tokens,
            'patterns': pattern_analysis
        }

    def _analyze_scientific_patterns(
        self,
        input_text: str,
        attention_weights: Dict[str, torch.Tensor],
        tokenizer
    ) -> Dict[str, Any]:
        """Analyze scientific text patterns."""
        patterns = DomainSpecificBERTHandler.DOMAIN_PATTERNS['scientific']

        features = {}
        tokens = {}
        pattern_analysis = {}

        # Find citations
        citations = []
        for pattern in patterns['citation_patterns']:
            matches = list(re.finditer(pattern, input_text))
            citations.extend([
                {'start': m.start(), 'end': m.end(), 'text': m.group(), 'type': pattern}
                for m in matches
            ])

        pattern_analysis['citations'] = citations

        # Scientific terminology analysis
        terminology_found = []
        for term in patterns['terminology']:
            if term.lower() in input_text.lower():
                terminology_found.append(term)

        tokens['scientific_terms'] = terminology_found

        # Calculate scientific features
        features['citation_count'] = len(citations)
        features['terminology_density'] = len(terminology_found) / max(len(input_text.split()), 1)
        features['sentence_complexity'] = np.mean([len(sent.split()) for sent in input_text.split('.')])

        return {
            'features': features,
            'tokens': tokens,
            'patterns': pattern_analysis
        }

    def _analyze_biomedical_patterns(
        self,
        input_text: str,
        attention_weights: Dict[str, torch.Tensor],
        tokenizer
    ) -> Dict[str, Any]:
        """Analyze biomedical text patterns."""
        patterns = DomainSpecificBERTHandler.DOMAIN_PATTERNS['biomedical']

        features = {}
        tokens = {}
        pattern_analysis = {}

        # Find drug patterns
        drug_entities = []
        for pattern in patterns['drug_patterns']:
            matches = list(re.finditer(pattern, input_text))
            drug_entities.extend([
                {'start': m.start(), 'end': m.end(), 'text': m.group(), 'type': 'drug'}
                for m in matches
            ])

        # Find protein patterns
        protein_entities = []
        for pattern in patterns['protein_patterns']:
            matches = list(re.finditer(pattern, input_text))
            protein_entities.extend([
                {'start': m.start(), 'end': m.end(), 'text': m.group(), 'type': 'protein'}
                for m in matches
            ])

        pattern_analysis['drug_entities'] = drug_entities
        pattern_analysis['protein_entities'] = protein_entities

        # Biomedical entities
        entities_found = []
        for entity in patterns['entities']:
            if entity.lower() in input_text.lower():
                entities_found.append(entity)

        tokens['biomedical_entities'] = entities_found

        # Calculate biomedical features
        features['drug_entity_count'] = len(drug_entities)
        features['protein_entity_count'] = len(protein_entities)
        features['entity_density'] = len(entities_found) / max(len(input_text.split()), 1)
        features['technical_term_ratio'] = len([word for word in input_text.split() if len(word) > 8]) / max(len(input_text.split()), 1)

        return {
            'features': features,
            'tokens': tokens,
            'patterns': pattern_analysis
        }


class CrossDomainAnalyzer:
    """Analyzer for cross-domain transfer and comparison."""

    def compare_domains(
        self,
        domain_activations: Dict[str, DomainActivationResult],
        primary_domain: str
    ) -> Dict[str, Any]:
        """Compare activations across different domains."""
        comparison = {
            'primary_domain': primary_domain,
            'domain_similarities': {},
            'transfer_analysis': {},
            'performance_degradation': {}
        }

        # Calculate domain similarities
        for domain1, activation1 in domain_activations.items():
            for domain2, activation2 in domain_activations.items():
                if domain1 != domain2:
                    similarity = self._calculate_activation_similarity(
                        activation1.hidden_states['layers'],
                        activation2.hidden_states['layers']
                    )
                    comparison['domain_similarities'][f'{domain1}_vs_{domain2}'] = similarity

        # Analyze transfer capabilities
        for domain, activation in domain_activations.items():
            if domain != primary_domain:
                transfer_metrics = self._analyze_domain_transfer(
                    activation,
                    primary_domain,
                    domain
                )
                comparison['transfer_analysis'][domain] = transfer_metrics

        return comparison

    def _calculate_activation_similarity(
        self,
        activations1: List[torch.Tensor],
        activations2: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Calculate similarity between activation patterns."""
        similarities = {}

        min_layers = min(len(activations1), len(activations2))

        layer_similarities = []
        for i in range(min_layers):
            if activations1[i] is not None and activations2[i] is not None:
                # Flatten and calculate cosine similarity
                act1_flat = activations1[i].flatten()
                act2_flat = activations2[i].flatten()

                # Ensure same length
                min_len = min(len(act1_flat), len(act2_flat))
                if min_len > 0:
                    similarity = torch.cosine_similarity(
                        act1_flat[:min_len].unsqueeze(0),
                        act2_flat[:min_len].unsqueeze(0)
                    )
                    layer_similarities.append(float(similarity))

        if layer_similarities:
            similarities['mean_similarity'] = np.mean(layer_similarities)
            similarities['std_similarity'] = np.std(layer_similarities)
            similarities['max_similarity'] = np.max(layer_similarities)
            similarities['min_similarity'] = np.min(layer_similarities)

        return similarities

    def _analyze_domain_transfer(
        self,
        activation: DomainActivationResult,
        source_domain: str,
        target_domain: str
    ) -> Dict[str, Any]:
        """Analyze domain transfer capabilities."""
        transfer_metrics = {}

        # Analyze evaluation metrics degradation
        if activation.evaluation_metrics:
            transfer_metrics['evaluation_metrics'] = activation.evaluation_metrics

            # Calculate normalized performance
            metric_values = list(activation.evaluation_metrics.values())
            if metric_values:
                transfer_metrics['avg_performance'] = np.mean(metric_values)
                transfer_metrics['performance_variance'] = np.var(metric_values)

        # Analyze domain-specific feature detection
        if activation.domain_specific_features:
            feature_count = len(activation.domain_specific_features)
            transfer_metrics['feature_adaptation'] = feature_count / 10  # Normalized score

        # Domain pattern recognition
        if activation.domain_patterns:
            pattern_complexity = sum(
                len(patterns) if isinstance(patterns, list)
                else len(patterns) if isinstance(patterns, dict)
                else 1
                for patterns in activation.domain_patterns.values()
            )
            transfer_metrics['pattern_recognition'] = min(1.0, pattern_complexity / 20)

        return transfer_metrics


# Register domain-specific handler with the factory
from .base_model_handler import ModelFactory
ModelFactory.register_handler('codebert', DomainSpecificBERTHandler)
ModelFactory.register_handler('scibert', DomainSpecificBERTHandler)
ModelFactory.register_handler('biobert', DomainSpecificBERTHandler)
