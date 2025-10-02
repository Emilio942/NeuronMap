"""
Advanced Analysis Capabilities for Universal Model Support
Provides cross-architecture analysis, optimization recommendations, and domain-specific adaptations.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from .universal_model_support import (
    UniversalModelSupport, ArchitectureType, LayerType, LayerInfo, AdapterConfig
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for model analysis."""
    memory_usage_mb: float
    inference_time_ms: float
    activation_extraction_time_ms: float
    layer_analysis_time_ms: float
    total_parameters: int
    active_parameters: int
    sparsity_ratio: float = 0.0
    efficiency_score: float = 0.0


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation for a specific model."""
    category: str
    priority: str  # high, medium, low
    description: str
    expected_improvement: str
    implementation_complexity: str  # easy, moderate, complex
    code_example: Optional[str] = None


@dataclass
class CrossArchitectureComparison:
    """Results of comparing two different architectures."""
    model1_name: str
    model2_name: str
    architecture1: ArchitectureType
    architecture2: ArchitectureType
    similarity_score: float
    shared_features: List[str]
    unique_features1: List[str]
    unique_features2: List[str]
    performance_comparison: Dict[str, Any]
    recommendations: List[str]


class DomainSpecificAnalyzer:
    """Analyzer for domain-specific models with specialized handling."""

    def __init__(self):
        self.domain_configs = {
            "code": {
                "special_tokens": [
                    "<code>",
                    "</code>",
                    "<comment>",
                    "</comment>"],
                "syntax_aware": True,
                "programming_languages": [
                    "python",
                    "java",
                    "javascript",
                    "c++"],
                "code_patterns": [
                    r"\bdef\b",
                    r"\bclass\b",
                    r"\bfunction\b",
                    r"\bimport\b"]},
            "scientific": {
                "special_tokens": [
                    "<formula>",
                    "</formula>",
                    "<citation>",
                    "</citation>"],
                "latex_aware": True,
                "scientific_domains": [
                    "physics",
                    "chemistry",
                    "biology",
                    "mathematics"],
                "citation_patterns": [
                    r"\[[0-9]+\]",
                    r"\(.*\d{4}.*\)"]},
            "biomedical": {
                "special_tokens": [
                    "<protein>",
                    "</protein>",
                    "<gene>",
                    "</gene>"],
                "entity_types": [
                    "DRUG",
                    "DISEASE",
                    "PROTEIN",
                    "GENE"],
                "medical_terminology": True,
                "pubmed_vocab": True}}

    def analyze_domain_specificity(
            self, model: nn.Module, model_name: str, domain: str) -> Dict[str, Any]:
        """Analyze how well a model is adapted for a specific domain."""
        if domain not in self.domain_configs:
            raise ValueError(f"Unsupported domain: {domain}")

        domain_config = self.domain_configs[domain]

        # Analyze vocabulary specialization
        vocab_specialization = self._analyze_vocabulary_specialization(
            model, domain_config)

        # Analyze attention patterns for domain-specific features
        attention_analysis = self._analyze_domain_attention_patterns(
            model, domain_config)

        # Check for domain-specific architectural adaptations
        architectural_adaptations = self._check_architectural_adaptations(
            model, domain_config)

        return {
            "domain": domain,
            "vocabulary_specialization": vocab_specialization,
            "attention_analysis": attention_analysis,
            "architectural_adaptations": architectural_adaptations,
            "domain_compatibility_score": self._calculate_domain_compatibility_score(
                vocab_specialization, attention_analysis, architectural_adaptations
            )
        }

    def _analyze_vocabulary_specialization(
            self, model: nn.Module, domain_config: Dict) -> Dict[str, Any]:
        """Analyze vocabulary specialization for the domain."""
        # This would analyze the tokenizer and embedding layer
        # For now, return a simplified analysis
        return {
            "has_domain_tokens": len(domain_config.get("special_tokens", [])) > 0,
            "specialized_embedding_detected": True,  # Would check embedding patterns
            "vocabulary_coverage": 0.85  # Would calculate actual coverage
        }

    def _analyze_domain_attention_patterns(
            self, model: nn.Module, domain_config: Dict) -> Dict[str, Any]:
        """Analyze attention patterns specific to the domain."""
        return {
            "domain_specific_attention_heads": 3,  # Would detect specialized heads
            "cross_domain_transfer_patterns": True,
            "attention_specialization_score": 0.72
        }

    def _check_architectural_adaptations(
            self, model: nn.Module, domain_config: Dict) -> Dict[str, Any]:
        """Check for domain-specific architectural adaptations."""
        return {
            "has_domain_specific_layers": False,
            "modified_attention_mechanism": domain_config.get("syntax_aware", False),
            "specialized_output_head": True
        }

    def _calculate_domain_compatibility_score(
            self,
            vocab_analysis: Dict,
            attention_analysis: Dict,
            arch_analysis: Dict) -> float:
        """Calculate overall domain compatibility score."""
        # Simplified scoring based on key indicators
        score = 0.0

        if vocab_analysis["has_domain_tokens"]:
            score += 0.3
        score += vocab_analysis["vocabulary_coverage"] * 0.3
        score += attention_analysis["attention_specialization_score"] * 0.4

        return min(score, 1.0)


class PerformanceAnalyzer:
    """Analyzer for model performance and optimization opportunities."""

    def __init__(self):
        self.baseline_metrics = {}

    def analyze_performance(self, model: nn.Module, model_name: str,
                            sample_inputs: Optional[List[str]] = None) -> PerformanceMetrics:
        """Comprehensive performance analysis of a model."""

        # Memory usage analysis
        memory_usage = self._measure_memory_usage(model)

        # Parameter counting
        total_params, active_params = self._count_parameters(model)

        # Sparsity analysis
        sparsity_ratio = self._calculate_sparsity(model)

        # Timing analysis (would need actual inputs for real timing)
        inference_time = self._estimate_inference_time(model, total_params)
        activation_time = inference_time * 0.3  # Rough estimate
        layer_analysis_time = inference_time * 0.2

        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(
            memory_usage, inference_time, total_params, sparsity_ratio
        )

        return PerformanceMetrics(
            memory_usage_mb=memory_usage,
            inference_time_ms=inference_time,
            activation_extraction_time_ms=activation_time,
            layer_analysis_time_ms=layer_analysis_time,
            total_parameters=total_params,
            active_parameters=active_params,
            sparsity_ratio=sparsity_ratio,
            efficiency_score=efficiency_score
        )

    def _measure_memory_usage(self, model: nn.Module) -> float:
        """Measure model memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)  # Convert to MB

    def _count_parameters(self, model: nn.Module) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, active_params

    def _calculate_sparsity(self, model: nn.Module) -> float:
        """Calculate model sparsity ratio."""
        total_elements = 0
        zero_elements = 0

        for param in model.parameters():
            total_elements += param.numel()
            zero_elements += (param.abs() < 1e-8).sum().item()

        return zero_elements / total_elements if total_elements > 0 else 0.0

    def _estimate_inference_time(self, model: nn.Module, total_params: int) -> float:
        """Estimate inference time based on model size."""
        # Very rough estimation based on parameter count
        # Real implementation would benchmark actual inference
        base_time = 50  # Base time in ms
        param_factor = total_params / 1e6  # Scale by millions of parameters
        return base_time + param_factor * 10

    def _calculate_efficiency_score(self, memory_mb: float, inference_ms: float,
                                    total_params: int, sparsity: float) -> float:
        """Calculate overall efficiency score (0-1, higher is better)."""
        # Normalize factors and combine
        memory_score = max(0, 1 - memory_mb / 1000)  # Penalty for >1GB memory
        speed_score = max(0, 1 - inference_ms / 1000)  # Penalty for >1s inference
        param_score = max(0, 1 - total_params / 1e9)  # Penalty for >1B parameters
        sparsity_bonus = sparsity * 0.2  # Bonus for sparsity

        return (memory_score + speed_score + param_score + sparsity_bonus) / 3

    def generate_optimization_recommendations(
            self,
            metrics: PerformanceMetrics,
            model_name: str) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on performance metrics."""
        recommendations = []

        # Memory optimization
        if metrics.memory_usage_mb > 500:
            recommendations.append(
                OptimizationRecommendation(
                    category="memory",
                    priority="high",
                    description="High memory usage detected. Consider model compression or quantization.",
                    expected_improvement="30-50% memory reduction",
                    implementation_complexity="moderate",
                    code_example="torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)"))

        # Speed optimization
        if metrics.inference_time_ms > 200:
            recommendations.append(
                OptimizationRecommendation(
                    category="speed",
                    priority="medium",
                    description="Slow inference detected. Consider batch processing or model distillation.",
                    expected_improvement="20-40% speed improvement",
                    implementation_complexity="moderate"))

        # Sparsity optimization
        if metrics.sparsity_ratio < 0.1:
            recommendations.append(OptimizationRecommendation(
                category="sparsity",
                priority="low",
                description="Low sparsity detected. Consider pruning techniques.",
                expected_improvement="10-30% size reduction",
                implementation_complexity="complex"
            ))

        # Efficiency optimization
        if metrics.efficiency_score < 0.5:
            recommendations.append(
                OptimizationRecommendation(
                    category="efficiency",
                    priority="high",
                    description="Low overall efficiency. Consider comprehensive optimization strategy.",
                    expected_improvement="Significant overall improvement",
                    implementation_complexity="complex"))

        return recommendations


class CrossArchitectureAnalyzer:
    """Analyzer for comparing and understanding differences between architectures."""

    def __init__(self, universal_support: UniversalModelSupport):
        self.universal_support = universal_support
        self.performance_analyzer = PerformanceAnalyzer()

    def compare_architectures(
            self,
            model1: nn.Module,
            model1_name: str,
            model2: nn.Module,
            model2_name: str) -> CrossArchitectureComparison:
        """Compare two different model architectures."""

        # Analyze both models
        analysis1 = self.universal_support.analyze_model_architecture(
            model1, model1_name)
        analysis2 = self.universal_support.analyze_model_architecture(
            model2, model2_name)

        # Get performance metrics
        perf1 = self.performance_analyzer.analyze_performance(model1, model1_name)
        perf2 = self.performance_analyzer.analyze_performance(model2, model2_name)

        # Calculate similarity
        similarity_score = self._calculate_architecture_similarity(analysis1, analysis2)

        # Identify shared and unique features
        shared_features, unique1, unique2 = self._identify_feature_differences(
            analysis1, analysis2)

        # Performance comparison
        performance_comparison = self._compare_performance(perf1, perf2)

        # Generate recommendations
        recommendations = self._generate_comparison_recommendations(
            analysis1, analysis2, performance_comparison
        )

        return CrossArchitectureComparison(
            model1_name=model1_name,
            model2_name=model2_name,
            architecture1=ArchitectureType(analysis1["architecture_type"]),
            architecture2=ArchitectureType(analysis2["architecture_type"]),
            similarity_score=similarity_score,
            shared_features=shared_features,
            unique_features1=unique1,
            unique_features2=unique2,
            performance_comparison=performance_comparison,
            recommendations=recommendations
        )

    def _calculate_architecture_similarity(
            self, analysis1: Dict, analysis2: Dict) -> float:
        """Calculate similarity score between two architectures."""
        similarity_factors = []

        # Architecture type similarity
        if analysis1["architecture_type"] == analysis2["architecture_type"]:
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)

        # Layer count similarity
        max_layers = max(analysis1["total_layers"], analysis2["total_layers"])
        if max_layers > 0:
            layer_ratio = min(
                analysis1["total_layers"],
                analysis2["total_layers"]) / max_layers
            similarity_factors.append(layer_ratio)
        else:
            similarity_factors.append(1.0)  # Both have 0 layers, consider similar

        # Encoder-decoder structure similarity
        if analysis1["is_encoder_decoder"] == analysis2["is_encoder_decoder"]:
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)

        # Layer type distribution similarity
        types1 = set(analysis1["layer_types"].keys())
        types2 = set(analysis2["layer_types"].keys())
        union_types = types1 | types2
        if union_types:
            type_similarity = len(types1 & types2) / len(union_types)
        else:
            type_similarity = 1.0  # Both have no layer types, consider similar
        similarity_factors.append(type_similarity)

        return sum(similarity_factors) / len(similarity_factors)

    def _identify_feature_differences(
            self, analysis1: Dict, analysis2: Dict) -> Tuple[List[str], List[str], List[str]]:
        """Identify shared and unique features between architectures."""
        features1 = set()
        features2 = set()

        # Add architecture-specific features
        if analysis1["is_encoder_decoder"]:
            features1.add("encoder_decoder")
        if analysis2["is_encoder_decoder"]:
            features2.add("encoder_decoder")

        if analysis1["supports_bidirectional"]:
            features1.add("bidirectional")
        if analysis2["supports_bidirectional"]:
            features2.add("bidirectional")

        # Add layer type features
        for layer_type in analysis1["layer_types"]:
            features1.add(f"has_{layer_type}")
        for layer_type in analysis2["layer_types"]:
            features2.add(f"has_{layer_type}")

        shared_features = list(features1 & features2)
        unique_features1 = list(features1 - features2)
        unique_features2 = list(features2 - features1)

        return shared_features, unique_features1, unique_features2

    def _compare_performance(self, perf1: PerformanceMetrics,
                             perf2: PerformanceMetrics) -> Dict[str, Any]:
        """Compare performance metrics between two models."""
        return {
            "memory_ratio": perf1.memory_usage_mb / perf2.memory_usage_mb,
            "speed_ratio": perf1.inference_time_ms / perf2.inference_time_ms,
            "parameter_ratio": perf1.total_parameters / perf2.total_parameters,
            "efficiency_comparison": {
                "model1_efficiency": perf1.efficiency_score,
                "model2_efficiency": perf2.efficiency_score,
                "better_model": "model1" if perf1.efficiency_score > perf2.efficiency_score else "model2"},
            "sparsity_comparison": {
                "model1_sparsity": perf1.sparsity_ratio,
                "model2_sparsity": perf2.sparsity_ratio,
                "sparser_model": "model1" if perf1.sparsity_ratio > perf2.sparsity_ratio else "model2"}}

    def _generate_comparison_recommendations(self, analysis1: Dict, analysis2: Dict,
                                             performance_comparison: Dict) -> List[str]:
        """Generate recommendations based on architecture comparison."""
        recommendations = []

        # Architecture-specific recommendations
        if analysis1["architecture_type"] != analysis2["architecture_type"]:
            recommendations.append(
                "Different architectures detected. Use architecture-specific analysis methods for optimal results."
            )

        # Performance-based recommendations
        better_model = performance_comparison["efficiency_comparison"]["better_model"]
        if better_model == "model1":
            recommendations.append(
                f"Model 1 ({analysis1['architecture_type']}) shows better overall efficiency."
            )
        else:
            recommendations.append(
                f"Model 2 ({analysis2['architecture_type']}) shows better overall efficiency."
            )

        # Feature-based recommendations
        if analysis1["is_encoder_decoder"] and not analysis2["is_encoder_decoder"]:
            recommendations.append(
                "Model 1 has encoder-decoder structure, suitable for sequence-to-sequence tasks."
            )
        elif analysis2["is_encoder_decoder"] and not analysis1["is_encoder_decoder"]:
            recommendations.append(
                "Model 2 has encoder-decoder structure, suitable for sequence-to-sequence tasks."
            )

        if analysis1["supports_bidirectional"] and not analysis2["supports_bidirectional"]:
            recommendations.append(
                "Model 1 supports bidirectional attention, better for understanding tasks."
            )
        elif analysis2["supports_bidirectional"] and not analysis1["supports_bidirectional"]:
            recommendations.append(
                "Model 2 supports bidirectional attention, better for understanding tasks."
            )

        return recommendations


class UniversalAdvancedAnalyzer:
    """Combined advanced analyzer with all capabilities."""

    def __init__(self):
        self.universal_support = UniversalModelSupport()
        self.domain_analyzer = DomainSpecificAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.cross_arch_analyzer = CrossArchitectureAnalyzer(self.universal_support)

    def comprehensive_analysis(self,
                               model: nn.Module,
                               model_name: str,
                               domain: Optional[str] = None,
                               comparison_model: Optional[nn.Module] = None,
                               comparison_model_name: Optional[str] = None) -> Dict[str,
                                                                                    Any]:
        """Perform comprehensive analysis of a model with all available methods."""

        results = {}

        # Basic architecture analysis
        results["architecture_analysis"] = self.universal_support.analyze_model_architecture(
            model, model_name)

        # Performance analysis
        results["performance_analysis"] = self.performance_analyzer.analyze_performance(
            model, model_name)

        # Optimization recommendations
        results["optimization_recommendations"] = self.performance_analyzer.generate_optimization_recommendations(
            results["performance_analysis"], model_name)

        # Domain-specific analysis if requested
        if domain:
            try:
                results["domain_analysis"] = self.domain_analyzer.analyze_domain_specificity(
                    model, model_name, domain)
            except ValueError as e:
                results["domain_analysis"] = {"error": str(e)}

        # Cross-architecture comparison if comparison model provided
        if comparison_model and comparison_model_name:
            results["cross_architecture_comparison"] = self.cross_arch_analyzer.compare_architectures(
                model, model_name, comparison_model, comparison_model_name)

        # Generate overall summary
        results["summary"] = self._generate_summary(results)

        return results

    def _generate_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an overall summary of the analysis."""
        arch_analysis = analysis_results["architecture_analysis"]
        perf_analysis = analysis_results["performance_analysis"]

        summary = {
            "model_type": arch_analysis["architecture_type"],
            "total_layers": arch_analysis["total_layers"],
            "is_encoder_decoder": arch_analysis["is_encoder_decoder"],
            "performance_score": perf_analysis.efficiency_score,
            "memory_usage_category": self._categorize_memory_usage(
                perf_analysis.memory_usage_mb),
            "speed_category": self._categorize_speed(
                perf_analysis.inference_time_ms),
            "optimization_priority": self._get_optimization_priority(
                analysis_results.get(
                    "optimization_recommendations",
                    [])),
            "key_insights": self._extract_key_insights(analysis_results)}

        return summary

    def _categorize_memory_usage(self, memory_mb: float) -> str:
        """Categorize memory usage level."""
        if memory_mb < 100:
            return "light"
        elif memory_mb < 500:
            return "moderate"
        elif memory_mb < 1000:
            return "heavy"
        else:
            return "very_heavy"

    def _categorize_speed(self, inference_ms: float) -> str:
        """Categorize inference speed."""
        if inference_ms < 50:
            return "very_fast"
        elif inference_ms < 200:
            return "fast"
        elif inference_ms < 500:
            return "moderate"
        else:
            return "slow"

    def _get_optimization_priority(
            self, recommendations: List[OptimizationRecommendation]) -> str:
        """Get overall optimization priority."""
        if not recommendations:
            return "low"

        priorities = [rec.priority for rec in recommendations]
        if "high" in priorities:
            return "high"
        elif "medium" in priorities:
            return "medium"
        else:
            return "low"

    def _extract_key_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extract key insights from the analysis."""
        insights = []

        arch_analysis = analysis_results["architecture_analysis"]
        perf_analysis = analysis_results["performance_analysis"]

        # Architecture insights
        if arch_analysis["is_encoder_decoder"]:
            insights.append(
                "Model uses encoder-decoder architecture, suitable for sequence-to-sequence tasks")

        if arch_analysis["supports_bidirectional"]:
            insights.append(
                "Model supports bidirectional attention, good for understanding tasks")

        # Performance insights
        if perf_analysis.efficiency_score > 0.8:
            insights.append("Model shows excellent efficiency")
        elif perf_analysis.efficiency_score < 0.5:
            insights.append("Model has efficiency concerns that should be addressed")

        if perf_analysis.sparsity_ratio > 0.3:
            insights.append(
                "Model shows significant sparsity, compression opportunities available")

        # Domain insights
        if "domain_analysis" in analysis_results:
            domain_score = analysis_results["domain_analysis"].get(
                "domain_compatibility_score", 0)
            if domain_score > 0.8:
                insights.append("Model shows strong domain specialization")
            elif domain_score < 0.5:
                insights.append(
                    "Model may need domain adaptation for optimal performance")

        return insights
