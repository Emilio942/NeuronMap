"""
NeuronMap Python API Client

Provides a high-level Python interface for interacting with NeuronMap,
both locally and through the REST API.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import concurrent.futures
from dataclasses import dataclass

try:
    import requests
    import aiohttp
except ImportError:
    requests = None
    aiohttp = None

# Internal imports
from core.neuron_map import NeuronMap
from config.config_manager import ConfigManager
from utils.error_handling import NeuronMapError
from utils.monitoring import setup_monitoring

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Represents the result of an analysis operation."""
    analysis_id: str
    model_name: str
    analysis_type: str
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: str


@dataclass
class JobInfo:
    """Information about a background job."""
    job_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class NeuronMapClient:
    """
    High-level Python client for NeuronMap.

    Can work both locally (direct integration) and remotely (via REST API).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config_path: Optional[str] = None,
        timeout: int = 300
    ):
        """
        Initialize the NeuronMap client.

        Args:
            base_url: URL of the NeuronMap API server (None for local mode)
            api_key: API key for authentication
            config_path: Path to configuration file
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

        # Setup monitoring
        self.monitor = setup_monitoring()

        if base_url:
            # Remote mode - use REST API
            if requests is None:
                raise ImportError("requests is required for remote API access")
            self.mode = "remote"
            self.session = requests.Session()
            if api_key:
                self.session.headers.update({"Authorization": f"Bearer {api_key}"})
            logger.info(f"Initialized NeuronMap client in remote mode: {base_url}")
        else:
            # Local mode - direct integration
            self.mode = "local"
            config_manager = ConfigManager()
            if config_path:
                self.config = config_manager.load_config(config_path)
            else:
                self.config = config_manager.get_default_config()

            self.neuron_map = NeuronMap(config=self.config)
            logger.info("Initialized NeuronMap client in local mode")

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        if self.mode == "remote":
            response = self.session.get(f"{self.base_url}/models", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        else:
            # In local mode, return built-in model support
            return [
                {
                    "name": "bert-base-uncased",
                    "type": "huggingface",
                    "supported_tasks": ["text-analysis", "sentiment"],
                    "description": "BERT base model, uncased"
                },
                {
                    "name": "gpt2",
                    "type": "huggingface",
                    "supported_tasks": ["text-generation", "analysis"],
                    "description": "GPT-2 base model"
                }
            ]

    def analyze_text(
        self,
        text: str,
        model_name: str = "bert-base-uncased",
        analysis_type: str = "basic",
        layers: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None,
        wait_for_completion: bool = True
    ) -> Union[AnalysisResult, JobInfo]:
        """
        Analyze text with specified model.

        Args:
            text: Text to analyze
            model_name: Name of the model to use
            analysis_type: Type of analysis (basic, sentiment, interpretability)
            layers: Specific layers to analyze
            config: Additional configuration
            wait_for_completion: Whether to wait for analysis to complete

        Returns:
            AnalysisResult if wait_for_completion=True, JobInfo otherwise
        """
        if self.mode == "remote":
            return self._analyze_text_remote(
                text, model_name, analysis_type, layers, config, wait_for_completion
            )
        else:
            return self._analyze_text_local(
                text, model_name, analysis_type, layers, config
            )

    def _analyze_text_remote(
        self,
        text: str,
        model_name: str,
        analysis_type: str,
        layers: Optional[List[int]],
        config: Optional[Dict[str, Any]],
        wait_for_completion: bool
    ) -> Union[AnalysisResult, JobInfo]:
        """Analyze text using remote API."""

        # Prepare request
        request_data = {
            "model_name": model_name,
            "input_text": text,
            "analysis_type": analysis_type
        }

        if layers:
            if not config:
                config = {}
            config.setdefault("analysis", {})["layers"] = layers

        if config:
            request_data["config"] = config

        # Submit analysis job
        response = self.session.post(
            f"{self.base_url}/analyze",
            json=request_data,
            timeout=self.timeout
        )
        response.raise_for_status()
        job_data = response.json()

        job_id = job_data["job_id"]

        if not wait_for_completion:
            return JobInfo(job_id=job_id, status="submitted")

        # Wait for completion
        return self._wait_for_job_completion(job_id)

    def _analyze_text_local(
        self,
        text: str,
        model_name: str,
        analysis_type: str,
        layers: Optional[List[int]],
        config: Optional[Dict[str, Any]]
    ) -> AnalysisResult:
        """Analyze text using local NeuronMap instance."""

        # Load model
        model_config = {
            'name': model_name,
            'type': 'huggingface'
        }

        if config and 'model' in config:
            model_config.update(config['model'])

        model = self.neuron_map.load_model(model_config)

        # Prepare analysis parameters
        analysis_params = {
            'layers': layers or [0, 6, 11],
            'include_attention': True,
            'include_hidden_states': True
        }

        if config and 'analysis' in config:
            analysis_params.update(config['analysis'])

        # Run analysis
        if analysis_type == "sentiment":
            results = self.neuron_map.analyze_sentiment(
                model=model,
                texts=[text],
                **analysis_params
            )
        elif analysis_type == "interpretability":
            results = self.neuron_map.analyze_interpretability(
                model=model,
                input_text=text,
                **analysis_params
            )
        else:  # basic analysis
            results = self.neuron_map.generate_activations(
                text=text,
                **analysis_params
            )

        # Create result object
        import uuid
        from datetime import datetime

        analysis_id = str(uuid.uuid4())

        return AnalysisResult(
            analysis_id=analysis_id,
            model_name=model_name,
            analysis_type=analysis_type,
            results=results,
            metadata={
                'layers_analyzed': analysis_params['layers'],
                'input_length': len(text),
                'model_config': model_config
            },
            created_at=datetime.now().isoformat()
        )

    def get_job_status(self, job_id: str) -> JobInfo:
        """Get the status of a background job."""
        if self.mode == "local":
            raise ValueError("Job status only available in remote mode")

        response = self.session.get(
            f"{self.base_url}/jobs/{job_id}",
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        return JobInfo(**data)

    def get_analysis_results(self, analysis_id: str) -> AnalysisResult:
        """Get analysis results by ID."""
        if self.mode == "local":
            raise ValueError("Result retrieval by ID only available in remote mode")

        response = self.session.get(
            f"{self.base_url}/results/{analysis_id}",
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        return AnalysisResult(**data)

    def _wait_for_job_completion(
            self,
            job_id: str,
            check_interval: float = 2.0) -> AnalysisResult:
        """Wait for a job to complete and return results."""
        while True:
            job_info = self.get_job_status(job_id)

            if job_info.status == "completed":
                # Get analysis results
                analysis_id = job_info.result["analysis_id"]
                return self.get_analysis_results(analysis_id)

            elif job_info.status == "failed":
                raise NeuronMapError(f"Analysis failed: {job_info.error}")

            elif job_info.status in ["pending", "running"]:
                # Still processing, wait and check again
                time.sleep(check_interval)

            else:
                raise NeuronMapError(f"Unknown job status: {job_info.status}")

    def create_visualization(
        self,
        analysis_result: AnalysisResult,
        plot_type: str = "heatmap",
        config: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create visualization from analysis results.

        Args:
            analysis_result: Analysis result to visualize
            plot_type: Type of plot to create
            config: Visualization configuration
            save_path: Path to save visualization

        Returns:
            Path to created visualization
        """
        if self.mode == "remote":
            # Use remote visualization API
            request_data = {
                "analysis_id": analysis_result.analysis_id,
                "plot_type": plot_type
            }

            if config:
                request_data["config"] = config

            response = self.session.post(
                f"{self.base_url}/visualize",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            job_data = response.json()

            # Wait for visualization completion
            job_id = job_data["job_id"]
            job_info = self._wait_for_visualization_completion(job_id)

            return job_info.result["path"]

        else:
            # Local visualization
            viz_config = config or {}
            if save_path:
                viz_config["save_path"] = save_path

            return self.neuron_map.create_visualization(
                results=analysis_result.results,
                plot_type=plot_type,
                **viz_config
            )

    def _wait_for_visualization_completion(self, job_id: str) -> JobInfo:
        """Wait for visualization job to complete."""
        while True:
            job_info = self.get_job_status(job_id)

            if job_info.status == "completed":
                return job_info
            elif job_info.status == "failed":
                raise NeuronMapError(f"Visualization failed: {job_info.error}")
            elif job_info.status in ["pending", "running"]:
                time.sleep(1.0)
            else:
                raise NeuronMapError(f"Unknown job status: {job_info.status}")

    def batch_analyze(
        self,
        texts: List[str],
        model_name: str = "bert-base-uncased",
        analysis_type: str = "basic",
        max_workers: int = 4,
        **kwargs
    ) -> List[AnalysisResult]:
        """
        Analyze multiple texts in parallel.

        Args:
            texts: List of texts to analyze
            model_name: Model to use
            analysis_type: Type of analysis
            max_workers: Maximum number of parallel workers
            **kwargs: Additional arguments for analyze_text

        Returns:
            List of analysis results
        """
        results = []

        if self.mode == "local":
            # Use local parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for text in texts:
                    future = executor.submit(
                        self.analyze_text,
                        text=text,
                        model_name=model_name,
                        analysis_type=analysis_type,
                        **kwargs
                    )
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

        else:
            # Remote mode - submit all jobs then wait
            job_ids = []

            # Submit all jobs
            for text in texts:
                job_info = self.analyze_text(
                    text=text,
                    model_name=model_name,
                    analysis_type=analysis_type,
                    wait_for_completion=False,
                    **kwargs
                )
                job_ids.append(job_info.job_id)

            # Wait for all to complete
            for job_id in job_ids:
                result = self._wait_for_job_completion(job_id)
                results.append(result)

        return results

    def compare_models(
        self,
        text: str,
        model_names: List[str],
        analysis_type: str = "basic",
        **kwargs
    ) -> Dict[str, AnalysisResult]:
        """
        Compare multiple models on the same text.

        Args:
            text: Text to analyze
            model_names: List of model names to compare
            analysis_type: Type of analysis
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping model names to results
        """
        results = {}

        for model_name in model_names:
            result = self.analyze_text(
                text=text,
                model_name=model_name,
                analysis_type=analysis_type,
                **kwargs
            )
            results[model_name] = result

        return results

    def close(self):
        """Close the client and clean up resources."""
        if self.mode == "remote" and hasattr(self, 'session'):
            self.session.close()

        logger.info("NeuronMap client closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class AsyncNeuronMapClient:
    """
    Asynchronous version of the NeuronMap client.

    Provides async/await interface for non-blocking operations.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 300
    ):
        """Initialize async client."""
        if aiohttp is None:
            raise ImportError("aiohttp is required for async client")

        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

        self.session = None

        if not base_url:
            raise ValueError("Async client requires remote API (base_url)")

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )

        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models asynchronously."""
        async with self.session.get(f"{self.base_url}/models") as response:
            response.raise_for_status()
            return await response.json()

    async def analyze_text(
        self,
        text: str,
        model_name: str = "bert-base-uncased",
        analysis_type: str = "basic",
        layers: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None,
        wait_for_completion: bool = True
    ) -> Union[AnalysisResult, JobInfo]:
        """Analyze text asynchronously."""

        # Prepare request
        request_data = {
            "model_name": model_name,
            "input_text": text,
            "analysis_type": analysis_type
        }

        if layers:
            if not config:
                config = {}
            config.setdefault("analysis", {})["layers"] = layers

        if config:
            request_data["config"] = config

        # Submit job
        async with self.session.post(f"{self.base_url}/analyze", json=request_data) as response:
            response.raise_for_status()
            job_data = await response.json()

        job_id = job_data["job_id"]

        if not wait_for_completion:
            return JobInfo(job_id=job_id, status="submitted")

        # Wait for completion
        return await self._wait_for_job_completion(job_id)

    async def get_job_status(self, job_id: str) -> JobInfo:
        """Get job status asynchronously."""
        async with self.session.get(f"{self.base_url}/jobs/{job_id}") as response:
            response.raise_for_status()
            data = await response.json()
            return JobInfo(**data)

    async def get_analysis_results(self, analysis_id: str) -> AnalysisResult:
        """Get analysis results asynchronously."""
        async with self.session.get(f"{self.base_url}/results/{analysis_id}") as response:
            response.raise_for_status()
            data = await response.json()
            return AnalysisResult(**data)

    async def _wait_for_job_completion(self, job_id: str, check_interval: float = 2.0) -> AnalysisResult:
        """Wait for job completion asynchronously."""
        while True:
            job_info = await self.get_job_status(job_id)

            if job_info.status == "completed":
                analysis_id = job_info.result["analysis_id"]
                return await self.get_analysis_results(analysis_id)

            elif job_info.status == "failed":
                raise NeuronMapError(f"Analysis failed: {job_info.error}")

            elif job_info.status in ["pending", "running"]:
                await asyncio.sleep(check_interval)

            else:
                raise NeuronMapError(f"Unknown job status: {job_info.status}")

    async def batch_analyze(
        self,
        texts: List[str],
        model_name: str = "bert-base-uncased",
        analysis_type: str = "basic",
        **kwargs
    ) -> List[AnalysisResult]:
        """Analyze multiple texts concurrently."""

        # Submit all jobs concurrently
        tasks = []
        for text in texts:
            task = self.analyze_text(
                text=text,
                model_name=model_name,
                analysis_type=analysis_type,
                wait_for_completion=True,
                **kwargs
            )
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        return results

# Convenience functions for quick usage


def quick_analyze(
    text: str,
    model_name: str = "bert-base-uncased",
    analysis_type: str = "basic"
) -> AnalysisResult:
    """Quick local analysis function."""
    with NeuronMapClient() as client:
        return client.analyze_text(
            text=text,
            model_name=model_name,
            analysis_type=analysis_type
        )


def remote_analyze(
    text: str,
    base_url: str,
    api_key: Optional[str] = None,
    model_name: str = "bert-base-uncased",
    analysis_type: str = "basic"
) -> AnalysisResult:
    """Quick remote analysis function."""
    with NeuronMapClient(base_url=base_url, api_key=api_key) as client:
        return client.analyze_text(
            text=text,
            model_name=model_name,
            analysis_type=analysis_type
        )


# Example usage
if __name__ == "__main__":
    # Local usage example
    print("Local analysis example:")
    result = quick_analyze(
        text="This is a great example!",
        analysis_type="sentiment"
    )
    print(f"Analysis ID: {result.analysis_id}")
    print(f"Model: {result.model_name}")

    # Remote usage example (commented out)
    # print("Remote analysis example:")
    # result = remote_analyze(
    #     text="This is a great example!",
    #     base_url="http://localhost:8000",
    #     analysis_type="sentiment"
    # )
    # print(f"Analysis ID: {result.analysis_id}")
