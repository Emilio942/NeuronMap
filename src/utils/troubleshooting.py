"""
Intelligent Troubleshooting System for NeuronMap
==============================================

This module provides automated problem detection, solution suggestions,
and community knowledge base integration for common issues.
"""

import re
import json
import logging
import traceback
import platform
import psutil
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import sys
from datetime import datetime

logger = logging.getLogger(__name__)


class ProblemSeverity(Enum):
    """Severity levels for detected problems."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SolutionConfidence(Enum):
    """Confidence levels for solution suggestions."""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class SystemInfo:
    """System information for diagnostics."""
    os_type: str
    os_version: str
    python_version: str
    gpu_available: bool
    gpu_memory: Optional[float]
    ram_total: float
    ram_available: float
    disk_space: float
    torch_version: str
    cuda_version: Optional[str]
    
    @classmethod
    def collect(cls) -> 'SystemInfo':
        """Collect current system information."""
        gpu_available = torch.cuda.is_available()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if gpu_available else None
        cuda_version = torch.version.cuda if gpu_available else None
        
        return cls(
            os_type=platform.system(),
            os_version=platform.version(),
            python_version=platform.python_version(),
            gpu_available=gpu_available,
            gpu_memory=gpu_memory,
            ram_total=psutil.virtual_memory().total / 1e9,
            ram_available=psutil.virtual_memory().available / 1e9,
            disk_space=psutil.disk_usage('/').free / 1e9,
            torch_version=torch.__version__,
            cuda_version=cuda_version
        )


@dataclass
class Solution:
    """A solution suggestion for a detected problem."""
    description: str
    steps: List[str]
    verification_steps: List[str] = field(default_factory=list)
    alternative_solutions: List['Solution'] = field(default_factory=list)
    confidence: SolutionConfidence = SolutionConfidence.MEDIUM
    auto_fix_available: bool = False
    prevention_tips: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert solution to dictionary."""
        return {
            'description': self.description,
            'steps': self.steps,
            'verification_steps': self.verification_steps,
            'confidence': self.confidence.value,
            'auto_fix_available': self.auto_fix_available,
            'prevention_tips': self.prevention_tips
        }


@dataclass
class DetectedProblem:
    """A detected problem with potential solutions."""
    problem_type: str
    severity: ProblemSeverity
    message: str
    context: Dict[str, Any]
    solutions: List[Solution] = field(default_factory=list)
    
    def add_solution(self, solution: Solution):
        """Add a solution to this problem."""
        self.solutions.append(solution)
    
    def get_best_solution(self) -> Optional[Solution]:
        """Get the solution with highest confidence."""
        if not self.solutions:
            return None
        return max(self.solutions, key=lambda s: s.confidence.value)


class ErrorPatternAnalyzer:
    """Analyzes error patterns to identify common issues."""
    
    def __init__(self):
        self.error_patterns = {
            'memory_error': [
                r'CUDA out of memory',
                r'RuntimeError.*memory',
                r'torch.cuda.OutOfMemoryError',
                r'MemoryError'
            ],
            'model_loading_error': [
                r'No module named.*transformers',
                r'Model.*not found',
                r'HTTPSConnectionPool.*timed out',
                r'Connection.*refused.*11434'  # Ollama connection
            ],
            'dependency_error': [
                r'ModuleNotFoundError',
                r'ImportError',
                r'No module named'
            ],
            'file_not_found': [
                r'FileNotFoundError',
                r'No such file or directory',
                r'\[Errno 2\]'
            ],
            'permission_error': [
                r'PermissionError',
                r'Permission denied',
                r'\[Errno 13\]'
            ],
            'configuration_error': [
                r'ValidationError',
                r'Invalid configuration',
                r'Config.*not found'
            ],
            'gpu_error': [
                r'CUDA.*not available',
                r'No CUDA devices',
                r'cuDNN error'
            ]
        }
    
    def analyze_error(self, error_log: str) -> List[str]:
        """Analyze error log and return list of detected problem types."""
        detected_problems = []
        
        for problem_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, error_log, re.IGNORECASE):
                    detected_problems.append(problem_type)
                    break
        
        return detected_problems


class EnvironmentChecker:
    """Checks environment compatibility and common issues."""
    
    def check_environment(self, system_info: SystemInfo) -> List[DetectedProblem]:
        """Check environment for potential issues."""
        problems = []
        
        # Check memory requirements
        if system_info.ram_available < 4.0:
            problems.append(DetectedProblem(
                problem_type="low_memory",
                severity=ProblemSeverity.HIGH,
                message=f"Low available RAM: {system_info.ram_available:.1f}GB",
                context={'available_ram': system_info.ram_available}
            ))
        
        # Check GPU availability
        if not system_info.gpu_available:
            problems.append(DetectedProblem(
                problem_type="no_gpu",
                severity=ProblemSeverity.MEDIUM,
                message="No GPU available - computations will be slow",
                context={'gpu_available': False}
            ))
        
        # Check disk space
        if system_info.disk_space < 5.0:
            problems.append(DetectedProblem(
                problem_type="low_disk_space",
                severity=ProblemSeverity.HIGH,
                message=f"Low disk space: {system_info.disk_space:.1f}GB",
                context={'disk_space': system_info.disk_space}
            ))
        
        # Check Python version
        python_version = tuple(map(int, system_info.python_version.split('.')))
        if python_version < (3, 8):
            problems.append(DetectedProblem(
                problem_type="old_python",
                severity=ProblemSeverity.HIGH,
                message=f"Python {system_info.python_version} is too old",
                context={'python_version': system_info.python_version}
            ))
        
        return problems


class SolutionDatabase:
    """Database of solutions for common problems."""
    
    def __init__(self):
        self.solutions = self._initialize_solutions()
    
    def _initialize_solutions(self) -> Dict[str, List[Solution]]:
        """Initialize the solutions database."""
        return {
            'memory_error': [
                Solution(
                    description="Reduce batch size to fit in available memory",
                    steps=[
                        "Reduce batch_size parameter (try halving it)",
                        "Use gradient accumulation instead of large batches",
                        "Enable model parallelism if multiple GPUs available"
                    ],
                    verification_steps=["Run the analysis again with smaller batch size"],
                    confidence=SolutionConfidence.HIGH,
                    prevention_tips=[
                        "Monitor GPU memory usage with nvidia-smi",
                        "Start with small batch sizes and increase gradually"
                    ]
                ),
                Solution(
                    description="Use CPU instead of GPU for smaller models",
                    steps=[
                        "Set device='cpu' in configuration",
                        "Ensure model is small enough for CPU processing"
                    ],
                    verification_steps=["Check that model loads without memory errors"],
                    confidence=SolutionConfidence.MEDIUM
                )
            ],
            'model_loading_error': [
                Solution(
                    description="Install missing transformers library",
                    steps=[
                        "pip install transformers",
                        "pip install torch",
                        "Restart your Python environment"
                    ],
                    verification_steps=["Try importing transformers in Python"],
                    confidence=SolutionConfidence.VERY_HIGH,
                    auto_fix_available=True
                ),
                Solution(
                    description="Check network connection and retry",
                    steps=[
                        "Verify internet connection",
                        "Try downloading model manually",
                        "Check firewall settings"
                    ],
                    verification_steps=["Test download with: huggingface-cli download model_name"],
                    confidence=SolutionConfidence.HIGH
                )
            ],
            'dependency_error': [
                Solution(
                    description="Install missing dependencies",
                    steps=[
                        "pip install -r requirements.txt",
                        "Check if virtual environment is activated",
                        "Update pip: pip install --upgrade pip"
                    ],
                    verification_steps=["Import the missing module in Python"],
                    confidence=SolutionConfidence.VERY_HIGH,
                    auto_fix_available=True
                )
            ],
            'file_not_found': [
                Solution(
                    description="Verify file paths and create missing directories",
                    steps=[
                        "Check if the file path is correct",
                        "Create missing directories: mkdir -p /path/to/directory",
                        "Verify file permissions"
                    ],
                    verification_steps=["Check that the file exists: ls -la /path/to/file"],
                    confidence=SolutionConfidence.HIGH
                )
            ],
            'low_memory': [
                Solution(
                    description="Free up system memory",
                    steps=[
                        "Close unnecessary applications",
                        "Use smaller models (e.g., distilgpt2 instead of gpt2-large)",
                        "Increase virtual memory/swap space"
                    ],
                    verification_steps=["Check available memory: free -h"],
                    confidence=SolutionConfidence.HIGH
                )
            ],
            'no_gpu': [
                Solution(
                    description="Use CPU processing with optimizations",
                    steps=[
                        "Set device='cpu' in configuration",
                        "Use smaller models for CPU processing",
                        "Enable threading: export OMP_NUM_THREADS=4"
                    ],
                    verification_steps=["Monitor CPU usage during processing"],
                    confidence=SolutionConfidence.HIGH
                )
            ],
            'low_disk_space': [
                Solution(
                    description="Free up disk space",
                    steps=[
                        "Remove temporary files: rm -rf /tmp/*",
                        "Clear pip cache: pip cache purge",
                        "Move large files to external storage"
                    ],
                    verification_steps=["Check disk space: df -h"],
                    confidence=SolutionConfidence.HIGH
                )
            ]
        }
    
    def get_solutions(self, problem_type: str) -> List[Solution]:
        """Get solutions for a specific problem type."""
        return self.solutions.get(problem_type, [])


class TroubleshootingEngine:
    """Main troubleshooting engine that diagnoses and suggests solutions."""
    
    def __init__(self):
        self.error_analyzer = ErrorPatternAnalyzer()
        self.env_checker = EnvironmentChecker()
        self.solution_db = SolutionDatabase()
        self.knowledge_base = self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load community knowledge base if available."""
        kb_file = Path("troubleshooting_kb.json")
        if kb_file.exists():
            with open(kb_file, 'r') as f:
                return json.load(f)
        return {}
    
    def diagnose_common_issues(self, error_log: str = None, system_info: SystemInfo = None) -> List[DetectedProblem]:
        """Diagnose common issues from error log and system information."""
        problems = []
        
        # Collect system info if not provided
        if system_info is None:
            system_info = SystemInfo.collect()
        
        # Check environment issues
        env_problems = self.env_checker.check_environment(system_info)
        problems.extend(env_problems)
        
        # Analyze error log if provided
        if error_log:
            error_types = self.error_analyzer.analyze_error(error_log)
            
            for error_type in error_types:
                problem = DetectedProblem(
                    problem_type=error_type,
                    severity=self._get_severity(error_type),
                    message=f"Detected {error_type.replace('_', ' ')} in error log",
                    context={'error_log_snippet': error_log[:500]}
                )
                problems.append(problem)
        
        # Add solutions to problems
        for problem in problems:
            solutions = self.solution_db.get_solutions(problem.problem_type)
            for solution in solutions:
                problem.add_solution(solution)
        
        return self.rank_solutions_by_probability(problems)
    
    def _get_severity(self, error_type: str) -> ProblemSeverity:
        """Get severity level for error type."""
        severity_map = {
            'memory_error': ProblemSeverity.HIGH,
            'model_loading_error': ProblemSeverity.HIGH,
            'dependency_error': ProblemSeverity.HIGH,
            'file_not_found': ProblemSeverity.MEDIUM,
            'permission_error': ProblemSeverity.MEDIUM,
            'configuration_error': ProblemSeverity.MEDIUM,
            'gpu_error': ProblemSeverity.LOW
        }
        return severity_map.get(error_type, ProblemSeverity.MEDIUM)
    
    def rank_solutions_by_probability(self, problems: List[DetectedProblem]) -> List[DetectedProblem]:
        """Rank solutions by probability of success."""
        for problem in problems:
            # Sort solutions by confidence
            problem.solutions.sort(key=lambda s: s.confidence.value, reverse=True)
        
        # Sort problems by severity
        problems.sort(key=lambda p: p.severity.value, reverse=True)
        
        return problems
    
    def auto_fix_problem(self, problem: DetectedProblem) -> bool:
        """Attempt to automatically fix a problem if auto-fix is available."""
        best_solution = problem.get_best_solution()
        
        if not best_solution or not best_solution.auto_fix_available:
            return False
        
        try:
            if problem.problem_type == 'dependency_error':
                return self._auto_fix_dependencies()
            elif problem.problem_type == 'model_loading_error':
                return self._auto_fix_model_loading()
            # Add more auto-fix implementations as needed
            
        except Exception as e:
            logger.error(f"Auto-fix failed for {problem.problem_type}: {e}")
            return False
        
        return False
    
    def _auto_fix_dependencies(self) -> bool:
        """Auto-fix dependency issues."""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], capture_output=True, text=True, timeout=300)
            return result.returncode == 0
        except Exception:
            return False
    
    def _auto_fix_model_loading(self) -> bool:
        """Auto-fix model loading issues."""
        try:
            # Try to install transformers if missing
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 'transformers', 'torch'
            ], capture_output=True, text=True, timeout=300)
            return result.returncode == 0
        except Exception:
            return False
    
    def generate_diagnostic_report(self, problems: List[DetectedProblem]) -> Dict[str, Any]:
        """Generate a comprehensive diagnostic report."""
        system_info = SystemInfo.collect()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'os': f"{system_info.os_type} {system_info.os_version}",
                'python': system_info.python_version,
                'gpu_available': system_info.gpu_available,
                'ram_available': f"{system_info.ram_available:.1f}GB",
                'disk_space': f"{system_info.disk_space:.1f}GB"
            },
            'problems_detected': len(problems),
            'critical_issues': len([p for p in problems if p.severity == ProblemSeverity.CRITICAL]),
            'high_priority_issues': len([p for p in problems if p.severity == ProblemSeverity.HIGH]),
            'problems': []
        }
        
        for problem in problems:
            problem_data = {
                'type': problem.problem_type,
                'severity': problem.severity.value,
                'message': problem.message,
                'solutions_available': len(problem.solutions),
                'auto_fix_available': any(s.auto_fix_available for s in problem.solutions),
                'best_solution': problem.get_best_solution().to_dict() if problem.get_best_solution() else None
            }
            report['problems'].append(problem_data)
        
        return report


def run_quick_diagnosis() -> Dict[str, Any]:
    """Run a quick diagnosis of the current system."""
    engine = TroubleshootingEngine()
    
    # Collect recent error logs if available
    error_log = ""
    log_file = Path("neuronmap.log")
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Get last 100 lines
            error_log = ''.join(lines[-100:])
    
    problems = engine.diagnose_common_issues(error_log)
    report = engine.generate_diagnostic_report(problems)
    
    return report


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuronMap Troubleshooting System')
    parser.add_argument('--quick', action='store_true', help='Run quick diagnosis')
    parser.add_argument('--error-log', help='Path to error log file')
    parser.add_argument('--auto-fix', action='store_true', help='Attempt auto-fix of detected issues')
    parser.add_argument('--output', help='Output file for diagnostic report')
    
    args = parser.parse_args()
    
    engine = TroubleshootingEngine()
    
    if args.quick:
        report = run_quick_diagnosis()
        print("Quick Diagnosis Results:")
        print(f"Problems detected: {report['problems_detected']}")
        print(f"Critical issues: {report['critical_issues']}")
        print(f"High priority issues: {report['high_priority_issues']}")
    else:
        error_log = ""
        if args.error_log and Path(args.error_log).exists():
            with open(args.error_log, 'r') as f:
                error_log = f.read()
        
        problems = engine.diagnose_common_issues(error_log)
        
        if args.auto_fix:
            for problem in problems:
                if engine.auto_fix_problem(problem):
                    print(f"Auto-fixed: {problem.problem_type}")
        
        report = engine.generate_diagnostic_report(problems)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Diagnostic report saved to {args.output}")
        else:
            print("Diagnostic Report:")
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
