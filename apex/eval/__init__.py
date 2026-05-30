"""APEX-1 evaluation utilities.

This package is intentionally small and CPU-friendly. It helps learners answer
three practical questions:

1. Is my model producing numerically valid logits?
2. How do I measure simple language-model metrics such as loss/perplexity?
3. How do I benchmark and sanity-check generation/vision outputs?

The goal is not to replace production eval suites such as lm-eval-harness. The
goal is to make evaluation understandable inside the APEX-1 course codebase.
"""

from apex.eval.benchmark import BenchmarkResult, run_forward_benchmark
from apex.eval.generation_quality import (
    GenerationQualityReport,
    average_length,
    distinct_n,
    evaluate_generated_texts,
    repetition_rate,
)
from apex.eval.metrics import ClassificationMetrics, next_token_accuracy, token_cross_entropy
from apex.eval.perplexity import PerplexityResult, compute_perplexity
from apex.eval.vision_eval import VisionEvalReport, validate_vision_forward_output

__all__ = [
    "BenchmarkResult",
    "run_forward_benchmark",
    "GenerationQualityReport",
    "average_length",
    "distinct_n",
    "evaluate_generated_texts",
    "repetition_rate",
    "ClassificationMetrics",
    "next_token_accuracy",
    "token_cross_entropy",
    "PerplexityResult",
    "compute_perplexity",
    "VisionEvalReport",
    "validate_vision_forward_output",
]
