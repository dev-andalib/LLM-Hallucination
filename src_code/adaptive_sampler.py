# src/adaptive_sampler.py
import time
from typing import Callable, List, Dict, Any

def adaptive_sampling_loop(prompt_id: str,
                           sampler_fn: Callable[[], Dict],
                           estimator,
                           gamma: float = 0.01,
                           initial_samples: int = 2,
                           max_samples: int = 50) -> Dict:
    """
    Adaptive sampling controller.

    sampler_fn: a callable that returns one sample dict: {'text': str, 'meaning_id': int, 'prob': float}
    estimator: instance of BayesianEntropyEstimator with .estimate(samples) -> dict
    Returns final dict with E_h, Var_h, samples list, and meta info.
    """
    samples = []
    # initial sampling
    for _ in range(initial_samples):
        s = sampler_fn()
        samples.append(s)

    result = estimator.estimate(samples)
    # continue until Var[h] <= gamma or max_samples
    while result["Var_h"] > gamma and len(samples) < max_samples:
        s = sampler_fn()
        samples.append(s)
        result = estimator.estimate(samples)

    result.update({
        "samples": samples,
        "n_samples": len(samples),
        "prompt_id": prompt_id
    })
    return result
