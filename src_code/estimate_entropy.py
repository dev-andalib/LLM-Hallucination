# src/estimate_entropy.py
import argparse
import yaml
import json
from pathlib import Path
from bayesian_estimator.estimator import BayesianEntropyEstimator
from adaptive_sampler import adaptive_sampling_loop

def load_generations_and_meanings(gen_file: Path, meanings_file: Path):
    """
    Minimal loader that builds the sample list expected by estimator:
    [{'text':..., 'meaning_id': int, 'prob': float}, ...]
    It matches responses to meaning IDs using the meanings mapping.
    """
    gen = json.loads(gen_file.read_text())
    meanings = json.loads(meanings_file.read_text())

    # Build mapping: response text -> meaning_id (simple exact match; replace with entailment logic for fuzziness)
    text_to_mid = {}
    for c in meanings.get("clusters", []):
        mid = c["meaning_id"]
        for member in c["members"]:
            text_to_mid[member.strip()] = mid

    samples = []
    for r in gen.get("responses", []):
        text = r["text"].strip()
        mid = text_to_mid.get(text, 0)  # default meaning 0 if not found
        samples.append({"text": text, "meaning_id": mid, "prob": float(r.get("prob", 0.0))})
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml")
    parser.add_argument("--prompt_id", type=str, required=True)
    parser.add_argument("--gen_file", type=str, required=True)
    parser.add_argument("--meanings_file", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    estimator_cfg = cfg.get("estimator", {})
    prior_path = estimator_cfg.get("prior_K_file", "configs/prior_K.json")
    alpha_prior = estimator_cfg.get("alpha_prior", 0.5)
    num_mc = estimator_cfg.get("num_mc_samples", 500)
    use_trunc = estimator_cfg.get("use_truncated", True)

    est = BayesianEntropyEstimator(prior_K_path=prior_path,
                                   alpha_prior=alpha_prior,
                                   num_mc_samples=num_mc,
                                   use_truncation=use_trunc)

    samples = load_generations_and_meanings(Path(args.gen_file), Path(args.meanings_file))

    # If adaptive mode desired, you would provide sampler_fn that queries LLM and maps to meaning_id on the fly.
    # For now we just run once with currently available samples:
    result = est.estimate(samples)
    print("Result:", result)

if __name__ == "__main__":
    main()
