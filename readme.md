| 📄 / 📁                                                                                       | Description                                                                        |
| :-------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- |
| **📄 README.md**                                                                              | Main project documentation — explains setup, usage, and experiments.               |
| **📄 requirements.txt**                                                                       | Python dependencies required for the project.                                      |
| **📄 setup.py**                                                                               | Optional packaging file for `pip install -e .` (for local development).            |
|                                                                                               |                                                                                    |
| **📁 configs/**                                                                               | Centralized configuration files for models, datasets, estimators, and experiments. |
| ├── **base_config.yaml** — global defaults (device, logging, and sampling options).           |                                                                                    |
| ├── **model/** — defines LLM and entailment model settings (e.g., Llama-2, Mistral).          |                                                                                    |
| ├── **dataset/** — dataset-specific configs (TriviaQA, SQuAD, SVAMP, etc.).                   |                                                                                    |
| ├── **estimator/** — parameters for Bayesian, Histogram, and Rescaled estimators.             |                                                                                    |
| └── **experiments/** — pre-defined setups for fixed/adaptive budget experiments.              |                                                                                    |
|                                                                                               |                                                                                    |
| **📁 data/**                                                                                  | Stores all data artifacts (prompts, generations, meaning mappings, and results).   |
| ├── **prompts/** — input prompts for the LLM.                                                 |                                                                                    |
| ├── **generations/** — generated sequences and their probabilities *p(s‖x)*.                  |                                                                                    |
| ├── **meanings/** — semantic class mappings obtained via entailment models.                   |                                                                                    |
| └── **results/** — saved semantic entropy values, AUROC scores, and plots.                    |                                                                                    |
|                                                                                               |                                                                                    |
| **📁 src/**                                                                                   | Core implementation source code.                                                   |
| ├── **llm_interface.py** — handles LLM sampling and probability extraction.                   |                                                                                    |
| ├── **meaning_mapper.py** — clusters sequences into semantic meanings.                        |                                                                                    |
| ├── **data_utils.py** — helper functions for loading/saving datasets.                         |                                                                                    |
| ├── **train_prior.py** — learns prior distribution over meaning counts *K*.                   |                                                                                    |
| ├── **estimate_entropy.py** — main entry point for computing Bayesian semantic entropy.       |                                                                                    |
| ├── **adaptive_sampler.py** — dynamically allocates samples until target variance is reached. |                                                                                    |
| │                                                                                             |                                                                                    |
| ├── **bayesian_estimator/** — core Bayesian entropy estimation logic.                         |                                                                                    |
| │   ├── **dirichlet_entropy.py** — analytical Dirichlet expectation formulas.                 |                                                                                    |
| │   ├── **truncated_dirichlet.py** — Monte Carlo integration with constraints.                |                                                                                    |
| │   ├── **hierarchical_model.py** — Bayesian handling of unknown *K* values.                  |                                                                                    |
| │   ├── **estimator.py** — integrates all modules into a single estimator class.              |                                                                                    |
| │   └── **utils.py** — mathematical helpers for entropy and sampling.                         |                                                                                    |
| │                                                                                             |                                                                                    |
| └── **evaluation/** — scripts for assessing estimator performance.                            |                                                                                    |
|     ├── **metrics.py** — AUROC, F1, and statistical measures.                                 |                                                                                    |
|     ├── **compare_baselines.py** — compares Bayesian estimator vs. baselines.                 |                                                                                    |
|     └── **visualize_results.py** — creates plots for performance comparison.                  |                                                                                    |
|                                                                                               |                                                                                    |
| **📁 experiments/**                                                                           | Reproducible scripts and notebooks for replicating paper results.                  |
| ├── **run_fixed_budget.py** — runs fixed-sample (N) experiments.                              |                                                                                    |
| ├── **run_adaptive_budget.py** — runs adaptive-sampling experiments.                          |                                                                                    |
| └── **analyze_results.ipynb** — Jupyter notebook for analysis and visualization.              |                                                                                    |
|                                                                                               |                                                                                    |
| **📁 logs/**                                                                                  | Stores runtime and debugging logs.                                                 |
|                                                                                               |                                                                                    |
| **📁 tests/**                                                                                 | Unit and integration tests ensuring correctness and reproducibility.               |
| ├── **test_dirichlet_entropy.py**                                                             |                                                                                    |
| ├── **test_truncated_sampling.py**                                                            |                                                                                    |
| └── **test_end_to_end.py**                                                                    |                                                                                    |
