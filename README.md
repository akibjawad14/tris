# TRIS: Tri-Layer Retrieval Integrity Sieve

Code for **TRIS: A Tri-Layer Retrieval Integrity Sieve Against Knowledge Poisoning**.

TRIS filters retrieved evidence before it reaches the language model. The defense combines cross-embedding-space consistency, structural filtering of trigger-payload artifacts, and an optional LLM consistency check.

This repository builds on [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG). See [README_POISONEDRAG_UPSTREAM.md](README_POISONEDRAG_UPSTREAM.md) for the upstream codebase and [LICENSE](LICENSE) for licensing information.

## How TRIS works

TRIS sits between the retriever and the language model and filters retrieved documents before they are used for generation.

- **Layer 1 — Cross-embedding-space consistency.** Retrieved documents are re-embedded with an independent judge model and clustered in that space. Documents outside the majority cluster are filtered.
- **Layer 2 — Structural filtering.** The document prefix is checked for query overlap and n-gram patterns associated with trigger-payload poisoning. Documents that exceed the configured thresholds are removed.
- **Layer 3 — LLM consistency check.** An optional consistency step checks surviving documents against the language model's own answer and removes conflicting evidence.


## Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate tris
```

Prepare the BEIR datasets:

```bash
python prepare_dataset.py
```

For the original experiment path, copy the GPT configuration template:

```bash
mkdir -p model_configs
cp gpt3.5_config.json.template model_configs/gpt3.5_config.json
```

Add your API key to the copied file locally. Files under `model_configs/*.json` are ignored by Git.

Some of the later experiment scripts read the key from the environment instead:

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

## Experiments

The commands for the paper experiments are in [experiments.md](experiments.md).

That file includes the required frozen poison inputs and hashes, retrieval preparation, end-to-end defense evaluation, retrieval dynamics, layer ablations, injection sweeps, adaptive attacks, Layer 3 experiments, and the TrustRAG and RobustRAG comparisons.

The frozen poison inputs used by these experiments are already included under:

```text
results/adv_targeted_results/
```

Do not regenerate them if the goal is to reproduce the reported experiments. Fresh API generation can produce different poison text.

Generated retrieval files and experiment outputs are not committed.

## Main files

| File | Purpose |
|---|---|
| `main.py` | Original live-retrieval attack and defense driver |
| `run_defense.py` | Original baseline and defense runner |
| `main_adaptive.py` | Adaptive live-retrieval experiments |
| `compute_ir_metrics.py` | Retrieval-dynamics metrics |
| `pilot_cpu.py` | Forced-top adaptive experiments |
| `exp_robustrag_pareto.py` | RobustRAG comparison |
| `exp_l3_knowledge.py` | Layer 3 knowledge-dependency experiment |
| `prepare_dataset.py` | BEIR dataset preparation |
| `evaluate_beir.py` | Contriever retrieval generation |

The repository keeps the original and later experiment implementations separate where they differed. `experiments.md` identifies the code path used for each experiment.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

See [LICENSE](LICENSE).
