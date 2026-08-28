# TRIS Experiments
This document lists the commands to reproduce the experiments in **TRIS: A Tri-Layer Retrieval Integrity Sieve Against Knowledge Poisoning**.
This file lists the experiments in TRIS and the commands used to run them. Run the commands from the repository root after completing the setup steps. Where one step produces a file used by the next, the expected output path is shown explicitly.

## Files and inputs

The repo includes the source code and the frozen poison files you need to reproduce the experiments. It does not include generated model outputs or result JSONs — the commands below recreate those.

The required frozen inputs are:

| File | SHA-256 | Role |
|---|---|---|
| `results/adv_targeted_results/nq.json` | `44df711454a9bada08e72e9e4a003a2cc845c43707ac93a3493e5168ec415cf2` | Original NQ PoisonedRAG inputs |
| `results/adv_targeted_results/hotpotqa.json` | `5119d6f9fd53cb0ecb3f33de939bd940d77cbaf80284b7ccd3d256f63a430537` | Original HotpotQA PoisonedRAG inputs |
| `results/adv_targeted_results/msmarco.json` | `d6bf508ebb0e31e09095995061bd483678e2ae8634f6ba5de1611e4adf9b1987` | Original MSMARCO PoisonedRAG inputs |
| `results/adv_targeted_results/nq_adaptive.json` | `7be351140e01de3497573bbffaef448114bd4ef09f327f4d251aabcc7e256fee` | Adaptive NQ inputs |
| `results/adv_targeted_results/hotpotqa_adaptive.json` | `2ac764d317b15f899f3b5669fcfd8f62357138bd5ebab7cb9673396437e3ca78` | Adaptive HotpotQA inputs |
| `results/adv_targeted_results/nq_adaptive30.json` | `5ff75c7d88946172a03ca4349dfd17154962a995fba73b31dcb399c58b8cf235` | 30-query forced-top adaptive rebuttal input |
| `results/adv_targeted_results/nq_bb20.json` | `e5396d588573b64a604b474ff0e4d7a39d9a43a2a518cb870d242ea316d938d4` | 25-query, 20-poison input used for the 6/10/15/20 injection sweep |

The original NQ, HotpotQA, and MSMARCO poison files come straight from the upstream PoisonedRAG codebase. They are inputs, not outputs from this release.

## Environment

Python 3.10 was used for the release environment.

Using Conda:

```bash
conda env create -f environment.yml
conda activate tris
```

Or with an existing Python 3.10 environment:

```bash
python -m pip install -r requirements.txt
```

Contriever is loaded from `facebook/contriever` by default. To use an already-cached compatible copy without changing the code:

```bash
export TRIS_CONTRIEVER_MODEL=/path/to/facebook-contriever
```

### OpenAI configuration

The original `main.py` / `main_adaptive.py` experiment path reads a local JSON config:

```bash
mkdir -p model_configs
cp gpt3.5_config.json.template model_configs/gpt3.5_config.json
```

Edit the ignored local file `model_configs/gpt3.5_config.json` and replace the placeholder API key. Do not commit this file.

The later CPU/rebuttal utilities (`pilot_cpu.py`, `exp_robustrag_pareto.py`, `gen_adaptive_poisons.py`, and `exp_l3_knowledge.py`) read the key from the environment:

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

The original main-driver configuration uses the model name stored in `gpt3.5_config.json`. The rebuttal utilities explicitly used `gpt-3.5-turbo-0125`. Exact numeric reproduction may depend on continued provider availability and API-side model behavior.

## Download BEIR datasets

```bash
python prepare_dataset.py
```

This downloads NQ, HotpotQA, and MSMARCO into `datasets/`. The code uses the BEIR `train` split for MSMARCO even when `--split test` appears in historical command lines.

## Generate Contriever retrieval files

Create the output directory:

```bash
mkdir -p results/beir_results
```

NQ:

```bash
python evaluate_beir.py --model_code contriever --dataset nq --split test --score_function dot --result_output results/beir_results/nq-contriever.json --gpu_id 0
```

HotpotQA:

```bash
python evaluate_beir.py --model_code contriever --dataset hotpotqa --split test --score_function dot --result_output results/beir_results/hotpotqa-contriever.json --per_gpu_batch_size 32 --gpu_id 0
```

MSMARCO:

```bash
python evaluate_beir.py --model_code contriever --dataset msmarco --split test --score_function dot --result_output results/beir_results/msmarco-contriever.json --per_gpu_batch_size 32 --gpu_id 0
```

The historical HotpotQA retrieval-generation launcher initially wrote a `hotpotqa-contriever-v2.json` filename, while all evaluation code consumed `hotpotqa-contriever.json`. The recovered files had identical top-50 document ordering (only negligible floating-point score differences), so the public command writes the canonical consumed filename directly.

## clean and attacked end-to-end results

All original end-to-end experiments use `run_defense.py`, which invokes `main.py`.

### Clean

NQ:

```bash
python run_defense.py --eval_model_code contriever --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method None --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --skip_defense --eval_dataset nq --query_results_dir exp0_clean_nq --baseline_name clean_nq_k50
```

HotpotQA:

```bash
python run_defense.py --eval_model_code contriever --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method None --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --skip_defense --eval_dataset hotpotqa --query_results_dir exp0_clean_hotpotqa --baseline_name clean_hotpotqa_k50
```

MSMARCO:

```bash
python run_defense.py --eval_model_code contriever --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method None --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --skip_defense --eval_dataset msmarco --query_results_dir exp0_clean_msmarco --baseline_name clean_msmarco_k50
```

### Poisoned + defenses

The first Sieve command for each dataset also produces the no-defense baseline. TrustRAG and RobustRAG use `--skip_baseline` to avoid rerunning that baseline.

NQ, Sieve:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --query_results_dir exp1_k50_nq --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --cluster_count 2 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --jaccard_threshold 0.8 --ngram_threshold 0.8 --defense_method sieve --baseline_name baseline_nq_k50 --defense_name defense_sieve_nq_k50
```

NQ, TrustRAG:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --query_results_dir exp1_k50_nq --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --cluster_count 2 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --jaccard_threshold 0.8 --ngram_threshold 0.8 --defense_method trustrag --skip_baseline --baseline_name baseline_nq_k50 --defense_name defense_trustrag_nq_k50
```

NQ, RobustRAG:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --query_results_dir exp1_k50_nq --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --cluster_count 2 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --jaccard_threshold 0.8 --ngram_threshold 0.8 --defense_method robustrag --skip_baseline --baseline_name baseline_nq_k50 --defense_name defense_robustrag_nq_k50
```

HotpotQA, Sieve:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset hotpotqa --split test --query_results_dir exp1_k50_hotpotqa --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --cluster_count 2 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --jaccard_threshold 0.8 --ngram_threshold 0.8 --defense_method sieve --baseline_name baseline_hotpotqa_k50 --defense_name defense_sieve_hotpotqa_k50
```

HotpotQA, TrustRAG:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset hotpotqa --split test --query_results_dir exp1_k50_hotpotqa --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --cluster_count 2 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --jaccard_threshold 0.8 --ngram_threshold 0.8 --defense_method trustrag --skip_baseline --baseline_name baseline_hotpotqa_k50 --defense_name defense_trustrag_hotpotqa_k50
```

HotpotQA, RobustRAG:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset hotpotqa --split test --query_results_dir exp1_k50_hotpotqa --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --cluster_count 2 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --jaccard_threshold 0.8 --ngram_threshold 0.8 --defense_method robustrag --skip_baseline --baseline_name baseline_hotpotqa_k50 --defense_name defense_robustrag_hotpotqa_k50
```

MSMARCO, Sieve:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset msmarco --split test --query_results_dir exp1_k50_msmarco --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --cluster_count 2 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --jaccard_threshold 0.8 --ngram_threshold 0.8 --defense_method sieve --baseline_name baseline_msmarco_k50 --defense_name defense_sieve_msmarco_k50
```

MSMARCO, TrustRAG:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset msmarco --split test --query_results_dir exp1_k50_msmarco --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --cluster_count 2 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --jaccard_threshold 0.8 --ngram_threshold 0.8 --defense_method trustrag --skip_baseline --baseline_name baseline_msmarco_k50 --defense_name defense_trustrag_msmarco_k50
```

MSMARCO, RobustRAG:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset msmarco --split test --query_results_dir exp1_k50_msmarco --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --cluster_count 2 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --jaccard_threshold 0.8 --ngram_threshold 0.8 --defense_method robustrag --skip_baseline --baseline_name baseline_msmarco_k50 --defense_name defense_robustrag_msmarco_k50
```

The original end-to-end runs did **not** explicitly enable Layer 3.

## retrieval dynamics

After the NQ end-to-end runs above:

```bash
python compute_ir_metrics.py
```

`compute_ir_metrics.py` reads `datasets/nq/qrels/test.tsv`, `results/beir_results/nq-contriever.json`, the frozen NQ poison input, and the NQ Sieve/TrustRAG end-to-end outputs. It rebuilds the poisoned ranking to compute the retrieval metrics; it does not rerun the generator.

## layer ablations

All configurations below use NQ, `k=50`, five poisons/query, MiniLM as the independent judge, and the historical thresholds.

### NQ black-box baseline

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --query_results_dir exp3_k50_nq_baseline --baseline_name baseline_nq_k50 --skip_defense
```

L1 only:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --defense_method sieve --query_results_dir exp3_k50_nq_l1_only --baseline_name baseline_nq_k50 --defense_name defense_l1_only_nq_k50 --jaccard_threshold 10.0 --ngram_threshold 10.0 --cluster_count 2 --skip_baseline
```

L2 only:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --defense_method sieve --query_results_dir exp3_k50_nq_l2_only --baseline_name baseline_nq_k50 --defense_name defense_l2_only_nq_k50 --jaccard_threshold 0.8 --ngram_threshold 0.8 --cluster_count 1 --skip_baseline
```

L3 only:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --defense_method sieve --query_results_dir exp3_k50_nq_l3_only --baseline_name baseline_nq_k50 --defense_name defense_l3_only_nq_k50 --jaccard_threshold 10.0 --ngram_threshold 10.0 --cluster_count 1 --enable_layer3 --skip_baseline
```

L1+L2:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --defense_method sieve --query_results_dir exp3_k50_nq_l1_l2 --baseline_name baseline_nq_k50 --defense_name defense_l1_l2_nq_k50 --jaccard_threshold 0.8 --ngram_threshold 0.8 --cluster_count 2 --skip_baseline
```

L1+L3:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --defense_method sieve --query_results_dir exp3_k50_nq_l1_l3 --baseline_name baseline_nq_k50 --defense_name defense_l1_l3_nq_k50 --jaccard_threshold 10.0 --ngram_threshold 10.0 --cluster_count 2 --enable_layer3 --skip_baseline
```

L2+L3:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --defense_method sieve --query_results_dir exp3_k50_nq_l2_l3 --baseline_name baseline_nq_k50 --defense_name defense_l2_l3_nq_k50 --jaccard_threshold 0.8 --ngram_threshold 0.8 --cluster_count 1 --enable_layer3 --skip_baseline
```

Full L1+L2+L3:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --defense_method sieve --query_results_dir exp3_k50_nq_full --baseline_name baseline_nq_k50 --defense_name defense_full_nq_k50 --jaccard_threshold 0.8 --ngram_threshold 0.8 --cluster_count 2 --enable_layer3 --skip_baseline
```

### NQ HotFlip white-box column

Use the same eight configurations above with `--attack_method hotflip`. The historical white-box launcher used the following baseline and output directory:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method hotflip --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --query_results_dir exp3_k50_nq_hotflip --baseline_name baseline_nq_hotflip_k50 --skip_defense
```

For the seven defended rows, use the same L1/L2/L3 threshold combinations listed above, keep `--attack_method hotflip`, add `--skip_baseline`, use `--query_results_dir exp3_k50_nq_hotflip`, and use the corresponding names:

```text
defense_l1_only_nq_hotflip_k50
defense_l2_only_nq_hotflip_k50
defense_l3_only_nq_hotflip_k50
defense_l1_l2_nq_hotflip_k50
defense_l1_l3_nq_hotflip_k50
defense_l2_l3_nq_hotflip_k50
defense_full_nq_hotflip_k50
```

A later backup rerun of the L1+L3 HotFlip cell used the same Python parameters and wrote to `exp3_k50_nq_hotflip_backup`.

### HotpotQA supplementary ablation

The HotpotQA ablation uses the identical eight configurations with `--eval_dataset hotpotqa`, `--attack_method LM_targeted`, and output directories/names prefixed with `exp3_k50_hotpotqa_` / suffixed with `_hotpotqa_k50`.

## Poison injection sweep, 1–5

The original NQ injection experiment used `k=50`, MiniLM, C=2, thresholds 0.8, and **Layer 3 enabled**. For each `adv_per_query` value 1–5, run the Sieve command first (baseline + Sieve), then TrustRAG and RobustRAG with `--skip_baseline`.

The exact common configuration is:

```text
--eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --cluster_count 2 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --jaccard_threshold 0.8 --ngram_threshold 0.8 --enable_layer3
```

Example for 1 poison/query:

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --cluster_count 2 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --jaccard_threshold 0.8 --ngram_threshold 0.8 --enable_layer3 --adv_per_query 1 --defense_method sieve --query_results_dir exp4_nq_adv1 --baseline_name baseline_nq_adv1_k50 --defense_name defense_sieve_nq_adv1_k50
```

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --cluster_count 2 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --jaccard_threshold 0.8 --ngram_threshold 0.8 --enable_layer3 --adv_per_query 1 --defense_method trustrag --skip_baseline --query_results_dir exp4_nq_adv1 --baseline_name baseline_nq_adv1_k50 --defense_name defense_trustrag_nq_adv1_k50
```

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --score_function dot --repeat_times 10 --M 10 --seed 12 --top_k 50 --cluster_count 2 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --jaccard_threshold 0.8 --ngram_threshold 0.8 --enable_layer3 --adv_per_query 1 --defense_method robustrag --skip_baseline --query_results_dir exp4_nq_adv1 --baseline_name baseline_nq_adv1_k50 --defense_name defense_robustrag_nq_adv1_k50
```

Repeat those three commands with `--adv_per_query 2` and `exp4_nq_adv2`. The final 3/4/5 cells were rerun with the same parameters under `exp4b_nq_adv3`, `exp4b_nq_adv4`, and `exp4b_nq_adv5`.

## Top-k × cluster-count sensitivity

The actual historical sweep used `k ∈ {5,10,20,50}` and `C ∈ {2,3,5}`. Layer 3 was not enabled.

```bash
python run_defense.py --eval_model_code contriever --eval_dataset nq --split test --query_results_dir exp4_heatmap --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --judge_model_name all-MiniLM-L6-v2 --prefix_token_count 20 --jaccard_threshold 0.8 --ngram_threshold 0.8 --baseline_name baseline_heatmap --defense_name defense_heatmap --sweep_topk 5,10,20,50 --sweep_clusters 2,3,5
```

## Adaptive live-retrieval experiments

The frozen `nq_adaptive.json` and `hotpotqa_adaptive.json` files are already included, so regeneration is not required. Their historical generator is retained as `gen_adaptive_poisons_delta.py`.

For each dataset, round 1 evaluated six cells: verbatim/paraphrase × no defense/Sieve/faithful retriever-space TrustRAG. The following are the NQ commands; replace `nq` with `hotpotqa` and the poison filename accordingly for HotpotQA.

Verbatim, no defense:

```bash
python main_adaptive.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --top_k 50 --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --trigger_mode verbatim --poison_file results/adv_targeted_results/nq_adaptive.json --query_results_dir adaptive_nq_verbatim_none --name nq_verbatim_none --summary_out results/rebuttal_gpu/summary_nq_verbatim_none.json
```

Paraphrase, no defense:

```bash
python main_adaptive.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --top_k 50 --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --trigger_mode paraphrase --poison_file results/adv_targeted_results/nq_adaptive.json --query_results_dir adaptive_nq_paraphrase_none --name nq_paraphrase_none --summary_out results/rebuttal_gpu/summary_nq_paraphrase_none.json
```

Verbatim, Sieve:

```bash
python main_adaptive.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --top_k 50 --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --trigger_mode verbatim --poison_file results/adv_targeted_results/nq_adaptive.json --enable_defense --defense_method sieve --query_results_dir adaptive_nq_verbatim_sieve --name nq_verbatim_sieve --summary_out results/rebuttal_gpu/summary_nq_verbatim_sieve.json
```

Paraphrase, Sieve (Layer 3 off):

```bash
python main_adaptive.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --top_k 50 --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --trigger_mode paraphrase --poison_file results/adv_targeted_results/nq_adaptive.json --enable_defense --defense_method sieve --query_results_dir adaptive_nq_paraphrase_sieve --name nq_paraphrase_sieve --summary_out results/rebuttal_gpu/summary_nq_paraphrase_sieve.json
```

Verbatim, faithful retriever-space TrustRAG:

```bash
python main_adaptive.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --top_k 50 --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --trigger_mode verbatim --poison_file results/adv_targeted_results/nq_adaptive.json --enable_defense --defense_method faithful_trustrag --query_results_dir adaptive_nq_verbatim_faithful_trustrag --name nq_verbatim_faithful_trustrag --summary_out results/rebuttal_gpu/summary_nq_verbatim_faithful_trustrag.json
```

Paraphrase, faithful retriever-space TrustRAG:

```bash
python main_adaptive.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --top_k 50 --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --trigger_mode paraphrase --poison_file results/adv_targeted_results/nq_adaptive.json --enable_defense --defense_method faithful_trustrag --query_results_dir adaptive_nq_paraphrase_faithful_trustrag --name nq_paraphrase_faithful_trustrag --summary_out results/rebuttal_gpu/summary_nq_paraphrase_faithful_trustrag.json
```

`src/faithful_trustrag.py` is the retriever-space clustering path used for the independent-judge comparison. You can't get it by setting `--judge_model_name` to Contriever.

## Live-retrieval Layer-3 ablation

This is the controlled paraphrase + Sieve rerun with only Layer 3 toggled on. Compare each command below against the corresponding Layer-3-off paraphrase/Sieve command above.

NQ:

```bash
python main_adaptive.py --eval_model_code contriever --eval_dataset nq --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --top_k 50 --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --trigger_mode paraphrase --poison_file results/adv_targeted_results/nq_adaptive.json --enable_defense --defense_method sieve --enable_layer3 --layer3_mode selective --query_results_dir adaptive_nq_paraphrase_sieve_l3 --name nq_paraphrase_sieve_l3 --summary_out results/rebuttal_gpu/summary_nq_paraphrase_sieve_l3.json
```

HotpotQA:

```bash
python main_adaptive.py --eval_model_code contriever --eval_dataset hotpotqa --split test --model_config_path model_configs/gpt3.5_config.json --model_name gpt3.5 --gpu_id 0 --top_k 50 --adv_per_query 5 --score_function dot --repeat_times 10 --M 10 --seed 12 --trigger_mode paraphrase --poison_file results/adv_targeted_results/hotpotqa_adaptive.json --enable_defense --defense_method sieve --enable_layer3 --layer3_mode selective --query_results_dir adaptive_hotpotqa_paraphrase_sieve_l3 --name hotpotqa_paraphrase_sieve_l3 --summary_out results/rebuttal_gpu/summary_hotpotqa_paraphrase_sieve_l3.json
```

These live-retrieval L3 experiments use the original `src/defense.py` path.

## Fair RobustRAG frontier

These rebuttal experiments use the recovered `exp_robustrag_pareto.py` + `src/robustrag_fair.py` path and `OPENAI_API_KEY`.

NQ:

```bash
python exp_robustrag_pareto.py --dataset nq --num_queries 30 --poison_file results/adv_targeted_results/nq_adaptive30.json --attack_level blackbox --poison_mode prefix --max_docs 30 --alphas 0.1,0.2,0.3,0.5,0.7 --temperature 0.0 --gpt_model gpt-3.5-turbo-0125 --json_out results/rebuttal/robustrag_pareto_nq_repro.json
```

HotpotQA:

```bash
python exp_robustrag_pareto.py --dataset hotpotqa --num_queries 30 --poison_file results/adv_targeted_results/hotpotqa.json --poison_mode prefix --max_docs 30 --alphas 0.1,0.2,0.3,0.5,0.7 --temperature 0.0 --gpt_model gpt-3.5-turbo-0125 --json_out results/rebuttal/robustrag_pareto_hotpotqa.json
```

`src/robustrag_fair.py` is the fair-frontier version written for the rebuttal. It is separate from the older RobustRAG class in the original end-to-end path. It is an approximation used for the rebuttal frontier, not a replacement for the original baseline.

## Fair TrustRAG comparison

These forced-top rebuttal experiments use `pilot_cpu.py`; they are not live-retrieval experiments. `--enable_layer3` enables both TRIS Layer 3 and TrustRAG's LLM check.

HotpotQA companion:

```bash
python pilot_cpu.py --dataset hotpotqa --num_queries 30 --top_k 50 --poison_file results/adv_targeted_results/hotpotqa.json --defenses none,sieve,trustrag --enable_layer3 --temperature 0.0 --gpt_model gpt-3.5-turbo-0125 --json_out results/rebuttal/hotpotqa_bb_fair.json
```

NQ trigger-evasion:

```bash
python pilot_cpu.py --dataset nq --num_queries 30 --top_k 50 --poison_file results/adv_targeted_results/nq_adaptive30.json --defenses trustrag,sieve --attack_level trigger_evasion --enable_layer3 --temperature 0.0 --gpt_model gpt-3.5-turbo-0125 --json_out results/rebuttal/nq_trustrag_fair_trigger_evasion.json
```

NQ full-adaptive:

```bash
python pilot_cpu.py --dataset nq --num_queries 30 --top_k 50 --poison_file results/adv_targeted_results/nq_adaptive30.json --defenses trustrag,sieve --attack_level full_adaptive --enable_layer3 --temperature 0.0 --gpt_model gpt-3.5-turbo-0125 --json_out results/rebuttal/nq_trustrag_fair_full_adaptive.json
```

## Extended injection sweep, 6–20

Do **not** use the older `run_defense.py` 1–20 sweep for the camera-ready 6–20 result. That older path only had five frozen static poisons/query.

The camera-ready extension used the recovered 25-query file `nq_bb20.json`, which contains up to 20 genuine poison passages/query, together with `pilot_cpu.py`. Layer 3 was **off** in these four runs.

6 poisons/query:

```bash
python pilot_cpu.py --dataset nq --num_queries 25 --top_k 50 --adv_per_query 6 --defenses none,sieve --poison_file results/adv_targeted_results/nq_bb20.json --attack_level blackbox --poison_mode prefix --temperature 0.0 --gpt_model gpt-3.5-turbo-0125 --json_out results/rebuttal/inject_6.json
```

10 poisons/query:

```bash
python pilot_cpu.py --dataset nq --num_queries 25 --top_k 50 --adv_per_query 10 --defenses none,sieve --poison_file results/adv_targeted_results/nq_bb20.json --attack_level blackbox --poison_mode prefix --temperature 0.0 --gpt_model gpt-3.5-turbo-0125 --json_out results/rebuttal/inject_10.json
```

15 poisons/query:

```bash
python pilot_cpu.py --dataset nq --num_queries 25 --top_k 50 --adv_per_query 15 --defenses none,sieve --poison_file results/adv_targeted_results/nq_bb20.json --attack_level blackbox --poison_mode prefix --temperature 0.0 --gpt_model gpt-3.5-turbo-0125 --json_out results/rebuttal/inject_15.json
```

20 poisons/query:

```bash
python pilot_cpu.py --dataset nq --num_queries 25 --top_k 50 --adv_per_query 20 --defenses none,sieve --poison_file results/adv_targeted_results/nq_bb20.json --attack_level blackbox --poison_mode prefix --temperature 0.0 --gpt_model gpt-3.5-turbo-0125 --json_out results/rebuttal/inject_20.json
```

The frozen `nq_bb20.json` is enough to reproduce these four runs. `gen_adaptive_poisons.py` is kept as the generator that made it, but we don't have the exact original command it was run with.

## Layer-3 knowledge-dependency / fail-open test

The camera-ready post-cutoff knowledge-dependency experiment is recovered as:

```bash
python exp_l3_knowledge.py --json_out results/rebuttal/l3_knowledge.json
```

This experiment imports `TriLayerSieve` from `src/defense_failopen.py` on purpose. That file is the later fail-open version of Layer 3. It's kept separate so the original `src/defense.py` (used by the paper and the live-retrieval runs) isn't quietly replaced.

The test uses ten post-cutoff or niche questions and measures Layer-3 abstention, poison removal, and clean-document removal.

## Optional adaptive-poison regeneration

The frozen adaptive inputs are already included and should be used to reproduce the numbers. If you want to see how they were made, the round-1 adaptive generator supports:

```bash
python gen_adaptive_poisons_delta.py --dataset nq
```

```bash
python gen_adaptive_poisons_delta.py --dataset hotpotqa
```

Regenerating poison text through a live API can produce different text from the frozen historical artifact and therefore is **not** the recommended path for reproducing reported numbers.

## Output locations

Generated outputs are ignored by Git.

- Main experiments: `results/query_results/...`
- Adaptive summaries: `results/rebuttal_gpu/...`
- CPU/rebuttal JSONs: `results/rebuttal/...`
- Retrieval rankings: `results/beir_results/...`

Only `results/adv_targeted_results/*.json` is treated as release input data.

## Why some files have more than one version

Some of these files were written at different stages of the paper, so the release keeps them separate:

- `src/defense.py`: the original defense, used for the submission and the live-retrieval runs.
- `src/defense_failopen.py`: the later fail-open version of Layer 3, used only by `exp_l3_knowledge.py`.
- `src/baselines.py`: the original end-to-end baselines.
- `src/faithful_trustrag.py`: the retriever-space clustering path used for the independent-judge comparison.
- `src/robustrag_fair.py`: the later fair-comparison version of RobustRAG.
- `pilot_cpu.py`: a forced-top test harness. It does not rank poisons through a live Contriever, so don't read it as live retrieval.

Keeping them separate means we don't have to merge the experiments into one tidier version that was never actually run.

## Reproducibility limits

The poison inputs, code, parameters, and retrieval pipeline are all fixed. Even so, the exact answers can vary because the experiments call a hosted model API. Model retirement, model updates, sampling, quotas, and one-off API errors can shift the exact ASR/CleanAcc numbers even when everything else is reproduced correctly.

One thing to watch: the GPT client returns an empty string when an API call fails. An empty answer is scored as a failed attack, so a run that hit quota or rate limits can look like a perfect defense. Check the logs for API errors instead of trusting empty answers as real results.
