import argparse
import os
import shlex


def build_main_cmd(args, enable_defense, run_name):
    def q(val):
        return shlex.quote(str(val))

    cmd = [
        "python",
        "main.py",
        f"--eval_model_code {q(args.eval_model_code)}",
        f"--eval_dataset {q(args.eval_dataset)}",
        f"--split {q(args.split)}",
        f"--query_results_dir {q(args.query_results_dir)}",
        f"--model_name {q(args.model_name)}",
        f"--top_k {args.top_k}",
        f"--gpu_id {args.gpu_id}",
        f"--attack_method {q(args.attack_method)}",
        f"--adv_per_query {args.adv_per_query}",
        f"--score_function {q(args.score_function)}",
        f"--repeat_times {args.repeat_times}",
        f"--M {args.M}",
        f"--seed {args.seed}",
        f"--name {q(run_name)}",
    ]
    if args.model_config_path:
        cmd.append(f"--model_config_path {q(args.model_config_path)}")
    if enable_defense:
        cmd.extend(
            [
                "--enable_defense",
                f"--defense_method {q(args.defense_method)}",
                f"--judge_model_name {q(args.judge_model_name)}",
                f"--cluster_count {args.cluster_count}",
                f"--prefix_token_count {args.prefix_token_count}",
                f"--jaccard_threshold {args.jaccard_threshold}",
                f"--ngram_threshold {args.ngram_threshold}",
            ]
        )
        if args.enable_layer3:
            cmd.append("--enable_layer3")
            cmd.append(f"--layer3_mode {q(args.layer3_mode)}")
    return " ".join(cmd)


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline and defense experiments back-to-back.")

    # passthrough args to main.py
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument("--eval_dataset", type=str, default="nq")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--query_results_dir", type=str, default="phase2")
    parser.add_argument("--model_config_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="palm2")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--attack_method", type=str, default="LM_targeted")
    parser.add_argument("--adv_per_query", type=int, default=5)
    parser.add_argument("--score_function", type=str, default="dot", choices=["dot", "cos_sim"])
    parser.add_argument("--repeat_times", type=int, default=10)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12)

    # defense args
    parser.add_argument("--judge_model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--cluster_count", type=int, default=2)
    parser.add_argument("--prefix_token_count", type=int, default=20)
    parser.add_argument("--jaccard_threshold", type=float, default=0.8)
    parser.add_argument("--ngram_threshold", type=float, default=0.8)
    parser.add_argument("--enable_layer3", action="store_true")
    parser.add_argument("--layer3_mode", type=str, default="selective", choices=["selective", "always"])
    parser.add_argument("--defense_method", type=str, default="sieve",
                        choices=["sieve", "trustrag", "robustrag"])

    # runner behavior
    parser.add_argument("--baseline_name", type=str, default="no_defense")
    parser.add_argument("--defense_name", type=str, default="with_defense")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_defense", action="store_true")
    parser.add_argument("--make_summary", action="store_true", help="Run summarize_defense_results.py after runs.")
    parser.add_argument(
        "--summary_output_json",
        type=str,
        default=None,
        help="Optional output path for summary JSON; defaults under query_results_dir.",
    )
    parser.add_argument(
        "--sweep_topk",
        type=str,
        default=None,
        help="Comma-separated top-k sweep (e.g., 20,50,100).",
    )
    parser.add_argument(
        "--sweep_clusters",
        type=str,
        default=None,
        help="Comma-separated cluster-count sweep for defense runs (e.g., 2,3,5).",
    )
    return parser.parse_args()


def run_once(args):
    baseline_path = f"results/query_results/{args.query_results_dir}/{args.baseline_name}.json"
    defense_path = f"results/query_results/{args.query_results_dir}/{args.defense_name}.json"
    summary_path = (
        args.summary_output_json
        if args.summary_output_json
        else f"results/query_results/{args.query_results_dir}/defense_summary.json"
    )

    if not args.skip_baseline:
        baseline_cmd = build_main_cmd(args, enable_defense=False, run_name=args.baseline_name)
        print(f"[RUN] Baseline command:\n{baseline_cmd}\n")
        baseline_rc = os.system(baseline_cmd)
        if baseline_rc != 0:
            raise SystemExit(f"Baseline run failed with exit code {baseline_rc}")
    else:
        print("[SKIP] Baseline run skipped.")

    if not args.skip_defense:
        defense_cmd = build_main_cmd(args, enable_defense=True, run_name=args.defense_name)
        print(f"[RUN] Defense command:\n{defense_cmd}\n")
        defense_rc = os.system(defense_cmd)
        if defense_rc != 0:
            raise SystemExit(f"Defense run failed with exit code {defense_rc}")
    else:
        print("[SKIP] Defense run skipped.")

    if args.make_summary:
        summary_cmd = (
            f"python summarize_defense_results.py "
            f"--defense_json {defense_path} "
            f"--output_json {summary_path}"
        )
        if not args.skip_baseline:
            summary_cmd += f" --baseline_json {baseline_path}"
        print(f"[RUN] Summary command:\n{summary_cmd}\n")
        summary_rc = os.system(summary_cmd)
        if summary_rc != 0:
            raise SystemExit(f"Summary generation failed with exit code {summary_rc}")
        print(f"[DONE] Summary written to {summary_path}")
    else:
        print("[INFO] Summary generation skipped. Use summarize_defense_results.py manually if needed.")


def parse_csv_ints(val):
    if not val:
        return []
    return [int(x.strip()) for x in val.split(",") if x.strip()]


def main():
    args = parse_args()
    topk_values = parse_csv_ints(args.sweep_topk)
    cluster_values = parse_csv_ints(args.sweep_clusters)

    if not topk_values and not cluster_values:
        run_once(args)
        return

    if not topk_values:
        topk_values = [args.top_k]
    if not cluster_values:
        cluster_values = [args.cluster_count]

    base_dir = args.query_results_dir
    base_baseline = args.baseline_name
    base_defense = args.defense_name
    base_summary = args.summary_output_json

    for k in topk_values:
        for c in cluster_values:
            args.top_k = k
            args.cluster_count = c
            args.query_results_dir = f"{base_dir}_k{k}_c{c}"
            args.baseline_name = f"{base_baseline}_k{k}_c{c}"
            args.defense_name = f"{base_defense}_k{k}_c{c}"
            args.summary_output_json = (
                base_summary if base_summary else f"results/query_results/{args.query_results_dir}/defense_summary.json"
            )
            print(f"[SWEEP] Running config: top_k={k}, cluster_count={c}, dir={args.query_results_dir}")
            run_once(args)


if __name__ == "__main__":
    main()
