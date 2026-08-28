import argparse
import os
import json
import time
from tqdm import tqdm
import random
import numpy as np
from src.models import create_model
from src.utils import load_beir_datasets, load_models
from src.utils import save_results, load_json, setup_seeds, clean_str, f1_score
from src.attack import Attacker
from src.defense import TriLayerSieve
from src.baselines import TrustRAGDefense, RobustRAGDefense
from src.prompts import wrap_prompt
import torch



def parse_args():
    parser = argparse.ArgumentParser(description='test')

    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument('--eval_dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--query_results_dir", type=str, default='main')

    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='palm2')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--use_truth', type=str, default='False')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--enable_defense', action='store_true', help='Enable Tri-Layer Sieve phase-1 defense.')
    parser.add_argument('--judge_model_name', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--cluster_count', type=int, default=2)
    parser.add_argument('--prefix_token_count', type=int, default=20)
    parser.add_argument('--jaccard_threshold', type=float, default=0.8)
    parser.add_argument('--ngram_threshold', type=float, default=0.8)
    parser.add_argument('--enable_layer3', action='store_true', help='Enable Layer-3 LLM consistency verification.')
    parser.add_argument('--layer3_mode', type=str, default='selective', choices=['selective', 'always'])
    parser.add_argument('--defense_method', type=str, default='sieve',
                        choices=['sieve', 'trustrag', 'robustrag'],
                        help='Which defense to apply: sieve (Tri-Layer), trustrag, robustrag.')

    # attack
    parser.add_argument('--attack_method', type=str, default='LM_targeted')
    parser.add_argument('--adv_per_query', type=int, default=5, help='The number of adv texts for each target query.')
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    parser.add_argument('--repeat_times', type=int, default=10, help='repeat several times to compute average')
    parser.add_argument('--M', type=int, default=10, help='one of our parameters, the number of target queries')
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'
    setup_seeds(args.seed)
    if args.model_config_path == None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'

    # load target queries and answers
    if args.eval_dataset == 'msmarco':
        corpus, queries, qrels = load_beir_datasets("msmarco", "train")
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)

    incorrect_answers = load_json(f'results/adv_targeted_results/{args.eval_dataset}.json')
    incorrect_answers = list(incorrect_answers.values())

    orig_beir_path = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}.json"
    if args.score_function == 'cos_sim':
        orig_beir_path = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-cos.json"
    with open(orig_beir_path, 'r') as f:
        results = json.load(f)
    # assert len(qrels) <= len(results)
    print('Total samples:', len(results))

    if args.use_truth == 'True':
        args.attack_method = None

    if args.attack_method not in [None, 'None']:
        # Load retrieval models
        model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
        model.eval()
        model.to(device)
        c_model.eval()
        c_model.to(device) 
        attacker = Attacker(args,
                            model=model,
                            c_model=c_model,
                            tokenizer=tokenizer,
                            get_emb=get_emb) 
    
    llm = create_model(args.model_config_path)
    defense = None
    if args.enable_defense and args.use_truth != 'True':
        verifier_fn = lambda prompt: llm.query(prompt)
        if args.defense_method == 'sieve':
            if not args.enable_layer3:
                verifier_fn = None
            defense = TriLayerSieve(
                judge_model_name=args.judge_model_name,
                cluster_count=args.cluster_count,
                prefix_token_count=args.prefix_token_count,
                jaccard_threshold=args.jaccard_threshold,
                ngram_threshold=args.ngram_threshold,
                enable_layer3=args.enable_layer3,
                layer3_mode=args.layer3_mode,
                verifier_fn=verifier_fn,
            )
        elif args.defense_method == 'trustrag':
            defense = TrustRAGDefense(
                judge_model_name=args.judge_model_name,
                cluster_count=args.cluster_count,
                rouge_threshold=0.3,
                enable_llm_check=args.enable_layer3,
                verifier_fn=verifier_fn,
            )
        elif args.defense_method == 'robustrag':
            defense = RobustRAGDefense(llm=llm)

    all_results = []
    asr_list = []
    correct_acc_list = []
    defense_latency_ms = []
    ret_list = []
    poison_before_counts = []
    poison_after_counts = []
    l1_removed_counts = []
    l2_removed_counts = []
    l3_removed_counts = []
    fallback_flags = []

    for iter in range(args.repeat_times):
        print(f'######################## Iter: {iter+1}/{args.repeat_times} #######################')

        target_queries_idx = range(iter * args.M, iter * args.M + args.M)

        target_queries = [incorrect_answers[idx]['question'] for idx in target_queries_idx]
        
        if args.attack_method not in [None, 'None']:
            for i in target_queries_idx:
                top1_idx = list(results[incorrect_answers[i]['id']].keys())[0]
                top1_score = results[incorrect_answers[i]['id']][top1_idx]
                target_queries[i - iter * args.M] = {'query': target_queries[i - iter * args.M], 'top1_score': top1_score, 'id': incorrect_answers[i]['id']}
                
            adv_text_groups = attacker.get_attack(target_queries)
            adv_text_list = sum(adv_text_groups, []) # convert 2D array to 1D array

            adv_input = tokenizer(adv_text_list, padding=True, truncation=True, return_tensors="pt")
            adv_input = {key: value.cuda() for key, value in adv_input.items()}
            with torch.no_grad():
                adv_embs = get_emb(c_model, adv_input)        
                      
        asr_cnt = 0
        correct_cnt = 0
        ret_sublist = []

        iter_results = []
        for i in target_queries_idx:
            iter_idx = i - iter * args.M # iter index
            print(f'############# Target Question: {iter_idx+1}/{args.M} #############')
            question = incorrect_answers[i]['question']
            print(f'Question: {question}\n') 
            
            gt_ids = list(qrels[incorrect_answers[i]['id']].keys())
            ground_truth = [corpus[id]["text"] for id in gt_ids]
            incco_ans = incorrect_answers[i]['incorrect answer']            

            if args.use_truth == 'True':
                query_prompt = wrap_prompt(question, ground_truth, 4)
                response = llm.query(query_prompt)
                print(f"Output: {response}\n\n")
                iter_results.append(
                    {
                        "question": question,
                        "input_prompt": query_prompt,
                        "output": response,
                    }
                )  

            else: # topk
                topk_idx = list(results[incorrect_answers[i]['id']].keys())[:args.top_k]
                topk_results = [{'score': results[incorrect_answers[i]['id']][idx], 'context': corpus[idx]['text']} for idx in topk_idx]               
                adv_text_set = set()
                defense_info = None

                if args.attack_method not in [None, 'None']: 
                    query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
                    query_input = {key: value.cuda() for key, value in query_input.items()}
                    with torch.no_grad():
                        query_emb = get_emb(model, query_input) 
                    for j in range(len(adv_text_list)):
                        adv_emb = adv_embs[j, :].unsqueeze(0) 
                        # similarity     
                        if args.score_function == 'dot':
                            adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
                        elif args.score_function == 'cos_sim':
                            adv_sim = torch.cosine_similarity(adv_emb, query_emb).cpu().item()
                                               
                        topk_results.append({'score': adv_sim, 'context': adv_text_list[j]})
                    
                    adv_text_set = set(adv_text_groups[iter_idx])
                topk_results = sorted(topk_results, key=lambda x: float(x['score']), reverse=True)
                pre_defense_topk_results = topk_results[:args.top_k]
                pre_defense_topk_contents = [item["context"] for item in pre_defense_topk_results]

                poison_pre_count = sum([ctx in adv_text_set for ctx in pre_defense_topk_contents])
                poison_pre_ranks = [idx + 1 for idx, ctx in enumerate(pre_defense_topk_contents) if ctx in adv_text_set]

                topk_results = pre_defense_topk_results
                if defense is not None:
                    _t0 = time.perf_counter()
                    topk_results, defense_info = defense.filter_documents(question, topk_results)
                    defense_latency_ms.append((time.perf_counter() - _t0) * 1000)
                topk_contents = [item["context"] for item in topk_results]
                poison_post_count = sum([ctx in adv_text_set for ctx in topk_contents])
                poison_post_ranks = [idx + 1 for idx, ctx in enumerate(topk_contents) if ctx in adv_text_set]
                # tracking the num of adv_text in topk
                if args.attack_method not in [None, 'None']:
                    ret_sublist.append(poison_post_count)
                    poison_before_counts.append(poison_pre_count)
                    poison_after_counts.append(poison_post_count)
                    if isinstance(defense_info, dict):
                        l1_removed_counts.append(len(defense_info.get("layer1_removed_idx", [])))
                        l2_removed_counts.append(len(defense_info.get("layer2_removed_idx", [])))
                        l3_removed_counts.append(len(defense_info.get("layer3_removed_idx", [])))
                        fallback_flags.append(1 if defense_info.get("fallback_to_original", False) else 0)
                query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)

                if isinstance(defense, RobustRAGDefense):
                    response = defense.get_answer(
                        question, topk_results,
                        wrap_prompt_fn=lambda q, docs: wrap_prompt(q, docs, prompt_id=4),
                    )
                else:
                    response = llm.query(query_prompt)

                print(f'Output: {response}\n\n')
                correct_ans = incorrect_answers[i].get('correct answer', '')
                if correct_ans and clean_str(correct_ans) in clean_str(response):
                    correct_cnt += 1
                injected_adv = [ctx for ctx in topk_contents if ctx in adv_text_set]
                iter_results.append(
                    {
                        "id":incorrect_answers[i]['id'],
                        "question": question,
                        "injected_adv": injected_adv,
                        "retrieval_stats": {
                            "poison_count_pre_defense_topk": poison_pre_count,
                            "poison_ranks_pre_defense_topk": poison_pre_ranks,
                            "poison_count_post_defense": poison_post_count,
                            "poison_ranks_post_defense": poison_post_ranks,
                            "topk_size_post_defense": len(topk_contents),
                        },
                        "defense_info": defense_info,
                        "input_prompt": query_prompt,
                        "output_poison": response,
                        "incorrect_answer": incco_ans,
                        "answer": incorrect_answers[i]['correct answer']
                    }
                )

                if clean_str(incco_ans) in clean_str(response):
                    asr_cnt += 1  

        asr_list.append(asr_cnt)
        correct_acc_list.append(correct_cnt)
        ret_list.append(ret_sublist)

        all_results.append({f'iter_{iter}': iter_results})
        save_results(all_results, args.query_results_dir, args.name)
        print(f'Saving iter results to results/query_results/{args.query_results_dir}/{args.name}.json')


    asr = np.array(asr_list) / args.M
    asr_mean = round(np.mean(asr), 2)
    if args.attack_method not in [None, 'None'] and len(ret_list) > 0 and len(ret_list[0]) > 0:
        ret_precision_array = np.array(ret_list) / args.top_k
        ret_precision_mean = round(np.mean(ret_precision_array), 2)
        ret_recall_array = np.array(ret_list) / args.adv_per_query
        ret_recall_mean = round(np.mean(ret_recall_array), 2)
        ret_f1_array = f1_score(ret_precision_array, ret_recall_array)
        ret_f1_mean = round(np.mean(ret_f1_array), 2)
    else:
        ret_precision_mean = 0.0
        ret_recall_mean = 0.0
        ret_f1_mean = 0.0
  
    print(f"ASR: {asr}")
    print(f"ASR Mean: {asr_mean}\n") 

    print(f"Ret: {ret_list}")
    print(f"Precision mean: {ret_precision_mean}")
    print(f"Recall mean: {ret_recall_mean}")
    print(f"F1 mean: {ret_f1_mean}\n")

    if args.attack_method not in [None, 'None'] and len(poison_before_counts) > 0:
        poison_before_mean = round(float(np.mean(poison_before_counts)), 2)
        poison_after_mean = round(float(np.mean(poison_after_counts)), 2)
        print(f"Poison docs in top-k (pre-defense mean): {poison_before_mean}")
        print(f"Poison docs in top-k (post-defense mean): {poison_after_mean}")
    if len(l1_removed_counts) > 0:
        print(f"Layer1 removed/query mean: {round(float(np.mean(l1_removed_counts)), 2)}")
        print(f"Layer2 removed/query mean: {round(float(np.mean(l2_removed_counts)), 2)}")
        print(f"Layer3 removed/query mean: {round(float(np.mean(l3_removed_counts)), 2)}")
        print(f"Fallback rate: {round(float(np.mean(fallback_flags)), 2)}")

    correct_acc = np.array(correct_acc_list) / args.M
    correct_acc_mean = round(float(np.mean(correct_acc)), 2)
    print(f"Correct Acc: {correct_acc}")
    print(f"Correct Acc Mean: {correct_acc_mean}\n")

    if defense_latency_ms:
        print(f"Avg defense latency: {round(float(np.mean(defense_latency_ms)), 1)} ms/query")
        print(f"Total defense latency samples: {len(defense_latency_ms)}")

    print(f"Ending...")


if __name__ == '__main__':
    main()