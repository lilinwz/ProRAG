import re
import torch
from vllm import SamplingParams
from reward import format_reward_step

MAX_INTERACTION_STEPS = 11

def rag_rollout_with_prm(prompts, trainer):
    generation_kwargs = {
        "n": 1,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0 if trainer.min_p is None else trainer.min_p,
        "max_tokens": trainer.max_completion_length,
        "stop": ["</subquery>", "</subanswer>", "</answer>", "<|im_end|>"],
        "include_stop_str_in_output": True,
        "skip_special_tokens": False,
        "logprobs": 0,
    }
    if trainer.args.generation_kwargs is not None:
        generation_kwargs.update(trainer.args.generation_kwargs)
    sampling_params = SamplingParams(**generation_kwargs)

    if trainer.vllm_tensor_parallel_size > 1:
        orig_size = len(prompts)
        gathered_prompts = [None for _ in range(trainer.vllm_tensor_parallel_size)]
        torch.distributed.all_gather_object(gathered_prompts, prompts, group=trainer.tp_group)
        all_prompts = [p for sublist in gathered_prompts for p in sublist]
    else:
        all_prompts = prompts

    if trainer.args.vllm_enable_sleep_mode:
        trainer.llm.wake_up(tags=["kv_cache"])

    total_num_prompts = len(all_prompts)
    current_texts = list(all_prompts)

    accumulated_ids = [[] for _ in range(total_num_prompts)]
    accumulated_lps = [[] for _ in range(total_num_prompts)]
    accumulated_mask = [[] for _ in range(total_num_prompts)] 
    accumulated_prm = [[] for _ in range(total_num_prompts)]
    finished = [False] * total_num_prompts

    for step in range(MAX_INTERACTION_STEPS):
        if all(finished): 
            break
        
        outputs = trainer.llm.generate(current_texts, sampling_params, use_tqdm=False)
        
        # if step == 0:
        #     print(f"\n[DEBUG] SamplingParams logprobs: {sampling_params.logprobs}", flush=True)
        #     first_res = outputs[0].outputs[0]
        #     print(f"[DEBUG] res.logprobs type: {type(first_res.logprobs)}", flush=True)
        #     if first_res.logprobs:
        #         print(f"[DEBUG] First token logprob data: {first_res.logprobs[0]}", flush=True)
        #     else:
        #         print(f"[DEBUG] res.logprobs is None or Empty!", flush=True)

        step_updates = [None] * total_num_prompts
        search_queries = []
        search_idx_map = []
        for i, output in enumerate(outputs):
            if finished[i]: 
                continue
            
            res = output.outputs[0]
            gen_text = res.text
            gen_ids = res.token_ids
            gen_lps = [lp[tid].logprob for tid, lp in zip(gen_ids, res.logprobs)] if res.logprobs else [0.0]*len(gen_ids)

            update_info = {
                "gen_text": gen_text,
                "gen_ids": gen_ids,
                "gen_lps": gen_lps,
                "ret_text": "", 
                "ret_ids": [],
                "is_finished": False
            }

            if "</answer>" in gen_text:
                update_info["is_finished"] = True
            elif "</subquery>" in gen_text:
                match = re.search(r"<subquery>(.*?)</subquery>", gen_text, re.DOTALL)
                if match:
                    query = match.group(1).strip()
                    search_queries.append(query)
                    search_idx_map.append(i)
            
            step_updates[i] = update_info

        if search_queries:
            search_results = trainer.retriever.batch_search(search_queries)
            for idx_in_batch, res_text in enumerate(search_results):
                original_idx = search_idx_map[idx_in_batch]
                formatted_ret = f"\n<retrieval>\n{res_text}\n</retrieval>"
                ret_ids = trainer.processing_class.encode(formatted_ret, add_special_tokens=False)
                
                step_updates[original_idx]["ret_text"] = formatted_ret
                step_updates[original_idx]["ret_ids"] = ret_ids
        
        for i in range(total_num_prompts):
            if finished[i] or step_updates[i] is None:
                continue
            
            info = step_updates[i]
            text_for_scoring = current_texts[i] + info["gen_text"] + info["ret_text"]
            step_score = trainer.get_prm_score(text_for_scoring) + format_reward_step(info["gen_text"]+info["ret_text"])
            
            accumulated_ids[i].extend(info["gen_ids"])
            accumulated_lps[i].extend(info["gen_lps"])
            accumulated_prm[i].extend([step_score] * len(info["gen_ids"]))
            accumulated_mask[i].extend([1] * len(info["gen_ids"])) 
            
            if info["ret_ids"]:
                accumulated_ids[i].extend(info["ret_ids"])
                accumulated_lps[i].extend([0.0] * len(info["ret_ids"]))
                accumulated_prm[i].extend([0.0] * len(info["ret_ids"]))
                accumulated_mask[i].extend([0] * len(info["ret_ids"])) 

            if info["is_finished"]:
                finished[i] = True
                finish_tag = "\n<|im_end|>"
                finish_ids = trainer.processing_class.encode(finish_tag, add_special_tokens=False)
                current_texts[i] = (text_for_scoring + finish_tag)
                accumulated_ids[i].extend(finish_ids)
                accumulated_lps[i].extend([0.0] * len(finish_ids))
                accumulated_prm[i].extend([0.0] * len(finish_ids))
                accumulated_mask[i].extend([0] * len(finish_ids))
            else:
                step_tag = "\n<step>\n"
                step_ids = trainer.processing_class.encode(step_tag, add_special_tokens=False)
                current_texts[i] = (text_for_scoring + step_tag)
                accumulated_ids[i].extend(step_ids)
                accumulated_lps[i].extend([0.0] * len(step_ids))
                accumulated_prm[i].extend([0.0] * len(step_ids))
                accumulated_mask[i].extend([0] * len(step_ids))

    if trainer.vllm_tensor_parallel_size > 1:
        local_rank = torch.distributed.get_rank(group=trainer.tp_group)
        start = local_rank * orig_size
        end = start + orig_size
        completion_ids = accumulated_ids[start:end]
        logprobs = accumulated_lps[start:end]
        prm_scores = accumulated_prm[start:end]
        completion_mask = accumulated_mask[start:end]
    else:
        completion_ids = accumulated_ids
        logprobs = accumulated_lps
        prm_scores = accumulated_prm
        completion_mask = accumulated_mask

    return {
        "prompt_ids": [trainer.processing_class.encode(p) for p in prompts], 
        "completion_ids": completion_ids, # List[List[int]]
        "logprobs": logprobs,             # List[List[float]]
        "rag_mask": completion_mask,      # List[List[int/float]]
        "prm_scores": prm_scores          # List[List[float]]
    }