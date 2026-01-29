import json
import re
import asyncio
from typing import Dict, Any, List
from prorag.utils.llm import AsyncLLMEngine
from prorag.utils.prompts import build_filter_system_prompt
import argparse

def normalize_mcts_q(q, n):
    return q / n if n > 0 else 0.0

def format_check(action):
    tags = re.findall(r"</?[a-zA-Z]+>", action)
    valid_patterns = [
        ["<step>", "</step>", "<subquery>", "</subquery>", "<retrieval>", "</retrieval>"],
        ["<step>", "</step>", "<subanswer>", "</subanswer>"],
        ["<step>", "</step>", "<answer>", "</answer>"]
    ]
    return tags in valid_patterns

def build_prompt(job: Dict) -> str:
    candidates_display = ""
    for cand in job['candidates']:
        candidates_display += f"ID: {cand['id']}\nAction: {cand['action']}\n[MCTS]: Q={cand['q']:.2f}, N={cand['n']}\n\n"

    prompt = f"""
### Question:
{job['question']}

### History:
{job['history']}

### Candidates:
{candidates_display}
    """

    return prompt

def parse_result(job: Dict, llm_response: Dict) -> Dict:
    if llm_response.get("has_valid_pair") is True:
        pos_id = llm_response.get('positive_id')
        neg_id = llm_response.get('negative_id')
        
        candidates = job['candidates']
        pos_cand = next((c for c in candidates if c['id'] == pos_id), None)
        neg_cand = next((c for c in candidates if c['id'] == neg_id), None)
        
        if pos_cand and neg_cand:
            return {
                "type": "gpt_selection",
                "input": {
                    "question": job['question'],
                    "history": job['history']
                },
                "chosen": {
                    "new_step": pos_cand['action'],
                    "mcts_stats": {"q": pos_cand['q'], "n": pos_cand['n']}
                },
                "rejected": {
                    "new_step": neg_cand['action'],
                    "mcts_stats": {"q": neg_cand['q'], "n": neg_cand['n']}
                },
                "label": {
                    "gpt_reason": llm_response.get("reason"),
                    "chosen_reward": 1.0,
                    "rejected_reward": 0.0 
                }
            }
    return None

def traverse_and_collect_jobs(node, question, history) -> List[Dict]:
    jobs = []
    children = node.get('children', [])
    if not children: return []

    visited_children = [c for c in children if c.get('n', 0) >= 1 and format_check(c.get('action', ""))]
    
    if len(visited_children) >= 2:
        candidates = []
        for idx, child in enumerate(visited_children):
            candidates.append({
                "id": idx,
                "action": child.get('action'),
                "q": normalize_mcts_q(child.get('q', 0), child.get('n', 0)),
                "n": child.get('n', 0)
            })
        jobs.append({"question": question, "history": history, "candidates": candidates})

    for child in visited_children:
        if child.get('n', 0) >= 3:
            jobs.extend(traverse_and_collect_jobs(child, question, history + child.get('action')))
    return jobs

async def main(args):
    print(f"Parsing trees from {args.input_file}...")
    all_jobs = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            tree_obj = json.loads(line)
            jobs = traverse_and_collect_jobs(tree_obj.get('mcts_tree', {}), tree_obj.get('question', ''), "")
            all_jobs.extend(jobs)
    
    for i, job in enumerate(all_jobs):
        job['id'] = i
        
    print(f"Collected {len(all_jobs)} jobs.")

    engine = AsyncLLMEngine(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        concurrency=args.concurrency
    )

    system_prompt = build_filter_system_prompt()
    await engine.run_batch(
        items=all_jobs,
        prompt_builder=build_prompt,
        system_prompt=system_prompt,
        output_file=args.output_file,
        extract_json=True,
        result_parser=parse_result
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MCTS trees to generate pairwise training data.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file containing MCTS trees")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output training data")

    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (e.g. gpt-4o, deepseek-chat)")
    parser.add_argument("--concurrency", type=int, default=10, help="Async request concurrency")

    parser.add_argument("--api_key", type=str, default=None, help="Optional: OpenAI API Key (or use env var)")
    parser.add_argument("--base_url", type=str, default=None, help="Optional: Custom API Base URL")
    
    args = parser.parse_args()
    
    asyncio.run(main(args))