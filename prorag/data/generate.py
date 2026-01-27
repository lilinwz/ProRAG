import sys
import os
import argparse
import asyncio
import json
from prorag.utils.llm import AsyncLLMEngine
from prorag.utils.prompts import (
    build_clean_system_prompt, 
    build_clean_user_prompt,
    build_cot_system_prompt, 
    build_cot_user_prompt
)

def prompt_clean_adapter(item):
    query = item.get("query")
    chain = item.get("chain_of_thought")
    answer = item.get("answer")
    
    if not query or not chain or answer is None:
        return None
        
    return build_clean_user_prompt(query, chain, answer)

def prompt_cot_adapter(item):
    query = item.get("query")
    answer = item.get("answer")

    new_cot = None
    if "response_data" in item and isinstance(item["response_data"], dict):
        new_cot = item["response_data"].get("new_chain_of_thought")
    
    if not query or not new_cot or not answer:
        return None
        
    return build_cot_user_prompt(query, new_cot, answer)

async def main(args):
    with open(args.input_file, "r") as f:
        raw_data = [json.loads(line) for line in f]
    
    if args.task == "clean":
        user_prompt = prompt_clean_adapter
        system_prompt = build_clean_system_prompt()
    else:
        user_prompt = prompt_cot_adapter
        system_prompt = build_cot_system_prompt()

    engine = AsyncLLMEngine(
        deployment_name="gpt-4o_2024-11-20",
        endpoint = f"https://trapi.research.microsoft.com/gcr/shared",
        concurrency=args.concurrency,
        use_azure_identity=True
    )

    print("🚀 Starting LLM generation...")
    await engine.run_batch(
        items=raw_data,
        prompt_builder=user_prompt,
        system_prompt=system_prompt,
        output_file=args.output_file,
        extract_json=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(main(args))