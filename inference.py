import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
from prorag.utils.prompts import build_user_prompt
from prorag.utils.retriever import RemoteRetriever

QUESTION = "Who was president of the United States in the year that Citibank was founded?"
MODEL_PATH = "bmbgsj/ProRAG"
MAX_TOKENS = 4096
MAX_HOP = 13

def main():
    print(f"Connecting to Retrieval Server")
    retriever = RemoteRetriever()

    print(f"Loading vLLM Model: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,  
        gpu_memory_utilization=0.80,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
        stop=["</subquery>", "</subanswer>", "</answer>", "<|im_end|>"],
        include_stop_str_in_output=True,
        skip_special_tokens=False
    )

    prompt = f"<|im_start|>user\n{build_user_prompt(QUESTION)}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n<step>\n"
    final_answer = ""
    for hop in range(MAX_HOP):
        outputs = llm.generate(prompt, sampling_params, use_tqdm=True)
        generated_text = outputs[0].outputs[0].text
        prompt += generated_text
            
        if "</answer>" in generated_text:
            match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
            if match: final_answer = match.group(1).strip()
            break
        elif "</subquery>" in generated_text:
            sq_match = re.search(r"<subquery>(.*?)</subquery>", generated_text, re.DOTALL)
            if sq_match:
                q_str = sq_match.group(1).strip()
                doc_str = retriever.batch_search([q_str])[0]
                retrieval_block = f"\n<retrieval>\n{doc_str}\n</retrieval>\n<step>\n"
                prompt += retrieval_block
            else:
                prompt += "\n<retrieval>Error in query parsing</retrieval>\n<step>\n"
        else:
            prompt += "\n<step>\n"

    print(f"Answer: {final_answer}")

if __name__ == "__main__":
    main()
