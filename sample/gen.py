import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset
import json
import os
import re
import concurrent.futures
from collections import defaultdict
import time

# --- Config for Monte Carlo Generation ---
MODEL_NAME = "Qwen/Qwen3-8B" # Base model name
SFT_ADAPTER_PATH = "/home/v-zhaowan/zhaowang/rag/save/730/final_adapter" # Path to your SFT LoRA adapter
TRAIN_DATA_PATH = "/home/v-zhaowan/zhaowang/rag/data/train.json" # Your SFT training data (questions only for generation)
MC_OUTPUT_DIR = "/home/v-zhaowan/zhaowang/rag/mc_generated_data" # Directory to save generated reasoning chains
MC_BATCH_SIZE = 8 # Batch size for model inference during generation
NUM_SAMPLES_PER_QUESTION = 10 # For each question, how many distinct reasoning chains to attempt to generate
MAX_GENERATION_STEPS = 7 # Maximum number of thought/sub-query/sub-answer steps
MAX_TOTAL_TOKENS = 2048 # Max tokens for the entire generated sequence for a path
TEMP_MIN = 0.7 # Minimum temperature for diverse sampling
TEMP_MAX = 1.0 # Maximum temperature for diverse sampling
TOP_P = 0.9 # Top-p for diverse sampling
NUM_WORKERS = os.cpu_count() if os.cpu_count() else 4 # Number of concurrent processes/threads for questions

# --- Load Model and Tokenizer ---
print(f"Loading base model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Ensure pad_token is set for batch generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Qwen often prefers left padding for generation

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16, # Use bfloat16 as in your training
    device_map="auto" # Distribute across available GPUs
)

print(f"Loading LoRA adapter from: {SFT_ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
model = model.merge_and_unload() # Merge LoRA weights for efficient inference
model.eval() # Set model to evaluation mode

# --- Global Retrieval Cache ---
retrieval_cache = {}

# --- Simulated Retrieval Function (REPLACE THIS WITH YOUR ACTUAL RETRIEVER) ---
def perform_retrieval(query: str) -> str:
    """
    Placeholder for your actual retrieval function.
    You MUST replace this with code that interacts with your knowledge base/search engine.
    Consider adding:
    - Batch retrieval capabilities if your API supports it.
    - Error handling for retrieval failures.
    - More sophisticated caching strategies (e.g., TTL, LRU).
    """
    if query in retrieval_cache:
        # print(f"Cache hit for query: {query[:50]}...")
        return retrieval_cache[query]

    # --- Actual Retrieval Logic Goes Here ---
    # Example: call your search API, query a vector database, etc.
    # For demonstration, we'll just simulate a delay and return dummy data.
    # time.sleep(0.1) # Simulate network delay
    retrieved_data = f"Retrieved_Evidence_for_query: '{query}'"
    # print(f"Performed actual retrieval for query: {query[:50]}...")
    # --- End Actual Retrieval Logic ---

    retrieval_cache[query] = retrieved_data
    return retrieved_data

def batch_perform_retrieval(queries: list[str]) -> dict[str, str]:
    """
    Performs retrieval for a batch of queries, leveraging cache.
    Queries that are not in cache will trigger actual retrieval.
    """
    results = {}
    queries_to_retrieve = []
    for q in queries:
        if q in retrieval_cache:
            results[q] = retrieval_cache[q]
        else:
            queries_to_retrieve.append(q)
    
    if queries_to_retrieve:
        # Here, you'd ideally call your retriever's batch API
        # For simulation, we'll just loop and call single `perform_retrieval`
        # In a real scenario, this would be a single call to a batch endpoint
        for q_to_r in queries_to_retrieve:
            results[q_to_r] = perform_retrieval(q_to_r) # This will also update retrieval_cache
    
    return results

# --- Reasoning Chain Parsing ---
# Regex patterns for parsing the specific format
THOUGHT_PATTERN = r"\[Thought\](.*?)\[/Thought\]"
SUB_QUERY_PATTERN = r"\[Sub-query\](.*?)(\n|$)" # Adjusted to catch newline or end of string
RETRIEVAL_PATTERN = r"\[Retrieval\] -->(.*?)(\n|$)"
SUB_ANSWER_PATTERN = r"\[Sub-answer\](.*?)(\n|$)"
FINAL_ANSWER_PATTERN = r"\[Final Answer\](.*?)(\n|$)"

def parse_reasoning_chain(full_text: str) -> list[tuple[str, str]]:
    """
    Parses a generated reasoning chain into a list of (step_type, content) tuples.
    Assumes the format: [Thought]...[/Thought]\n[Sub-query]...\n[Retrieval] --> ...\n[Sub-answer]...\n[Final Answer]...
    """
    steps = []
    
    # Use a greedy approach to find the first match of any pattern
    # Then consume that part of the string and search again
    remaining_text = full_text

    # Patterns in expected order of appearance
    patterns = {
        "Thought": THOUGHT_PATTERN,
        "Sub-query": SUB_QUERY_PATTERN,
        "Retrieval": RETRIEVAL_PATTERN,
        "Sub-answer": SUB_ANSWER_PATTERN,
        "Final Answer": FINAL_ANSWER_PATTERN
    }
    
    # Compile regexes for efficiency
    compiled_patterns = {k: re.compile(v, re.DOTALL) for k, v in patterns.items()}

    while remaining_text:
        match_found = False
        for step_type, pattern in compiled_patterns.items():
            match = pattern.match(remaining_text.strip()) # .match() tries from start
            if match:
                content = match.group(1).strip()
                steps.append((step_type, content))
                # Remove the matched part plus any surrounding whitespace/newlines
                remaining_text = remaining_text[match.end():].strip()
                match_found = True
                break # Move to next step of the same chain

        if not match_found:
            # If no pattern matches, it could be malformed text or the end
            break
            
    return steps

def validate_final_answer(generated_answer: str, ground_truth_answer: str) -> bool:
    """
    Compares the generated final answer to the ground truth.
    Implement your specific validation logic here.
    For simplicity, using simple string equality/containment.
    """
    if not generated_answer or not ground_truth_answer:
        return False
    # A more robust check might involve fuzzy matching, semantic similarity,
    # or a specific evaluation metric for your task.
    return generated_answer.lower().strip() == ground_truth_answer.lower().strip()

# --- Monte Carlo Generation Logic ---
def generate_single_question_paths(question_data: dict, model, tokenizer, retriever_func) -> list[dict]:
    question_text = question_data["question"]
    ground_truth_answer = question_data.get("answer", None) # Assume your train.json has "answer" for validation

    generated_paths = []

    # Store active paths. Each path is a dictionary.
    # Example path: {"prompt": "...", "steps": [], "is_complete": False, "final_answer": None, "raw_output": "..."}
    active_paths = [{"prompt": f"<|im_start|>user\n{question_text}<|im_end|>\n<|im_start|>assistant\n",
                     "steps": [],
                     "is_complete": False,
                     "final_answer": None,
                     "raw_output": "",
                     "current_prefix_tokens": [],
                     "temperature_used": None,
                     "top_p_used": None,
                     "initial_question": question_text,
                     "ground_truth_answer": ground_truth_answer} for _ in range(NUM_SAMPLES_PER_QUESTION)]

    for step_idx in range(MAX_GENERATION_STEPS):
        # Filter out completed paths and collect inputs for the next batch
        prompts_to_generate = []
        indices_to_process = []
        temperatures = []
        
        for i, path in enumerate(active_paths):
            if not path["is_complete"]:
                prompts_to_generate.append(path["prompt"])
                indices_to_process.append(i)
                # Vary temperature for diversity
                temp = TEMP_MIN + (TEMP_MAX - TEMP_MIN) * (i / NUM_SAMPLES_PER_QUESTION) # Simple linear variation
                temperatures.append(temp)
                path["temperature_used"] = temp
                path["top_p_used"] = TOP_P

        if not prompts_to_generate:
            break # All paths completed

        # Tokenize inputs for batch generation
        inputs = tokenizer(prompts_to_generate, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOTAL_TOKENS).to(model.device)
        
        # Determine max_new_tokens for the current generation step.
        # This is a heuristic; ideally, you'd generate till a stop token for each part (Thought, Sub-query etc.)
        # For simplicity, we'll let it generate up to a reasonable chunk and then parse.
        max_new_tokens_per_step = 128 # Generate a chunk for parsing each step

        # Perform batch generation
        generated_token_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens_per_step,
            do_sample=True,
            temperature=torch.tensor(temperatures).to(model.device), # Apply per-sample temperature
            top_p=TOP_P,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=False,
            return_dict_in_generate=False # Not needed for simple generation
        )
        
        # Decode generated text and update paths
        for i, gen_ids in enumerate(generated_token_ids):
            original_path_idx = indices_to_process[i]
            current_path = active_paths[original_path_idx]

            # Decode only the newly generated part (excluding prompt tokens)
            decoded_output = tokenizer.decode(gen_ids[inputs.input_ids[i].shape[0]:], skip_special_tokens=True).strip()
            
            # Append to raw_output for full chain
            current_path["raw_output"] += decoded_output
            
            # Update prompt for next turn (important for subsequent generations)
            current_path["prompt"] = prompts_to_generate[i] + decoded_output

            # --- Parse and Process Step by Step ---
            # This is where the core logic of your "multi-step" generation happens.
            # We are generating a chunk and trying to extract a *full* logical step.
            # If a step is incomplete, it means we need to generate more.
            # This requires careful parsing.
            
            # Try to extract the *last* complete step
            parsed_steps = parse_reasoning_chain(current_path["raw_output"])
            
            # If new steps were parsed compared to previous, process them
            if len(parsed_steps) > len(current_path["steps"]):
                newly_parsed_step_tuples = parsed_steps[len(current_path["steps"]):] # Get only the new ones
                
                for step_type, content in newly_parsed_step_tuples:
                    current_path["steps"].append((step_type, content))
                    
                    if step_type == "Sub-query":
                        # Perform retrieval immediately after a Sub-query is generated
                        # For now, we call single `perform_retrieval` but you should batch this if possible
                        # This means you'd need to collect all sub-queries across all paths,
                        # do a batch retrieval, and then update the specific path.
                        
                        # Simplified for now: assume retrieval happens in place for current step
                        retrieved_evidence = perform_retrieval(content)
                        current_path["steps"].append(("Retrieval", retrieved_evidence))
                        # Update prompt to include retrieval for next model generation
                        current_path["prompt"] += f"\n[Retrieval] --> {retrieved_evidence}\n"

                    elif step_type == "Final Answer":
                        current_path["final_answer"] = content
                        current_path["is_complete"] = True
                        break # Stop processing this path if final answer is found

        # Move completed paths to the final list
        new_active_paths = []
        for path in active_paths:
            if path["is_complete"]:
                generated_paths.append(path)
            else:
                new_active_paths.append(path)
        active_paths = new_active_paths
        
        if not active_paths:
            break # All paths completed or reached max steps

    # Add any remaining incomplete paths to the generated_paths list
    generated_paths.extend(active_paths)

    # Filter by final answer correctness (if ground truth is available)
    # This happens *after* all paths are generated for a question
    final_filtered_paths = []
    for path in generated_paths:
        if path["is_complete"] and path["ground_truth_answer"]:
            if validate_final_answer(path["final_answer"], path["ground_truth_answer"]):
                path["answer_correct"] = True
            else:
                path["answer_correct"] = False
        else:
            path["answer_correct"] = False # Or mark as not applicable if no ground truth

        # Remove temporary fields for cleaner output if desired
        path.pop("prompt", None)
        path.pop("current_prefix_tokens", None)

        final_filtered_paths.append(path)

    print(f"Generated {len(final_filtered_paths)} paths for question: '{question_text}'")
    return final_filtered_paths

# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs(MC_OUTPUT_DIR, exist_ok=True)

    print(f"Loading questions from {TRAIN_DATA_PATH} for Monte Carlo generation...")
    # Assume TRAIN_DATA_PATH contains a list of dictionaries, each with "question" and "answer"
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        raw_questions = json.load(f)
    
    # Extract just the questions if your `train.json` is in the conversation format
    # Adapt this part based on your actual `train.json` structure
    processed_questions_data = []
    for item in raw_questions:
        # Assuming the first "user" turn is the main question
        question_text = ""
        answer_text = ""
        for turn in item["conversation"]:
            if turn["role"] == "user":
                question_text = turn["content"]
                break # Take the first user turn as the main question
        
        # Assuming the last "assistant" turn's content might contain the final answer
        # This part might need adjustment based on how your ground truth is stored
        for turn in reversed(item["conversation"]):
            if turn["role"] == "assistant":
                # Attempt to extract final answer if it's formatted like [Final Answer]
                match = re.search(FINAL_ANSWER_PATTERN, turn["content"])
                if match:
                    answer_text = match.group(1).strip()
                else:
                    answer_text = turn["content"].strip() # Fallback if not specifically formatted
                break

        if question_text:
            processed_questions_data.append({"question": question_text, "answer": answer_text})


    print(f"Total questions to process: {len(processed_questions_data)}")

    all_generated_reasoning_chains = []
    
    # Use ThreadPoolExecutor for concurrent processing of questions
    # Each thread will generate multiple paths for one question
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_question = {
            executor.submit(generate_single_question_paths, q_data, model, tokenizer, batch_perform_retrieval): q_data["question"]
            for q_data in processed_questions_data
        }

        for i, future in enumerate(concurrent.futures.as_completed(future_to_question)):
            question = future_to_question[future]
            try:
                paths = future.result()
                all_generated_reasoning_chains.extend(paths)
                print(f"[{i+1}/{len(processed_questions_data)}] Finished processing: '{question[:50]}...'")
            except Exception as exc:
                print(f'Question "{question[:50]}..." generated an exception: {exc}')

    # Save all generated chains to a JSONL file
    output_filename = os.path.join(MC_OUTPUT_DIR, "generated_reasoning_chains.jsonl")
    with open(output_filename, 'w', encoding='utf-8') as f:
        for chain in all_generated_reasoning_chains:
            f.write(json.dumps(chain, ensure_ascii=False) + '\n')

    print(f"\nMonte Carlo generation complete! Total paths generated: {len(all_generated_reasoning_chains)}")
    print(f"Results saved to {output_filename}")

    # Example of how you might further process these for DPO
    # filtered_correct_paths = [p for p in all_generated_reasoning_chains if p["answer_correct"]]
    # print(f"Paths with correct final answer: {len(filtered_correct_paths)}")
    # Now you would select chosen/rejected pairs from `filtered_correct_paths`
    # based on the internal `steps` quality using your LLM or human annotators.