import json
import os
import time
import asyncio
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI

# ===================== 路径配置 =====================
input_file = "/home/v-zhaowan/ds/zhaowang/rag/data/raw/mulsique.json"
output_file = "/home/v-zhaowan/ds/zhaowang/rag/data/raw/mulsique_refined_v1.jsonl"

os.makedirs(os.path.dirname(output_file), exist_ok=True)
API_CACHE = {}

# ===================== Azure 配置 =====================
API_VERSION = "2024-10-21"
DEPLOYMENT_NAME = "gpt-4o_2024-11-20"
INSTANCE = "gcr/shared"
ENDPOINT = f"https://trapi.research.microsoft.com/{INSTANCE}"
SCOPE = "api://trapi/.default"

credential = get_bearer_token_provider(
    ChainedTokenCredential(
        AzureCliCredential(),
        ManagedIdentityCredential(),
    ),
    SCOPE,
)

client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    azure_ad_token_provider=credential,
    api_version=API_VERSION,
    timeout=60.0,
)

# ===================== Prompt 模板 =====================
SYSTEM_PROMPT = """You are an expert question decomposition and reasoning analyst.
You will be given a text passage and must produce reasoning steps in natural language.

Your task:
Break down the content into several reasoning steps, each represented as a JSON object with the following structure:

{
    "new_CoT": [
        {
            "subquery": "... (a natural-language question about one factual aspect of the text; DO NOT include the answer or any hint of it)",
            "sub-answer": "... (a concise factual answer)",
            "paragraph": "... (use the given text verbatim)"
        },
        ...
    ]
}

Guidelines:
- Each subquery must be a *complete sentence* question, clearly phrased, and must NOT contain the answer or suggest it.
- Each sub-answer should be a concise factual response; it may be a single word, a short phrase, a number, or a full sentence — full sentences are allowed but not required.
- Each paragraph must directly use the original text verbatim, without any rewriting, paraphrasing, or summarization.
- Only include factual and inferable information from the text.
- Output **only** one valid JSON object following this schema, no extra text, explanations, or Markdown.
"""

def build_user_prompt(query, chain, answer):
    return f"""
Below are the fields. Produce ONE JSON object that matches the REQUIRED OUTPUT SCHEMA specified by the system message exactly.

query: {query}
chain_of_thought: {chain}
answer: {answer}

Remember:
- Rewrite each subquery into a complete question sentence.
- Provide sub-answer as a concise factual response (may be a short phrase, number, or a full sentence).
- Provide paragraph as the original text (verbatim).
- Do not include any additional keys or text; the response must be exactly a single JSON object matching the schema.
"""

# ===================== 异步调用函数 =====================
async def async_call_gpt4(item, semaphore):
    uid = item.get("id")
    query = item.get("query")
    chain = item.get("chain_of_thought")
    answer = item.get("answer")

    if not query or not chain or answer is None:
        print(f"❌ Skip {uid}: Missing required field")
        return None

    cache_key = uid
    if cache_key in API_CACHE:
        item["new_chain_of_thought"] = API_CACHE[cache_key]
        return item

    user_prompt = build_user_prompt(query, chain, answer)

    async with semaphore:
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                top_p=0.9,
                max_tokens=4096,
            )
            raw_output = response.choices[0].message.content.strip()

           
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            parsed = json.loads(raw_output[start:end])
            new_cot = parsed.get("new_CoT", [])

            item["new_chain_of_thought"] = new_cot
            API_CACHE[cache_key] = new_cot
            print(f"[✔] {uid} done")
            return item

        except Exception as e:
            print(f"[API Error] {uid}: {e}")
            return None

# ===================== 主流程 =====================
async def main():
    # 加载已处理 ID
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as fout:
            for line in fout:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["id"])
                except:
                    continue
    print(f"✅ Already processed: {len(processed_ids)}")

    # 加载待处理数据
    with open(input_file, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    items = [item for item in data if item.get("id") not in processed_ids]
    print(f"🚀 Ready to process {len(items)} new items...")

    semaphore = asyncio.Semaphore(3)
    tasks = [asyncio.create_task(async_call_gpt4(item, semaphore)) for item in items]

    with open(output_file, "a", encoding="utf-8") as fout:
        for future in asyncio.as_completed(tasks):
            result = await future
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

    print("✅ Done! All refined reasoning chains saved.")

# ===================== 入口 =====================
if __name__ == "__main__":
    asyncio.run(main())
