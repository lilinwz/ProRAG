import os
import json
import asyncio
from typing import List, Dict, Callable, Any, Optional
from tqdm.asyncio import tqdm_asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)
from openai import AzureOpenAI, AsyncAzureOpenAI, RateLimitError, APIError
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)

class AsyncLLMEngine:
    def __init__(
        self,
        deployment_name: str = "gpt-4o",
        api_version: str = "2024-10-21",
        endpoint: Optional[str] = None,
        concurrency: int = 10,
        use_azure_identity: bool = True
    ):
        self.deployment = deployment_name
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        
        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("Endpoint is required (env: AZURE_OPENAI_ENDPOINT)")

        if use_azure_identity:
            print("🔐 Using Azure Managed Identity...")
            credential = get_bearer_token_provider(
                ChainedTokenCredential(
                    AzureCliCredential(),
                    ManagedIdentityCredential(),
                ),
                "api://trapi/.default",
            )
            self.client = AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=credential,
                api_version=api_version,
            )
        else:
            print("🔑 Using API Key...")
            self.client = AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=api_version,
            )

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((RateLimitError, APIError))
    )
    async def _generate_single(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float = 0.7,
        json_mode: bool = False
    ) -> str:
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"} if json_mode else None
            )
            return response.choices[0].message.content

    async def run_batch(
        self,
        items: List[Dict],
        prompt_builder: Callable[[Dict], str],
        system_prompt: str,
        output_file: str,
        extract_json: bool = True,
        result_parser: Optional[Callable[[Dict, Any], Optional[Dict]]] = None
    ):
        processed_ids = set()
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    try:
                        processed_ids.add(json.loads(line)["id"])
                    except: pass
        
        pending_items = [x for x in items if x.get("id") not in processed_ids]
        print(f"Total: {len(items)}, Processed: {len(processed_ids)}, Pending: {len(pending_items)}")

        if not pending_items:
            print("✅ All done!")
            return

        async def process_item(item):
            uid = item.get("id")
            try:
                prompt = prompt_builder(item)
                if not prompt: return None

                response_text = await self._generate_single(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    json_mode=extract_json
                )

                if extract_json:
                    try:
                        parsed_response = json.loads(response_text)
                    except json.JSONDecodeError:
                        start = response_text.find("{")
                        end = response_text.rfind("}") + 1
                        parsed_response = json.loads(response_text[start:end])
                    
                if result_parser:
                    return result_parser(item, parsed_response)
                else:
                    item["response_data"].update(parsed_response)
                    return item
            except Exception as e:
                print(f"❌ Error processing {uid}: {e}")
                return None

        tasks = [process_item(item) for item in pending_items]
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as f:
            for coro in tqdm_asyncio.as_completed(tasks, desc="LLM Processing"):
                result = await coro
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()