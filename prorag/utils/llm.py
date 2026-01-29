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
from openai import AsyncOpenAI, RateLimitError, APIError

class AsyncLLMEngine:
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        concurrency: int = 10,
        timeout: float = 60.0
    ):
        self.model = model
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key is required. Please set 'OPENAI_API_KEY' environment variable or pass it explicitly.")

        print(f"🚀 Initializing OpenAI Client (Model: {self.model})...")
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout
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
                model=self.model,
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
            print(f"📂 Found existing output file: {output_file}, loading processed IDs...")
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        line = line.strip()
                        if not line: continue
                        processed_ids.add(json.loads(line)["id"])
                    except Exception: 
                        pass
        
        pending_items = [x for x in items if x.get("id") not in processed_ids]
        print(f"📊 Total: {len(items)}, Processed: {len(processed_ids)}, Pending: {len(pending_items)}")

        if not pending_items:
            print("✅ All done! No items left to process.")
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

                parsed_response = response_text
                
                if extract_json:
                    try:
                        parsed_response = json.loads(response_text)
                    except json.JSONDecodeError:
                        start = response_text.find("{")
                        end = response_text.rfind("}") + 1
                        if start != -1 and end != -1:
                            try:
                                parsed_response = json.loads(response_text[start:end])
                            except:
                                parsed_response = {"raw_content": response_text, "error": "json_decode_failed"}
                        else:
                            parsed_response = {"raw_content": response_text, "error": "no_json_found"}
                    
                if result_parser:
                    return result_parser(item, parsed_response)
                else:
                    if "response_data" not in item:
                        item["response_data"] = {}
                    
                    if isinstance(parsed_response, dict):
                        item["response_data"].update(parsed_response)
                    else:
                        item["response_data"]["content"] = parsed_response
                        
                    return item

            except Exception as e:
                print(f"❌ Error processing ID {uid}: {e}")
                return None

        tasks = [process_item(item) for item in pending_items]
        
        if os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
        with open(output_file, "a", encoding="utf-8") as f:
            for coro in tqdm_asyncio.as_completed(tasks, desc="LLM Processing"):
                result = await coro
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()