from typing import List
import aiohttp
import requests

class RemoteRetriever:
    def __init__(self, url: str="http://localhost:8000/retrieve", topk: int=3):
        self.search_url = url
        self.topk = topk

    def batch_search(self, queries: List[str]) -> List[str]:
        results = self._batch_search(queries)['result']
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        payload = {
            "queries": queries,
            "topk": self.topk,
            "return_scores": True 
        }
        return requests.post(self.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

class AsyncRemoteRetriever:
    def __init__(self, url: str="http://localhost:8000/retrieve", topk: int=3):
        self.search_url = url
        self.topk = topk

    async def batch_search(self, queries: List[str]) -> List[str]:
        response = await self._batch_search(queries)
        results = response['result']
        return [self._passages2string(result) for result in results]

    async def _batch_search(self, queries):
        payload = {
            "queries": queries,
            "topk": self.topk,
            "return_scores": True 
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.search_url, json=payload, timeout=500) as response: 
                response.raise_for_status()
                return await response.json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference