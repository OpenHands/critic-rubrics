"""
Batch processing utilities for critic rubrics annotators.
"""

import json
import os
import time
import logging
from typing import Dict, List, Any, Optional, Union, TypeVar, Callable
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from .annotators.base import BaseAnnotator
from .types import BatchConfig

T = TypeVar('T')

logger = logging.getLogger(__name__)



class BatchProcessor:
    """Handles batch processing for critic rubrics annotators."""
    
    def __init__(self, annotator: BaseAnnotator[T], config: Optional[BatchConfig] = None):
        self.annotator = annotator
        self.config = config or BatchConfig()
        
        # Ensure output folder exists
        os.makedirs(self.config.output_folder, exist_ok=True)
    
    def to_openai_batch_request(self, request_data: Dict[str, Any], custom_id: str) -> Dict[str, Any]:
        """Convert annotator request to OpenAI batch JSONL format."""
        litellm_request = self.annotator.get_request(request_data)
        
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": litellm_request["model"],
                "messages": litellm_request["messages"],
                "tools": litellm_request["tools"],
                "tool_choice": litellm_request["tool_choice"],
                "temperature": self.annotator.temperature,
                "max_tokens": self.annotator.max_tokens
            }
        }
    
    def to_anthropic_batch_request(self, request_data: Dict[str, Any], custom_id: str) -> Dict[str, Any]:
        """Convert annotator request to Anthropic batch format as a serializable dict."""
        try:
            from litellm.llms.anthropic.chat.transformation import AnthropicConfig
        except ImportError:
            raise ImportError("Anthropic dependencies not available. Install with: pip install anthropic")
        
        litellm_request = self.annotator.get_request(request_data)
        
        anthropic_config = AnthropicConfig()
        tools, _ = anthropic_config._map_tools(litellm_request["tools"])
        transformed_request = anthropic_config.transform_request(
            model=litellm_request["model"],
            messages=litellm_request["messages"],
            optional_params={
                'tools': tools,
                'max_tokens': self.annotator.max_tokens,
            },
            litellm_params={},
            headers={},
        )
        
        # Return a plain dict suitable for JSONL or API submission
        return {
            "custom_id": custom_id,
            "params": {
                **transformed_request,
                "tool_choice": {'type': 'tool', 'name': tools[0]['name']}
            }
        }
    
    def create_batch_file(self, request_data_list: List[Dict[str, Any]], 
                         custom_ids: List[str], provider: str = "openai") -> str:
        """Create a batch file for the specified provider."""
        if len(request_data_list) != len(custom_ids):
            raise ValueError("request_data_list and custom_ids must have the same length")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = os.path.join(self.config.output_folder, f"batch_{provider}_{timestamp}.jsonl")
        
        with open(batch_file, 'w') as f:
            for request_data, custom_id in zip(request_data_list, custom_ids):
                if provider == "openai":
                    batch_request = self.to_openai_batch_request(request_data, custom_id)
                elif provider == "anthropic":
                    # Only write serializable dicts for Anthropic
                    batch_request = self.to_anthropic_batch_request(request_data, custom_id)
                else:
                    raise ValueError(f"Unsupported provider: {provider}")
                
                f.write(json.dumps(batch_request) + '\n')
        
        logger.info(f"Created batch file: {batch_file} with {len(request_data_list)} requests")
        return batch_file
    
    def submit_openai_batch(self, batch_file: str) -> str:
        """Submit batch file to OpenAI and return batch_id."""
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        client = openai.OpenAI(api_key=self.annotator.api_key)
        
        # Upload the batch file
        with open(batch_file, 'rb') as f:
            batch_input_file = client.files.create(
                file=f,
                purpose="batch"
            )
        
        # Create the batch
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        logger.info(f"Submitted OpenAI batch: {batch.id}")
        return batch.id
    
    def submit_anthropic_batch(self, requests: List[Dict[str, Any]]) -> str:
        """Submit batch requests to Anthropic and return batch_id."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic package not available. Install with: pip install anthropic")
        
        client = anthropic.Anthropic(api_key=self.annotator.api_key)
        
        # Submit the batch (requests should be dicts with 'custom_id' and 'params')
        message_batch = client.messages.batches.create(requests=requests)
        
        logger.info(f"Submitted Anthropic batch: {message_batch.id}")
        return message_batch.id
    
    def submit_batch(self, request_data_list: List[Dict[str, Any]], 
                    custom_ids: Optional[List[str]] = None, 
                    provider: Optional[str] = None) -> str:
        """Submit a batch of requests and return batch_id."""
        provider = provider or self.config.provider
        
        if custom_ids is None:
            custom_ids = [f"request_{i}" for i in range(len(request_data_list))]
        
        if provider == "openai":
            batch_file = self.create_batch_file(request_data_list, custom_ids, provider)
            return self.submit_openai_batch(batch_file)
        elif provider == "anthropic":
            # For Anthropic, we need to convert to Request objects
            requests = []
            for request_data, custom_id in zip(request_data_list, custom_ids):
                batch_request = self.to_anthropic_batch_request(request_data, custom_id)
                requests.append(batch_request)
            return self.submit_anthropic_batch(requests)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def check_batch_status(self, batch_id: str, provider: Optional[str] = None) -> Dict[str, Any]:
        """Check the status of a batch."""
        provider = provider or self.config.provider
        
        if provider == "openai":
            return self._check_openai_batch_status(batch_id)
        elif provider == "anthropic":
            return self._check_anthropic_batch_status(batch_id)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _check_openai_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Check OpenAI batch status."""
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        client = openai.OpenAI(api_key=self.annotator.api_key)
        batch = client.batches.retrieve(batch_id)
        
        return {
            'id': batch.id,
            'status': batch.status,
            'request_counts': {
                'total': getattr(batch.request_counts, 'total', None),
                'completed': getattr(batch.request_counts, 'completed', None),
                'failed': getattr(batch.request_counts, 'failed', None)
            } if batch.request_counts else {},
            'created_at': batch.created_at,
            'completed_at': getattr(batch, 'completed_at', None),
            'output_file_id': getattr(batch, 'output_file_id', None),
            'error_file_id': getattr(batch, 'error_file_id', None)
        }
    
    def _check_anthropic_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Check Anthropic batch status."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic package not available. Install with: pip install anthropic")
        
        client = anthropic.Anthropic(api_key=self.annotator.api_key)
        batch = client.messages.batches.retrieve(batch_id)
        
        return {
            'id': batch.id,
            'type': batch.type,
            'processing_status': batch.processing_status,
            'request_counts': {
                'processing': batch.request_counts.processing,
                'succeeded': batch.request_counts.succeeded,
                'errored': batch.request_counts.errored,
                'canceled': batch.request_counts.canceled,
                'expired': batch.request_counts.expired,
            },
            'created_at': batch.created_at,
            'expires_at': batch.expires_at,
            'results_url': getattr(batch, 'results_url', None)
        }
    
    def download_batch_results(self, batch_id: str, provider: Optional[str] = None) -> List[T]:
        """Download and parse batch results."""
        provider = provider or self.config.provider
        
        if provider == "openai":
            return self._download_openai_batch_results(batch_id)
        elif provider == "anthropic":
            return self._download_anthropic_batch_results(batch_id)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _download_openai_batch_results(self, batch_id: str) -> List[T]:
        """Download and parse OpenAI batch results."""
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        client = openai.OpenAI(api_key=self.annotator.api_key)
        
        # Get batch details
        batch = client.batches.retrieve(batch_id)
        if not batch.output_file_id:
            raise ValueError(f"Batch {batch_id} has no output file")
        
        # Download results
        file_content = client.files.content(batch.output_file_id)
        results = []
        
        for line in file_content.text.strip().split('\n'):
            if not line:
                continue
            try:
                result = json.loads(line)
            except Exception:
                logger.error("Malformed JSONL line in OpenAI results: %s", line[:200])
                continue

            if not result.get('response'):
                # Possibly an error line; surface custom_id for debugging
                cid = result.get('custom_id')
                logger.warning("OpenAI batch line missing response (custom_id=%s): %s", cid, list(result.keys()))
                continue

            response_body = result['response'].get('body') or {}
            choices = response_body.get('choices') or []
            if not choices:
                logger.warning("OpenAI response has no choices (custom_id=%s)", result.get('custom_id'))
                continue
            message = choices[0].get('message', {})
            tool_calls = message.get('tool_calls') or []
            if not tool_calls:
                logger.warning("No tool_calls in OpenAI response (custom_id=%s)", result.get('custom_id'))
                continue
            try:
                tool_call = tool_calls[0]
                tool_call_args = json.loads(tool_call['function']['arguments'])
                parsed_result = self.annotator._parse_result(tool_call_args)
                results.append(parsed_result)
            except Exception as e:
                logger.error("Failed to parse tool call (custom_id=%s): %s", result.get('custom_id'), e)
        
        return results
    
    def _download_anthropic_batch_results(self, batch_id: str) -> List[T]:
        """Download and parse Anthropic batch results."""
        try:
            import anthropic
            import requests
        except ImportError:
            raise ImportError("Anthropic package not available. Install with: pip install anthropic requests")
        
        client = anthropic.Anthropic(api_key=self.annotator.api_key)
        
        # Get batch details
        batch = client.messages.batches.retrieve(batch_id)
        if not batch.results_url:
            raise ValueError(f"Batch {batch_id} has no results URL")
        
        # Download results
        # Prefer client-managed retrieval; fall back to signed URL with requests
        results = []
        text = None
        try:
            file = client.messages.batches.results(batch_id)
            text = getattr(file, 'text', None)
        except Exception:
            pass
        if text is None:
            import requests
            response = requests.get(batch.results_url, headers={"Authorization": f"Bearer {self.annotator.api_key}"})
            response.raise_for_status()
            text = response.text

        for line in (text or '').strip().split('\n'):
            if not line:
                continue
            try:
                result = json.loads(line)
            except Exception:
                logger.error("Malformed JSONL line in Anthropic results: %s", line[:200])
                continue
            res = result.get('result') or {}
            if res.get('type') != 'succeeded':
                logger.warning("Anthropic line not succeeded (custom_id=%s): %s", result.get('custom_id'), res.get('type'))
                continue
            message = res.get('message') or {}
            content = message.get('content') or []
            if not content:
                logger.warning("Anthropic message has no content (custom_id=%s)", result.get('custom_id'))
                continue
            first = content[0]
            if first.get('type') != 'tool_use':
                logger.warning("Anthropic first content is not tool_use (custom_id=%s)", result.get('custom_id'))
                continue
            try:
                tool_call_args = first.get('input', {})
                parsed_result = self.annotator._parse_result(tool_call_args)
                results.append(parsed_result)
            except Exception as e:
                logger.error("Failed parsing Anthropic tool_use (custom_id=%s): %s", result.get('custom_id'), e)
        
        return results
    
    def process_async(self, request_data_list: List[Dict[str, Any]], 
                     max_workers: int = 5) -> List[T]:
        """Process requests asynchronously with global rate limiting and retries."""
        try:
            import litellm
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from threading import Semaphore, Lock
            import time as _time
            import random
        except ImportError:
            raise ImportError("Required packages not available for async processing")
        
        results: List[T] = []
        semaphore = Semaphore(max_workers)
        rate_lock = Lock()
        interval = 60.0 / max(1, self.config.rate_limit_rpm)
        next_allowed_time = {'t': 0.0}
        
        def acquire_rate_slot():
            with rate_lock:
                now = _time.monotonic()
                wait = max(0.0, next_allowed_time['t'] - now)
                if wait > 0:
                    _time.sleep(wait)
                next_allowed_time['t'] = max(now, next_allowed_time['t']) + interval
        
        def process_single_request(request_data):
            with semaphore:
                retries = 0
                while True:
                    try:
                        acquire_rate_slot()
                        litellm_request = self.annotator.get_request(request_data)
                        if self.config.request_timeout is not None:
                            litellm_request["timeout"] = self.config.request_timeout
                        response = litellm.completion(**litellm_request)

                        choices = getattr(response, 'choices', None)
                        if not choices:
                            raise ValueError("No choices in response")
                        message = choices[0].message
                        tool_calls = getattr(message, 'tool_calls', None)
                        if not tool_calls:
                            raise ValueError("No tool_calls in response")
                        tool_call = tool_calls[0]
                        tool_call_args = json.loads(tool_call.function.arguments)
                        return self.annotator._parse_result(tool_call_args)
                    except Exception as e:
                        retries += 1
                        if retries > getattr(self.config, 'max_retries', 0):
                            logger.error(f"Error processing request after retries: {e}")
                            return None
                        backoff = min(10.0, (2 ** (retries - 1)) + random.uniform(0, 0.5))
                        logger.warning(f"Retrying request (attempt {retries}) after error: {e}. Backoff {backoff:.2f}s")
                        _time.sleep(backoff)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_request = {
                executor.submit(process_single_request, request_data): request_data 
                for request_data in request_data_list
            }
            
            for future in as_completed(future_to_request):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        return results
