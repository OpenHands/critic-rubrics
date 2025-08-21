"""
Batch processing utilities for critic rubrics annotators.
"""

import json
import os
import time
import logging
from typing import Dict, List, Any, Optional, Union, TypeVar
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from .annotators.base import BaseAnnotator

T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    provider: str = "openai"  # openai, anthropic, litellm
    batch_size: int = 1000
    max_retries: int = 3
    rate_limit_rpm: int = 60
    output_folder: str = "./batch_results"
    max_tokens: int = 8192
    temperature: float = 0.1


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
                "temperature": litellm_request["temperature"],
                "max_tokens": self.config.max_tokens
            }
        }
    
    def to_anthropic_batch_request(self, request_data: Dict[str, Any], custom_id: str) -> Dict[str, Any]:
        """Convert annotator request to Anthropic batch format."""
        try:
            from litellm.llms.anthropic.chat.transformation import AnthropicConfig
            import anthropic
            from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
            from anthropic.types.messages.batch_create_params import Request
        except ImportError:
            raise ImportError("Anthropic dependencies not available. Install with: pip install anthropic")
        
        litellm_request = self.annotator.get_request(request_data)
        
        # Use litellm's Anthropic config to transform the request
        anthropic_config = AnthropicConfig()
        tools, _ = anthropic_config._map_tools(litellm_request["tools"])
        
        transformed_request = anthropic_config.transform_request(
            model=litellm_request["model"],
            messages=litellm_request["messages"],
            optional_params={
                'tools': tools,
                'max_tokens': self.config.max_tokens,
            },
            litellm_params={},
            headers={},
        )
        
        request = Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(
                tool_choice={'type': 'tool', 'name': tools[0]['name']},
                **transformed_request
            )
        )
        return request
    
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
        
        # Submit the batch
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
            if line:
                result = json.loads(line)
                if result.get('response') and result['response'].get('body'):
                    # Parse the tool call result
                    response_body = result['response']['body']
                    if response_body.get('choices') and response_body['choices'][0].get('message', {}).get('tool_calls'):
                        tool_call = response_body['choices'][0]['message']['tool_calls'][0]
                        tool_call_args = json.loads(tool_call['function']['arguments'])
                        parsed_result = self.annotator._parse_result(tool_call_args)
                        results.append(parsed_result)
        
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
        response = requests.get(batch.results_url)
        response.raise_for_status()
        
        results = []
        for line in response.text.strip().split('\n'):
            if line:
                result = json.loads(line)
                if result.get('result') and result['result'].get('type') == 'succeeded':
                    # Parse the tool call result
                    message = result['result']['message']
                    if message.get('content') and message['content'][0].get('type') == 'tool_use':
                        tool_use = message['content'][0]
                        tool_call_args = tool_use['input']
                        parsed_result = self.annotator._parse_result(tool_call_args)
                        results.append(parsed_result)
        
        return results
    
    def process_async(self, request_data_list: List[Dict[str, Any]], 
                     max_workers: int = 5) -> List[T]:
        """Process requests asynchronously with rate limiting."""
        try:
            import litellm
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from threading import Semaphore
            import time
        except ImportError:
            raise ImportError("Required packages not available for async processing")
        
        results = []
        semaphore = Semaphore(max_workers)
        
        def process_single_request(request_data):
            with semaphore:
                try:
                    # Rate limiting
                    time.sleep(60.0 / self.config.rate_limit_rpm)
                    
                    litellm_request = self.annotator.get_request(request_data)
                    response = litellm.completion(**litellm_request)
                    
                    tool_call = response.choices[0].message.tool_calls[0]
                    tool_call_args = json.loads(tool_call.function.arguments)
                    return self.annotator._parse_result(tool_call_args)
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    return None
        
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