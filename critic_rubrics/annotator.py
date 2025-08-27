"""Mixin classes for rubrics functionality."""

import json
import time
from pathlib import Path
from typing import Any, Iterable, Literal, cast

import litellm
from litellm import ChatCompletionRequest, HttpxBinaryResponseContent, OpenAIFileObject, completion
from litellm.types.utils import LiteLLMBatch, ModelResponse


def content_to_dicts(content: HttpxBinaryResponseContent) -> list[dict[str, Any]]:
    """
    Convert HTTP response content to a list of result dictionaries.
    """
    raw_bytes = content.read()
    results = []
    for line in raw_bytes.decode("utf-8").splitlines():
        if line.strip():
            results.append(json.loads(line))
    return results


class Annotator:
    """Mixin providing annotation capabilities for rubrics."""

    @staticmethod
    def annotate(
        request: ChatCompletionRequest,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> ModelResponse:
        """Send a single request to LiteLLM."""
        payload = dict(request)
        if model:  # override model if provided
            payload["model"] = model

        kwargs = {}
        if base_url:
            kwargs["api_base"] = base_url
        if api_key:
            kwargs["api_key"] = api_key

        for attempt in range(max_retries):
            try:
                response = completion(**payload, **kwargs)
                response = cast(ModelResponse, response)
                return response
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2**attempt)  # exponential backoff
        raise RuntimeError("Unreachable")

    @staticmethod
    def batch_annotate(
        requests: Iterable[ChatCompletionRequest],
        output_dir: str | Path,
        custom_llm_provider: Literal["openai", "azure", "vertex_ai"],
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        endpoint: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"] = "/v1/chat/completions",
        completion_window: Literal["24h"] = "24h",
        max_requests: int = 50_000,
        max_bytes: int = 200 * 1024 * 1024,
        delete_after_upload: bool = True,
    ) -> list[str]:
        """Send batch requests to LiteLLM. Returns list of batch IDs."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        kwargs = {}
        if base_url:
            kwargs["api_base"] = base_url
        if api_key:
            kwargs["api_key"] = api_key

        batch_ids = []
        current_batch = []
        current_size = 0
        batch_num = 0

        def flush_batch():
            nonlocal batch_num, current_batch, current_size
            if not current_batch:
                return

            batch_input_file = output_dir / f"batch_{batch_num:06d}_inputs.jsonl"
            with open(batch_input_file, "w") as f:
                for line in current_batch:
                    f.write(line + "\n")
                print(f"  Flushing batch {batch_num} with {len(current_batch)} requests to {batch_input_file}...")

            with open(batch_input_file, "rb") as f:
                file_obj = litellm.create_file(file=f, purpose="batch", custom_llm_provider=custom_llm_provider, **kwargs)
                file_obj = cast(OpenAIFileObject, file_obj)

            if delete_after_upload:
                batch_input_file.unlink()

            # Create batch
            batch = litellm.create_batch(
                completion_window=completion_window,
                endpoint=endpoint,
                input_file_id=file_obj.id,
                custom_llm_provider=custom_llm_provider,
                metadata={},
                **kwargs,
            )
            batch = cast(LiteLLMBatch, batch)

            # Save batch info
            batch_info = {
                "batch_id": batch.id,
                "input_file_id": file_obj.id,
                "created_at": time.time(),
                "request_count": len(current_batch),
                "custom_llm_provider": custom_llm_provider,
            }

            batch_file = output_dir / f"batch_{batch_num:06d}.json"
            batch_file.write_text(json.dumps(batch_info, indent=2))

            batch_ids.append(batch.id)
            batch_num += 1
            current_batch = []
            current_size = 0

        # Process requests
        for i, request in enumerate(requests):
            body = dict(request)
            if model:  # always override model if provided
                body["model"] = model

            custom_id = f"req_{output_dir.name}_{i:08d}"
            if "metadata" in request:
                custom_id = request.get("metadata", {}).get("custom_request_id", custom_id)
                request.pop("metadata")  # remove so it don't cause issues with LLM completions

            line_obj = {
                "custom_id": custom_id,
                "method": "POST",
                "url": endpoint,
                "body": body,
            }
            line = json.dumps(line_obj, separators=(",", ":"))
            line_size = len(line.encode("utf-8"))

            # Check if we need to flush
            if current_batch and (len(current_batch) >= max_requests or current_size + line_size > max_bytes):
                flush_batch()

            current_batch.append(line)
            current_size += line_size

        # Flush remaining
        flush_batch()

        return batch_ids

    @staticmethod
    def get_batch_results(
        batch_id: str,
        custom_llm_provider: Literal["openai", "azure", "vertex_ai"],
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Get batch status and results if ready."""
        kwargs = {}
        if base_url:
            kwargs["api_base"] = base_url
        if api_key:
            kwargs["api_key"] = api_key

        # Get batch status
        batch = litellm.retrieve_batch(batch_id=batch_id, custom_llm_provider=custom_llm_provider, **kwargs)
        batch = cast(LiteLLMBatch, batch)
        status = {
            "batch_id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "request_counts": batch.request_counts,
            "error": False
        }

        # If not complete, return status only
        if batch.status != "completed":
            return status, []

        if batch.error_file_id:
            error_content = litellm.file_content(file_id=batch.error_file_id, custom_llm_provider=custom_llm_provider, **kwargs)
            error_content = cast(HttpxBinaryResponseContent, error_content)
            status["error"] = True
            return status, content_to_dicts(error_content)

        # Download results
        if not batch.output_file_id:
            return status, []

        content = litellm.file_content(file_id=batch.output_file_id, custom_llm_provider=custom_llm_provider, **kwargs)
        content = cast(HttpxBinaryResponseContent, content)

        return status, content_to_dicts(content)
