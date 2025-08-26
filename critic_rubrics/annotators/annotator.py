from __future__ import annotations

import io
import json
import time
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, cast

import litellm
from litellm import ChatCompletionRequest, completion


class Annotator:
    """Simple LiteLLM-backed annotator.

    - annotate: send a single ChatCompletionRequest to a LiteLLM proxy in real time
    - batch_annotate: submit a batch of ChatCompletionRequests via LiteLLM Batches API
      and block until the batch finishes, then return parsed NDJSON outputs.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"] = "/v1/chat/completions",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.endpoint: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"] = endpoint
        self._client_kwargs: Dict[str, Any] = {}
        if base_url is not None:
            self._client_kwargs["base_url"] = base_url
        if api_key is not None:
            self._client_kwargs["api_key"] = api_key

    # -----------------------
    # Real-time (single) call
    # -----------------------
    def annotate(self, request: ChatCompletionRequest) -> Any:
        """Send a single request to LiteLLM.

        Returns the LiteLLM response object.
        """
        payload: Dict[str, Any] = dict(request)  # TypedDict -> real dict
        if "model" not in payload:
            if self.model is None:
                raise ValueError("ChatCompletionRequest missing 'model' and no default model set")
            payload["model"] = self.model
        resp = completion(**payload, **self._client_kwargs)
        return resp

    # -----------------------
    # Batch processing
    # -----------------------
    def batch_annotate(
        self,
        requests: Iterable[ChatCompletionRequest],
        *,
        completion_window: Literal["24h"] = "24h",
        poll_interval_seconds: float = 30.0,
        timeout_seconds: Optional[float] = None,
        request_custom_ids: Optional[Iterable[str]] = None,
    ) -> Tuple[litellm.LiteLLMBatch, List[Dict[str, Any]]]:
        """Submit a batch to LiteLLM Proxy and wait for completion.

        Returns (batch_metadata, parsed_output_records)
        where parsed_output_records is a list of dicts parsed from NDJSON output.
        """
        # Build NDJSON body in-memory to avoid temp files on disk
        lines: List[str] = []
        ids_iter = iter(request_custom_ids) if request_custom_ids is not None else None
        for i, req in enumerate(requests):
            body: Dict[str, Any] = dict(req)
            if "model" not in body:
                if self.model is None:
                    raise ValueError("ChatCompletionRequest missing 'model' and no default model set")
                body["model"] = self.model
            custom_id = next(ids_iter) if ids_iter is not None else f"req_{i:08d}"
            line = {
                "custom_id": str(custom_id),
                "method": "POST",
                "url": self.endpoint,
                "body": body,
            }
            lines.append(json.dumps(line, separators=(",", ":")))
        if not lines:
            # Create an empty batch is pointless; return early
            raise ValueError("No requests provided to batch_annotate")

        ndjson_bytes = ("\n".join(lines)).encode("utf-8")
        file_like = io.BytesIO(ndjson_bytes)

        # Upload NDJSON as a file for batch input
        file_obj = litellm.create_file(
            file=file_like, purpose="batch", **self._client_kwargs
        )

        # Create the batch
        batch = cast(
            litellm.LiteLLMBatch,
            litellm.create_batch(
                completion_window=completion_window,
                endpoint=self.endpoint,
                input_file_id=cast(str, getattr(file_obj, "id")),
                **self._client_kwargs,
            ),
        )

        # Poll until completion (or timeout)
        start = time.time()
        status = getattr(batch, "status", None)
        while status not in {"completed", "failed", "cancelled", "expired"}:
            time.sleep(poll_interval_seconds)
            batch = cast(
                litellm.LiteLLMBatch,
                litellm.retrieve_batch(batch_id=cast(str, getattr(batch, "id")), **self._client_kwargs),
            )
            status = getattr(batch, "status", None)
            if timeout_seconds is not None and (time.time() - start) > timeout_seconds:
                break

        # Attempt to read outputs if available
        outputs: List[Dict[str, Any]] = []
        output_file_id = getattr(batch, "output_file_id", None)
        if status == "completed" and output_file_id:
            content = litellm.file_content(
                file_id=cast(str, output_file_id), **self._client_kwargs
            )
            try:
                raw_bytes: bytes
                # litellm.file_content may return an httpx response content wrapper or bytes
                if hasattr(content, "read"):
                    raw_bytes = content.read()  # type: ignore[assignment]
                else:
                    raw_bytes = cast(bytes, content)  # type: ignore[assignment]
                for line in raw_bytes.decode("utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        outputs.append(json.loads(line))
                    except Exception:
                        outputs.append({"raw": line})
            except Exception:
                pass

        return batch, outputs
