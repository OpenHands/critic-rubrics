from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, cast

import litellm
from litellm import ChatCompletionRequest, completion
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential


class Annotator:
    """Simple LiteLLM-backed annotator.

    - annotate: send a single ChatCompletionRequest to a LiteLLM proxy in real time (with retry)
    - batch_annotate: submit a batch of ChatCompletionRequests via LiteLLM Batches API (with retry for create/upload)
    - download_annotation: given a batch_id, download outputs to a results folder
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

    def _retrying(self, *, max_attempts: int = 3) -> Retrying:
        return Retrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),
        )

    # -----------------------
    # Real-time (single) call
    # -----------------------
    def annotate(self, request: ChatCompletionRequest, *, max_attempts: int = 3) -> Any:
        """Send a single request to LiteLLM with retry.

        Returns the LiteLLM response object.
        """
        payload: Dict[str, Any] = dict(request)  # TypedDict -> real dict
        if "model" not in payload:
            if self.model is None:
                raise ValueError("ChatCompletionRequest missing 'model' and no default model set")
            payload["model"] = self.model

        for attempt in self._retrying(max_attempts=max_attempts):
            with attempt:
                return completion(**payload, **self._client_kwargs)
        raise RuntimeError("unreachable")

    # -----------------------
    # Batch processing
    # -----------------------
    def batch_annotate(
        self,
        requests: Iterable[ChatCompletionRequest],
        *,
        completion_window: Literal["24h"] = "24h",
        results_dir: Optional[str | Path] = None,
        max_attempts: int = 3,
    ) -> litellm.LiteLLMBatch:
        """Submit a batch to LiteLLM Proxy and persist identifiers.

        This does not poll for completion. It uploads the NDJSON, creates the batch,
        and, if results_dir is provided, writes a metadata JSON file containing batch_id,
        input_file_id, endpoint, status, created_at, and request_count so that
        download_annotation can fetch outputs later.
        """
        # Build NDJSON body in-memory to avoid temp files on disk
        lines: List[str] = []
        for i, req in enumerate(requests):
            body: Dict[str, Any] = dict(req)
            if "model" not in body:
                if self.model is None:
                    raise ValueError("ChatCompletionRequest missing 'model' and no default model set")
                body["model"] = self.model
            line = {
                "custom_id": f"req_{i:08d}",
                "method": "POST",
                "url": self.endpoint,
                "body": body,
            }
            lines.append(json.dumps(line, separators=(",", ":")))
        if not lines:
            raise ValueError("No requests provided to batch_annotate")

        ndjson_bytes = ("\n".join(lines)).encode("utf-8")

        # Save inputs if requested
        folder_path: Optional[Path] = Path(results_dir) if results_dir is not None else None
        if folder_path is not None:
            folder_path.mkdir(parents=True, exist_ok=True)
            try:
                (folder_path / "inputs.ndjson").write_bytes(ndjson_bytes)
            except Exception:
                pass

        # Upload NDJSON as a file for batch input with retry; recreate BytesIO each attempt
        file_obj: Any | None = None
        for attempt in self._retrying(max_attempts=max_attempts):
            with attempt:
                file_like = io.BytesIO(ndjson_bytes)
                file_obj = litellm.create_file(file=file_like, purpose="batch", **self._client_kwargs)
        if file_obj is None:
            raise RuntimeError("Failed to create file for batch input after retries")

        input_file_id = cast(str, getattr(file_obj, "id"))

        # Create the batch with retry
        batch: Any | None = None
        for attempt in self._retrying(max_attempts=max_attempts):
            with attempt:
                batch = cast(
                    litellm.LiteLLMBatch,
                    litellm.create_batch(
                        completion_window=completion_window,
                        endpoint=self.endpoint,
                        input_file_id=input_file_id,
                        **self._client_kwargs,
                    ),
                )
        if batch is None:
            raise RuntimeError("Failed to create batch after retries")

        # Persist batch metadata
        if folder_path is not None:
            meta = {
                "batch_id": cast(str, getattr(batch, "id", None)),
                "input_file_id": input_file_id,
                "endpoint": self.endpoint,
                "status": getattr(batch, "status", None),
                "created_at": time.time(),
                "request_count": len(lines),
            }
            try:
                (folder_path / "batch.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            except Exception:
                pass

        return batch

    def download_annotation(
        self,
        results_dir: str | Path,
        *,
        max_attempts: int = 3,
    ) -> Tuple[litellm.LiteLLMBatch, List[Dict[str, Any]]]:
        """Fetch latest batch status and download outputs into results_dir if ready.

        Workflow:
        - Read results_dir/batch.json for batch_id
        - retrieve_batch with retry; update batch.json with latest status
        - If completed and output_file_id is present, download to outputs.ndjson and parse
        - Return (batch, parsed_outputs). If not completed, outputs is []
        """
        folder = Path(results_dir)
        folder.mkdir(parents=True, exist_ok=True)
        meta_path = folder / "batch.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing batch metadata: {meta_path}")

        try:
            meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to read batch metadata: {e}")

        batch_id = cast(Optional[str], meta_obj.get("batch_id"))
        if not batch_id:
            raise ValueError("batch.json missing 'batch_id'")

        # Retrieve batch with retry
        batch: Any | None = None
        for attempt in self._retrying(max_attempts=max_attempts):
            with attempt:
                batch = cast(
                    litellm.LiteLLMBatch,
                    litellm.retrieve_batch(batch_id=batch_id, **self._client_kwargs),
                )
        if batch is None:
            raise RuntimeError("Failed to retrieve batch after retries")

        # Update metadata with latest status
        meta_obj.update(
            {
                "batch_id": cast(str, getattr(batch, "id", batch_id)),
                "endpoint": self.endpoint,
                "status": getattr(batch, "status", None),
                "output_file_id": getattr(batch, "output_file_id", None),
                "updated_at": time.time(),
            }
        )
        try:
            meta_path.write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")
        except Exception:
            pass

        # If not completed, nothing to download yet
        if getattr(batch, "status", None) != "completed":
            return batch, []

        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            return batch, []

        # Download file content with retry
        content: Any | None = None
        for attempt in self._retrying(max_attempts=max_attempts):
            with attempt:
                content = litellm.file_content(file_id=cast(str, output_file_id), **self._client_kwargs)
        if content is None:
            raise RuntimeError("Failed to download output file after retries")

        raw_bytes: bytes
        if hasattr(content, "read"):
            raw_bytes = content.read()  # type: ignore[assignment]
        else:
            raw_bytes = cast(bytes, content)  # type: ignore[assignment]

        out_path = folder / "outputs.ndjson"
        try:
            out_path.write_bytes(raw_bytes)
        except Exception:
            pass

        # Parse NDJSON into a list of dicts
        outputs: List[Dict[str, Any]] = []
        for line in raw_bytes.decode("utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                outputs.append(json.loads(s))
            except Exception:
                outputs.append({"raw": s})
        return batch, outputs




