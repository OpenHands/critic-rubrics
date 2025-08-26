from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Tuple, cast

import litellm
from litellm import ChatCompletionRequest, LiteLLMBatch, OpenAIFileObject, completion
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential


class Annotator:
    """Simple LiteLLM-backed annotator.

    - annotate: send a single ChatCompletionRequest to a LiteLLM proxy in real time (with retry)
    - batch_annotate: submit a batch of ChatCompletionRequests via LiteLLM Batches API (with retry for create/upload)
    - batch_annotate_chunked: split requests into multiple batches by limits and persist per-batch metadata
    - download_annotation: given a batch_id, download outputs to a results folder
    - download_annotations: aggregate downloads for multiple per-batch folders
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
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
    # Batch processing helpers
    # -----------------------
    def _create_single_batch_from_lines(
        self,
        *,
        lines: List[str],
        results_subdir: Path,
        completion_window: Literal["24h"],
        max_attempts: int,
    ) -> LiteLLMBatch:
        results_subdir.mkdir(parents=True, exist_ok=True)
        batch_metadata_path = results_subdir / "batch.json"
        if batch_metadata_path.exists():
            raise FileExistsError(
                f"Batch metadata file already exists: {batch_metadata_path}. Cannot create a new batch in the same folder."
            )

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname) / "inputs.jsonl"
            tmp_path.write_text("\n".join(lines), encoding="utf-8")

            file_obj: OpenAIFileObject | None = None
            for attempt in self._retrying(max_attempts=max_attempts):
                with attempt:
                    with open(tmp_path, "rb") as file_like:
                        file_obj = cast(
                            OpenAIFileObject,
                            litellm.create_file(file=file_like, purpose="batch", **self._client_kwargs),
                        )
            if file_obj is None:
                raise RuntimeError("Failed to create file for batch input after retries")

        batch: LiteLLMBatch | None = None
        for attempt in self._retrying(max_attempts=max_attempts):
            with attempt:
                batch = cast(
                    LiteLLMBatch,
                    litellm.create_batch(
                        completion_window=completion_window,
                        endpoint=self.endpoint,
                        input_file_id=file_obj.id,
                        **self._client_kwargs,
                    ),
                )
        if batch is None:
            raise RuntimeError("Failed to create batch after retries")

        meta = {
            "batch_id": cast(str, getattr(batch, "id", None)),
            "input_file_id": cast(str, getattr(file_obj, "id", None)),
            "endpoint": self.endpoint,
            "status": getattr(batch, "status", None),
            "created_at": time.time(),
            "request_count": len(lines),
        }
        try:
            batch_metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass
        return batch

    # -----------------------
    # Single-batch API (back-compat)
    # -----------------------
    def batch_annotate(
        self,
        requests: Iterable[ChatCompletionRequest],
        results_dir: str | Path,
        completion_window: Literal["24h"] = "24h",
        max_attempts: int = 3,
    ) -> LiteLLMBatch:
        """Submit a batch to LiteLLM Proxy and persist identifiers.

        Backward-compatible API: packs all provided requests into a single batch and writes
        metadata to results_dir/batch.json. Use batch_annotate_chunked for large datasets.
        """
        folder_path = Path(results_dir)
        folder_path.mkdir(parents=True, exist_ok=True)
        batch_metadata_path = folder_path / "batch.json"
        if batch_metadata_path.exists():
            raise FileExistsError(
                f"Batch metadata file already exists: {batch_metadata_path}. Cannot create a new batch."
            )

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

        return self._create_single_batch_from_lines(
            lines=lines,
            results_subdir=folder_path,
            completion_window=completion_window,
            max_attempts=max_attempts,
        )

    # -----------------------
    # Multi-batch API with limits
    # -----------------------
    def batch_annotate_chunked(
        self,
        requests: Iterable[ChatCompletionRequest],
        results_dir: str | Path,
        *,
        completion_window: Literal["24h"] = "24h",
        max_attempts: int = 3,
        max_requests_per_batch: int | None = 50_000,
        max_bytes_per_batch: int | None = 200 * 1024 * 1024,
        custom_id_prefix: str = "req_",
    ) -> List[LiteLLMBatch]:
        """Split requests into multiple batches, respecting limits.

        - results_dir structure:
          results_dir/
            manifest.json
            batches/
              batch_000001/
                batch.json
                outputs.ndjson (when downloaded)
              batch_000002/
                ...

        Returns a list of LiteLLMBatch objects (one per created batch).
        """
        root = Path(results_dir)
        batches_dir = root / "batches"
        batches_dir.mkdir(parents=True, exist_ok=True)

        created_batches: List[LiteLLMBatch] = []
        manifest = {
            "endpoint": self.endpoint,
            "model": self.model,
            "limits": {
                "max_requests_per_batch": max_requests_per_batch,
                "max_bytes_per_batch": max_bytes_per_batch,
            },
            "created_at": time.time(),
            "batches": [],  # filled as we create batches
        }

        def flush_batch(batch_idx: int, lines: List[str]) -> None:
            if not lines:
                return
            subdir = batches_dir / f"batch_{batch_idx:06d}"
            batch = self._create_single_batch_from_lines(
                lines=lines,
                results_subdir=subdir,
                completion_window=completion_window,
                max_attempts=max_attempts,
            )
            created_batches.append(batch)
            manifest["batches"].append(
                {
                    "batch_index": batch_idx,
                    "batch_dir": str(subdir),
                    "batch_id": cast(str, getattr(batch, "id", None)),
                    "status": getattr(batch, "status", None),
                    "request_count": len(lines),
                }
            )

        current_lines: List[str] = []
        current_bytes = 0
        current_count = 0
        batch_index = 1
        global_i = 0

        for req in requests:
            body: Dict[str, Any] = dict(req)
            if "model" not in body:
                if self.model is None:
                    raise ValueError("ChatCompletionRequest missing 'model' and no default model set")
                body["model"] = self.model
            line_obj = {
                "custom_id": f"{custom_id_prefix}{global_i:08d}",
                "method": "POST",
                "url": self.endpoint,
                "body": body,
            }
            line_str = json.dumps(line_obj, separators=(",", ":"))
            line_bytes = len(line_str.encode("utf-8"))
            overhead = 1 if current_lines else 0  # newline between lines

            would_bytes = current_bytes + overhead + line_bytes
            would_count = current_count + 1

            if (
                (max_requests_per_batch is not None and would_count > max_requests_per_batch)
                or (max_bytes_per_batch is not None and would_bytes > max_bytes_per_batch)
            ):
                if not current_lines:
                    # Single line exceeds limits; refuse with a clear error
                    raise ValueError(
                        "Single request exceeds configured batch size limits; lower request payload or increase limits"
                    )
                flush_batch(batch_index, current_lines)
                batch_index += 1
                current_lines = []
                current_bytes = 0
                current_count = 0
                overhead = 0
                would_bytes = line_bytes
                would_count = 1

            current_lines.append(line_str)
            current_bytes = would_bytes
            current_count = would_count
            global_i += 1

        if current_lines:
            flush_batch(batch_index, current_lines)

        # Persist manifest at root
        manifest_path = root / "manifest.json"
        try:
            manifest["num_batches"] = len(created_batches)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception:
            pass

        return created_batches

    # -----------------------
    # Download helpers
    # -----------------------
    def download_annotation(
        self,
        results_dir: str | Path,
        *,
        max_attempts: int = 3,
    ) -> Tuple[litellm.LiteLLMBatch, List[Dict[str, Any]]]:
        """Fetch latest batch status and download outputs into results_dir if ready.

        Expects results_dir to contain a single batch.json. For multi-batch runs, see download_annotations.
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

        batch_id = cast(str | None, meta_obj.get("batch_id"))
        if not batch_id:
            raise ValueError("batch.json missing 'batch_id'")

        batch: Any | None = None
        for attempt in self._retrying(max_attempts=max_attempts):
            with attempt:
                batch = cast(
                    litellm.LiteLLMBatch,
                    litellm.retrieve_batch(batch_id=batch_id, **self._client_kwargs),
                )
        if batch is None:
            raise RuntimeError("Failed to retrieve batch after retries")

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

        if getattr(batch, "status", None) != "completed":
            return batch, []

        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            return batch, []

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

    def download_annotations(
        self,
        results_dir: str | Path,
        *,
        max_attempts: int = 3,
    ) -> Tuple[List[litellm.LiteLLMBatch], List[Dict[str, Any]]]:
        """Aggregate download for a multi-batch results directory.

        Looks for results_dir/manifest.json and results_dir/batches/*/batch.json. For each batch
        folder, calls download_annotation and concatenates outputs.
        Returns (batches, outputs).
        """
        root = Path(results_dir)
        batches_root = root / "batches"
        if not batches_root.exists():
            # Fallback: treat root as single-batch folder
            single_batch, outputs = self.download_annotation(root, max_attempts=max_attempts)
            return [single_batch], outputs

        all_batches: List[litellm.LiteLLMBatch] = []
        all_outputs: List[Dict[str, Any]] = []
        for sub in sorted(batches_root.iterdir()):
            if not sub.is_dir():
                continue
            meta = sub / "batch.json"
            if not meta.exists():
                continue
            batch, outputs = self.download_annotation(sub, max_attempts=max_attempts)
            all_batches.append(batch)
            all_outputs.extend(outputs)
        return all_batches, all_outputs




