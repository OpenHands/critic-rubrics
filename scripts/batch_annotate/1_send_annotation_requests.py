#!/usr/bin/env python3
"""Send annotation requests to LiteLLM batch API.

Reads annotation requests from files and sends them as batches to LiteLLM.
"""

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Iterable

import rich
from litellm import ChatCompletionRequest

from critic_rubrics import Annotator
from critic_rubrics.rubrics import get_trajectory_level_rubrics


def open_shard_fn(path: Path):
    """Open a file, handling gzip compression if needed."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def generate_requests_from_traces(trace_dir: Path, pattern: str, limit: int | None = None):
    """Generate annotation requests from trace files."""
    shards = sorted(trace_dir.glob(pattern))
    if not shards:
        rich.print(f"[bold red]No shards matched pattern {pattern} under {trace_dir}[/bold red]")
        return

    n_processed = 0
    for shard in shards:
        rich.print(f"[bold blue]Processing shard:[/bold blue] {shard.name}")
        with open_shard_fn(shard) as f:
            for line in f:
                data = json.loads(line)
                conversation_id = data["conversation_id"]
                segment_id = data["segment_id"]
                trace_segment = data["trace_segment"]

                assert "follow_up_user_message" in trace_segment
                has_user_follow_up = trace_segment["follow_up_user_message"] is not None
                rubric = get_trajectory_level_rubrics(has_user_follow_up=has_user_follow_up)

                if has_user_follow_up:
                    messages = trace_segment["trace"] + [trace_segment["follow_up_user_message"]]
                else:
                    messages = trace_segment["trace"]

                annotation_request = rubric.create_annotation_request(
                    inputs={
                        "messages": messages,
                        "tools": trace_segment["tools"],
                    }
                )
                if annotation_request is None:
                    rich.print(f"[yellow]Skipping {conversation_id}/{segment_id} (no tools)[/yellow]")
                    continue

                # Attach metadata
                if "metadata" not in annotation_request:
                    annotation_request["metadata"] = {}
                request_id = f"req__conv_{conversation_id}__seg_{segment_id}"
                annotation_request["metadata"]["custom_request_id"] = request_id

                yield annotation_request

                n_processed += 1
                if limit and n_processed >= limit:
                    rich.print(f"[bold green]Reached limit of {limit} instances[/bold green]")
                    return


def main():
    parser = argparse.ArgumentParser(description="Send annotation requests to LiteLLM batch API")

    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--trace-dir",
        type=str,
        help="Directory containing trace shards (for generating requests)",
        required=False,
    )

    # Input options
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl.gz",
        help="Glob pattern to match input files (default: *.jsonl.gz for traces, **/*.json for requests)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of requests to process (for testing)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save batch metadata",
    )

    # LiteLLM options
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use for annotations (e.g., openai/o3-2025-04-16)",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        required=True,
        choices=["openai", "azure", "vertex_ai"],
        help="Model provider to use (e.g., openai, azure, vertex_ai)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://llm-proxy.eval.all-hands.dev",
        help="LiteLLM proxy base URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for LiteLLM proxy (or use LITELLM_API_KEY env var)",
    )

    # Batch options
    parser.add_argument(
        "--max-requests",
        type=int,
        default=10_000,
        help="Maximum requests per batch",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=100 * 1024 * 1024,
        help="Maximum bytes per batch",
    )

    args = parser.parse_args()

    # Get API key from env if not provided
    api_key = args.api_key or os.environ.get("LITELLM_API_KEY")
    if not api_key:
        rich.print("[bold red]Error: API key required (--api-key or LITELLM_API_KEY env var)[/bold red]")
        return 1

    # Generate requests based on input source
    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        rich.print(f"[bold red]Trace directory not found: {trace_dir}[/bold red]")
        return 1

    requests = generate_requests_from_traces(trace_dir, args.pattern, args.limit)
    # Send batches
    rich.print("[bold green]Sending requests to LiteLLM batch API...[/bold green]")
    rich.print(f"  Output directory: {args.output_dir}")
    rich.print(f"  Model: {args.model or 'default'}")
    rich.print(f"  Base URL: {args.base_url or 'default'}")

    def modify_requests_generator(requests: Iterable[ChatCompletionRequest]) -> Iterable[ChatCompletionRequest]:
        for request in requests:
            if "o3" in args.model or "gpt-5" in args.model:
                if "temperature" in request:
                    request.pop("temperature")
            request['reasoning_effort'] = 'high'  # type: ignore
            yield request

    try:
        batch_ids = Annotator.batch_annotate(
            modify_requests_generator(requests),
            args.output_dir,
            model=args.model,
            custom_llm_provider=args.model_provider,
            base_url=args.base_url,
            api_key=api_key,
            max_requests=args.max_requests,
            max_bytes=args.max_bytes,
            delete_after_upload=False
        )

        rich.print(f"[bold green]Successfully created {len(batch_ids)} batch(es):[/bold green]")
        for i, batch_id in enumerate(batch_ids, 1):
            rich.print(f"  Batch {i}: {batch_id}")

        # Save batch IDs for easy reference
        batch_ids_file = Path(args.output_dir) / "batch_ids.txt"
        batch_ids_file.write_text("\n".join(batch_ids))
        rich.print(f"[bold green]Batch IDs saved to: {batch_ids_file}[/bold green]")

    except Exception as e:
        rich.print(f"[bold red]Error sending batches: {e}[/bold red]")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
