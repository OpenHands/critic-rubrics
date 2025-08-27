"""Script to read trace shards, create annotation requests, and write them to files.

NOTE: this script does NOT perform the actual annotation (LLM calls). the primary purpose is to
prepare the annotation requests in a way that is easy to debug and inspect.
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import rich
from litellm import ChatCompletionRequest

from critic_rubrics.rubrics import AnnotateConversationRubric
from critic_rubrics.rubrics.trajectory import get_trajectory_level_rubrics


def open_shard_fn(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    else:
        return path.open("r", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, default="./data/compressed_traces", help="Directory containing trace shards")
    parser.add_argument("--pattern", type=str, default="*.jsonl.gz", help="Glob pattern to match shards")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--limit", type=int, default=100, help="Max size of instances to process (for testing)")
    parser.add_argument("--output_dir", type=str, default="./data/trace_with_rubrics", help="Directory to write annotated outputs")
    args = parser.parse_args()

    TRACE_DIR = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    shards = sorted(TRACE_DIR.glob(args.pattern))
    if not shards:
        print(f"No shards matched pattern {args.pattern} under {TRACE_DIR}")
        return

    n_processed = 0
    for shard in shards:
        print(f",  Enqueuing traces from {shard.name}…")
        with open_shard_fn(shard) as f:
            for line in f:
                data = json.loads(line)
                assert "conversation_id" in data
                assert "segment_id" in data
                conversation_id = data["conversation_id"]
                segment_id = data["segment_id"]
                assert "trace_segment" in data
                trace_segment = data["trace_segment"]
                messages: list[dict[str, Any]]
                has_user_follow_up: bool = trace_segment.follow_up_user_message is not None
                
                rubric: AnnotateConversationRubric = get_trajectory_level_rubrics(has_user_follow_up=has_user_follow_up)
                if has_user_follow_up:
                    assert trace_segment.follow_up_user_message is not None
                    messages = trace_segment.trace + [trace_segment.follow_up_user_message]
                else:
                    messages = trace_segment.trace

                annotation_request: ChatCompletionRequest | None = rubric.create_annotation_request(
                    inputs={
                        "messages": messages,
                        "tools": trace_segment.tools,
                    }
                )

                if annotation_request is None:
                    rich.print(f"[bold yellow]Skipping annotation for conversation ID:[/bold yellow] {conversation_id} | [bold yellow]Segment ID:[/bold yellow] {segment_id} (no tools)")
                    continue

                output_filepath = output_dir / f"conv_{conversation_id}" / f"{segment_id}_followup_{str(has_user_follow_up).lower()}.json"
                output_filepath.parent.mkdir(parents=True, exist_ok=True)
                with output_filepath.open("w", encoding="utf-8") as out_f:
                    json.dump(annotation_request, out_f, indent=2)
                rich.print(
                    (
                        f"[bold blue]Conversation ID:[/bold blue] {conversation_id} | "
                        f"[bold green]Segment ID:[/bold green] {segment_id:>3} | "
                        f"[bold yellow]Has follow-up user?[/bold yellow] {str(has_user_follow_up):<5} | "
                        f"[bold magenta]Tools:[/bold magenta] {len(trace_segment.tools)}"
                    )
                )
                # rich.print(annotation_request)

                n_processed += 1
                if n_processed >= args.limit:
                    rich.print(f"Reached limit of {args.limit} instances, stopping.")
                    return


if __name__ == "__main__":
    main()
