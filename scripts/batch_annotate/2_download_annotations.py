#!/usr/bin/env python3
"""Download annotation results from LiteLLM batch API.

Checks batch status and downloads completed results.
"""

import argparse
import json
import os
import time
from pathlib import Path

import rich
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from critic_rubrics import Annotator


def load_batch_info(batch_dir: Path):
    """Load batch information from a batch directory."""
    batch_files = sorted(batch_dir.glob("batch_*.json"))
    batches = []

    for batch_file in batch_files:
        with batch_file.open("r") as f:
            batch_info = json.load(f)
            batch_info["file"] = batch_file
            batches.append(batch_info)

    return batches


def main():
    parser = argparse.ArgumentParser(description="Download annotation results from LiteLLM batch API")

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--batch-dir",
        type=str,
        help="Directory containing batch metadata files",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save results (defaults to batch-dir)",
    )

    # LiteLLM options
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
        help="LiteLLM proxy base URL",
        default="https://llm-proxy.eval.all-hands.dev",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for LiteLLM proxy (or use LITELLM_API_KEY env var)",
    )

    # Polling options
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Poll for completion if batches are not ready",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=2,
        help="Polling interval in seconds (default: 2)",
    )

    args = parser.parse_args()

    # Get API key from env if not provided
    api_key = args.api_key or os.environ.get("LITELLM_API_KEY")
    if not api_key:
        rich.print("[bold red]Error: API key required (--api-key or LITELLM_API_KEY env var)[/bold red]")
        return 1

    # Determine batch IDs to process
    batch_ids = []
    output_dir = batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        rich.print(f"[bold red]Batch directory not found: {batch_dir}[/bold red]")
        return 1

    batches = load_batch_info(batch_dir)
    batch_ids = [(b["batch_id"], b["file"].stem) for b in batches]

    if not batch_ids:
        rich.print("[bold red]No batch IDs to process[/bold red]")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process batches
    rich.print(f"[bold green]Checking status of {len(batch_ids)} batch(es)...[/bold green]")

    poll_interval = args.poll_interval

    pending_batches = batch_ids.copy()
    completed_batches = []
    failed_batches = []
    progress_log = []  # Keep permanent record of progress

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        while pending_batches:
            # Check status of all pending batches
            still_pending = []

            for batch_id, batch_name in pending_batches:
                task = progress.add_task(f"Checking {batch_name}...", total=None)

                try:
                    status, results = Annotator.get_batch_results(
                        batch_id,
                        custom_llm_provider=args.model_provider,
                        base_url=args.base_url,
                        api_key=api_key,
                    )

                    if status["status"] == "completed":
                        # Save results
                        if status["error"]:
                            error_file = output_dir / f"{batch_name}_errors.jsonl"
                            with error_file.open("w") as f:
                                for error in results:
                                    f.write(json.dumps(error) + "\n")
                            msg = f"✗ {batch_name} - {len(results)} errors saved"
                            progress.update(task, description=f"[red]{msg}")
                            progress_log.append(f"[red]{msg}[/red]")
                        else:
                            output_file = output_dir / f"{batch_name}_results.jsonl"
                            with output_file.open("w") as f:
                                for result in results:
                                    f.write(json.dumps(result) + "\n")
                            msg = f"✓ {batch_name} - {len(results)} results saved"
                            progress.update(task, description=f"[green]{msg}")
                            progress_log.append(f"[green]{msg}[/green]")

                        completed_batches.append((batch_id, batch_name, len(results)))

                    elif status["status"] in ["failed", "expired", "cancelled"]:
                        failed_batches.append((batch_id, batch_name, status["status"]))
                        msg = f"✗ {batch_name} - {status['status']}"
                        progress.update(task, description=f"[red]{msg}")
                        progress_log.append(f"[red]{msg}[/red]")

                    else:
                        still_pending.append((batch_id, batch_name))
                        msg = f"⏳ {batch_name} - {status['status']}"
                        progress.update(task, description=f"[yellow]{msg}")
                        progress_log.append(f"[yellow]{msg}[/yellow]")

                except Exception as e:
                    failed_batches.append((batch_id, batch_name, str(e)))
                    msg = f"✗ {batch_name} - Error: {e}"
                    progress.update(task, description=f"[red]{msg}")
                    progress_log.append(f"[red]{msg}[/red]")

                progress.remove_task(task)

            pending_batches = still_pending

            if pending_batches and args.poll:
                wait_task = progress.add_task(f"Waiting {poll_interval}s before next check...", total=None)
                time.sleep(poll_interval)
                progress.remove_task(wait_task)
            else:
                break

    # Print permanent progress log
    if progress_log:
        rich.print("\n[bold blue]Processing Log:[/bold blue]")
        for log_entry in progress_log:
            rich.print(f"  {log_entry}")

    # Print summary
    table = Table(title="Batch Processing Summary")
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Details")

    if completed_batches:
        total_results = sum(count for _, _, count in completed_batches)
        details = f"{total_results} total results"
        table.add_row("[green]Completed", str(len(completed_batches)), details)

    if failed_batches:
        reasons = {}
        for _, _, reason in failed_batches:
            reasons[reason] = reasons.get(reason, 0) + 1
        details = ", ".join(f"{r}: {c}" for r, c in reasons.items())
        table.add_row("[red]Failed", str(len(failed_batches)), details)

    if pending_batches:
        table.add_row("[yellow]Pending", str(len(pending_batches)), "Still processing")

    rich.print(table)

    # Save summary
    summary_file = output_dir / "download_summary.json"
    summary = {
        "completed": [{"id": id, "name": name, "results": count} for id, name, count in completed_batches],
        "failed": [{"id": id, "name": name, "reason": reason} for id, name, reason in failed_batches],
        "pending": [{"id": id, "name": name} for id, name in pending_batches],
        "timestamp": time.time(),
    }
    summary_file.write_text(json.dumps(summary, indent=2))

    rich.print(f"\n[bold green]Summary saved to: {summary_file}[/bold green]")

    if completed_batches:
        rich.print(f"[bold green]Results saved to: {output_dir}[/bold green]")

    return 0 if not failed_batches else 1


if __name__ == "__main__":
    exit(main())
