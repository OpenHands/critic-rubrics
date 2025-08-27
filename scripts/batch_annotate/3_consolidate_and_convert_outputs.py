#!/usr/bin/env python3
"""
Consolidate and convert batch annotation outputs to validated JSON structure.

Usage: python 3_consolidate_and_convert_outputs.py <batch_folder>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Generator, Iterator

from litellm.types.utils import ModelResponse

from critic_rubrics.rubrics.trajectory import annotate_conversation_rubrics, annotate_conversation_with_user_rubrics


def load_batch_data(batch_folder: Path) -> Generator[Dict[str, Any], None, None]:
    """Load and extract tool calls from all batch files in one pass."""
    batch_files = sorted(batch_folder.glob("batch_*_outputs.jsonl"))
    if not batch_files:
        print(f"No batch files found in {batch_folder}")
        return
    
    print(f"Processing {len(batch_files)} batch files...")
    
    for batch_file in batch_files:
        with open(batch_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    output = json.loads(line)
                    
                    # Extract tool calls path: response.body.choices[0].message.tool_calls
                    response = output.get("response", {})
                    if response.get("status_code") != 200:
                        continue
                    
                    body = response.get("body", {})
                    yield {
                        "batch_id": output.get("id", "unknown"),
                        "custom_id": output.get("custom_id", "unknown"),
                        "response": body,
                        "usage": body.get("usage", {}),
                        "model": body.get("model", "unknown")
                    }
                    
                except json.JSONDecodeError:
                    continue

def process_batch_data(batch_folder: Path) -> Iterator[Dict[str, Any]]:
    """Process batch data and convert to feature data."""
    for data in load_batch_data(batch_folder):
        model_response = ModelResponse(**data["response"])
        assert model_response.choices is not None and len(model_response.choices) > 0
        tool_calls = model_response.choices[0].message.tool_calls  # type: ignore
        assert tool_calls is not None and len(tool_calls) == 1
        tool_call = tool_calls[0]

        if annotate_conversation_rubrics.tool_call_match_rubrics(tool_call):
            feature_data_list = annotate_conversation_rubrics.tool_call_to_feature_data(tool_call)    
        else:
            feature_data_list = annotate_conversation_with_user_rubrics.tool_call_to_feature_data(tool_call)

        if not feature_data_list:
            print(f"No feature data extracted for batch_id {data['batch_id']}")
            continue
        features = {
            fd.feature.name: fd.prediction.to_dict()
            for fd in feature_data_list
        }
        
        yield {
            "batch_id": data["batch_id"],
            "custom_id": data["custom_id"],
            "model": data["model"],
            "usage": data["usage"],
            "features": features,
            "feature_count": len(features)
        }
            

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Consolidate batch annotation outputs")
    parser.add_argument("batch_folder", type=Path, help="Folder containing batch_*_outputs.jsonl files")
    parser.add_argument("--output-name", default="output.jsonl", help="Output file name")
    
    args = parser.parse_args()
    
    if not args.batch_folder.is_dir():
        print(f"Error: {args.batch_folder} is not a directory")
        sys.exit(1)
    
    # Process data and collect results
    results = list(process_batch_data(args.batch_folder))
    if not results:
        print("No results processed")
        sys.exit(1)
    
    # Save results
    output_file = args.batch_folder / args.output_name
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    # Print summary
    total_features = sum(r["feature_count"] for r in results)
    feature_types = {}
    for result in results:
        for feature_data in result["features"].values():
            feature_type = feature_data["type"]
            feature_types[feature_type] = feature_types.get(feature_type, 0) + 1
    
    print(f"\n✅ Saved {len(results)} results to {output_file}")
    print(f"Total features: {total_features} (avg: {total_features/len(results):.1f})")
    print("Feature types:", ", ".join(f"{k}: {v}" for k, v in sorted(feature_types.items())))


if __name__ == "__main__":
    main()
