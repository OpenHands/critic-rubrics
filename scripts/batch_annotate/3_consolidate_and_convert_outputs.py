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

from litellm.types.utils import Choices, Message, ModelResponse

from critic_rubrics.prediction import BinaryPrediction, ClassificationPrediction, TextPrediction
from critic_rubrics.rubrics.trajectory import annotate_conversation_with_user_rubrics


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
                    choices = body.get("choices", [])
                    if not choices:
                        continue
                    
                    tool_calls = choices[0].get("message", {}).get("tool_calls", [])
                    if not tool_calls:
                        continue
                    
                    yield {
                        "batch_id": output.get("id", "unknown"),
                        "custom_id": output.get("custom_id", "unknown"),
                        "tool_calls": tool_calls,
                        "usage": body.get("usage", {}),
                        "model": body.get("model", "unknown")
                    }
                    
                except json.JSONDecodeError:
                    continue


def serialize_prediction(prediction: Any) -> Dict[str, Any]:
    """Convert prediction to dict using proper type checking."""
    if isinstance(prediction, BinaryPrediction):
        return {
            "type": "binary",
            "detected": prediction.detected,
            "rationale": prediction.rationale
        }
    elif isinstance(prediction, ClassificationPrediction):
        return {
            "type": "classification", 
            "label": prediction.label,
            "rationale": prediction.rationale
        }
    elif isinstance(prediction, TextPrediction):
        return {
            "type": "text",
            "text": prediction.text
        }
    else:
        return {
            "type": "unknown",
            "value": str(prediction),
            "rationale": getattr(prediction, 'rationale', '')
        }


def process_batch_data(batch_folder: Path) -> Iterator[Dict[str, Any]]:
    """Process batch data and convert to feature data."""
    for data in load_batch_data(batch_folder):
        try:
            # Convert tool calls to ModelResponse
            model_response = ModelResponse(
                choices=[Choices(message=Message(tool_calls=data["tool_calls"]))]
            )
            
            # Extract features using rubrics
            feature_data_list = annotate_conversation_with_user_rubrics.model_response_to_feature_data(model_response)
            if not feature_data_list:
                continue
            
            # Convert to serializable format
            features = {
                fd.feature.name: serialize_prediction(fd.prediction)
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
            
        except Exception:
            continue


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