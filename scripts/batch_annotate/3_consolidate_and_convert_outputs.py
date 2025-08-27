#!/usr/bin/env python3
"""
Consolidate and convert batch annotation outputs to validated JSON structure.

This script:
1. Takes a batch folder containing batch_*_outputs.jsonl files
2. Imports the annotate_conversation_with_user_rubrics instance
3. Aggregates all batch output files
4. Uses the rubrics to convert tool calls to validated FeatureData
5. Saves the results to output.jsonl in the same folder

Usage:
    python 3_consolidate_and_convert_outputs.py <batch_folder>

Example:
    python 3_consolidate_and_convert_outputs.py data/annotated_traces/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, cast

from litellm.types.utils import Choices, Message, ModelResponse

from critic_rubrics.prediction import BinaryPrediction, ClassificationPrediction, TextPrediction
from critic_rubrics.rubrics.trajectory import annotate_conversation_with_user_rubrics


def find_batch_output_files(batch_folder: Path) -> List[Path]:
    """Find all batch_*_outputs.jsonl files in the given folder."""
    pattern = "batch_*_outputs.jsonl"
    batch_files = list(batch_folder.glob(pattern))
    
    if not batch_files:
        print(f"No batch output files found matching pattern '{pattern}' in {batch_folder}")
        return []
    
    # Sort files for consistent processing order
    batch_files.sort()
    print(f"Found {len(batch_files)} batch output files:")
    for file in batch_files:
        print(f"  - {file.name}")
    
    return batch_files


def load_batch_outputs(batch_files: List[Path]) -> List[Dict[str, Any]]:
    """Load all batch output data from JSONL files."""
    all_outputs = []
    
    for batch_file in batch_files:
        print(f"Loading {batch_file.name}...")
        file_outputs = []
        
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        output = json.loads(line)
                        file_outputs.append(output)
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Failed to parse line {line_num} in {batch_file.name}: {e}")
                        continue
            
            print(f"  Loaded {len(file_outputs)} outputs from {batch_file.name}")
            all_outputs.extend(file_outputs)
            
        except Exception as e:
            print(f"  Error reading {batch_file.name}: {e}")
            continue
    
    print(f"Total outputs loaded: {len(all_outputs)}")
    return all_outputs


def extract_tool_calls_from_outputs(batch_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract tool calls from batch outputs with metadata."""
    extracted_data = []
    
    for output in batch_outputs:
        try:
            # Extract metadata
            batch_id = output.get("id", "unknown")
            custom_id = output.get("custom_id", "unknown")
            
            # Extract tool calls from response
            response = output.get("response", {})
            if response.get("status_code") != 200:
                print(f"  Skipping output {custom_id} with status code {response.get('status_code')}")
                continue
            
            body = response.get("body", {})
            choices = body.get("choices", [])
            
            if not choices:
                print(f"  No choices found in output {custom_id}")
                continue
            
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            
            if not tool_calls:
                print(f"  No tool calls found in output {custom_id}")
                continue
            
            # Store with metadata
            extracted_data.append({
                "batch_id": batch_id,
                "custom_id": custom_id,
                "tool_calls": tool_calls,
                "usage": body.get("usage", {}),
                "model": body.get("model", "unknown")
            })
            
        except Exception as e:
            print(f"  Error processing output: {e}")
            continue
    
    print(f"Extracted tool calls from {len(extracted_data)} outputs")
    return extracted_data


def convert_to_feature_data(extracted_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert tool calls to validated FeatureData using rubrics."""
    converted_results = []
    
    print("Converting tool calls to validated FeatureData...")
    
    for i, data in enumerate(extracted_data):
        try:
            tool_calls = data["tool_calls"]
            
            # Create a ModelResponse object to match the new method signature
            model_response = ModelResponse(
                choices=[
                    Choices(
                        message=Message(
                            tool_calls=tool_calls
                        )
                    )
                ]
            )
            
            # Use rubrics to convert ModelResponse to FeatureData
            feature_data_list = annotate_conversation_with_user_rubrics.model_response_to_feature_data(model_response)
            
            if not feature_data_list:
                print(f"  Warning: No features extracted from {data['custom_id']}")
                continue
            
            # Convert FeatureData objects to serializable format
            features_dict = {}
            for feature_data in feature_data_list:
                feature_name = feature_data.feature.name
                prediction = feature_data.prediction
                
                # Serialize prediction based on type
                if hasattr(prediction, 'detected'):
                    # BinaryPrediction
                    binary_pred = cast(BinaryPrediction, prediction)
                    prediction_dict = {
                        "type": "binary",
                        "detected": binary_pred.detected,
                        "rationale": binary_pred.rationale
                    }
                elif hasattr(prediction, 'label'):
                    # ClassificationPrediction
                    class_pred = cast(ClassificationPrediction, prediction)
                    prediction_dict = {
                        "type": "classification",
                        "label": class_pred.label,
                        "rationale": class_pred.rationale
                    }
                elif hasattr(prediction, 'text'):
                    # TextPrediction
                    text_pred = cast(TextPrediction, prediction)
                    prediction_dict = {
                        "type": "text",
                        "text": text_pred.text
                    }
                else:
                    # Unknown prediction type
                    prediction_dict = {
                        "type": "unknown",
                        "value": str(prediction),
                        "rationale": getattr(prediction, 'rationale', '')
                    }
                
                features_dict[feature_name] = prediction_dict
            
            # Create result entry
            result = {
                "batch_id": data["batch_id"],
                "custom_id": data["custom_id"],
                "model": data["model"],
                "usage": data["usage"],
                "features": features_dict,
                "feature_count": len(features_dict)
            }
            
            converted_results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(extracted_data)} outputs...")
                
        except Exception as e:
            print(f"  Error converting {data.get('custom_id', 'unknown')}: {e}")
            continue
    
    print(f"Successfully converted {len(converted_results)} outputs to FeatureData")
    return converted_results


def save_consolidated_output(results: List[Dict[str, Any]], output_file: Path) -> None:
    """Save consolidated results to output.jsonl."""
    print(f"Saving results to {output_file}...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Successfully saved {len(results)} results to {output_file}")
        
        # Print summary statistics
        if results:
            total_features = sum(r["feature_count"] for r in results)
            avg_features = total_features / len(results)
            
            print("\nSummary:")
            print(f"  Total outputs: {len(results)}")
            print(f"  Total features: {total_features}")
            print(f"  Average features per output: {avg_features:.1f}")
            
            # Count feature types
            feature_type_counts = {}
            for result in results:
                for feature_name, feature_data in result["features"].items():
                    feature_type = feature_data["type"]
                    feature_type_counts[feature_type] = feature_type_counts.get(feature_type, 0) + 1
            
            print("  Feature type distribution:")
            for feature_type, count in sorted(feature_type_counts.items()):
                print(f"    {feature_type}: {count}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Consolidate and convert batch annotation outputs to validated JSON structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 3_consolidate_and_convert_outputs.py data/annotated_traces/
  python 3_consolidate_and_convert_outputs.py /path/to/batch/folder/
        """
    )
    
    parser.add_argument(
        "batch_folder",
        type=Path,
        help="Path to folder containing batch_*_outputs.jsonl files"
    )
    
    parser.add_argument(
        "--output-name",
        default="output.jsonl",
        help="Name of output file (default: output.jsonl)"
    )
    
    args = parser.parse_args()
    
    # Validate batch folder
    if not args.batch_folder.exists():
        print(f"Error: Batch folder does not exist: {args.batch_folder}")
        sys.exit(1)
    
    if not args.batch_folder.is_dir():
        print(f"Error: Path is not a directory: {args.batch_folder}")
        sys.exit(1)
    
    print(f"Processing batch folder: {args.batch_folder}")
    print("Using rubrics: annotate_conversation_with_user_rubrics")
    print(f"Features available: {len(annotate_conversation_with_user_rubrics.features)}")
    print()
    
    # Step 1: Find batch output files
    batch_files = find_batch_output_files(args.batch_folder)
    if not batch_files:
        sys.exit(1)
    
    print()
    
    # Step 2: Load batch outputs
    batch_outputs = load_batch_outputs(batch_files)
    if not batch_outputs:
        print("No batch outputs loaded. Exiting.")
        sys.exit(1)
    
    print()
    
    # Step 3: Extract tool calls
    extracted_data = extract_tool_calls_from_outputs(batch_outputs)
    if not extracted_data:
        print("No tool calls extracted. Exiting.")
        sys.exit(1)
    
    print()
    
    # Step 4: Convert to FeatureData
    converted_results = convert_to_feature_data(extracted_data)
    if not converted_results:
        print("No results converted. Exiting.")
        sys.exit(1)
    
    print()
    
    # Step 5: Save consolidated output
    output_file = args.batch_folder / args.output_name
    save_consolidated_output(converted_results, output_file)
    
    print(f"\n✅ Consolidation complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()