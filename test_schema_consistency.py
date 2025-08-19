#!/usr/bin/env python3
"""
Unit test to ensure annotator tool schemas match rubric dataclass fields.
"""

import inspect
from typing import get_type_hints
from critic_rubrics import SolvabilityAnnotator, TrajectoryAnnotator
from critic_rubrics.rubrics.solvability import SolvabilityRubrics
from critic_rubrics.rubrics.trajectory import TrajectoryRubrics
from critic_rubrics.core import Prediction


def get_prediction_fields(rubric_class):
    """Extract field names that are of type Prediction from a rubric dataclass."""
    type_hints = get_type_hints(rubric_class)
    prediction_fields = []
    
    for field_name, field_type in type_hints.items():
        if field_type == Prediction:
            prediction_fields.append(field_name)
    
    return set(prediction_fields)


def get_tool_schema_fields(annotator):
    """Extract field names from annotator's tool schema."""
    schema = annotator._get_tool_schema()
    properties = schema['function']['parameters']['properties']
    
    # For solvability: fields are nested objects with detected/rationale
    # For trajectory: fields might be flat _detected/_rationale pattern
    schema_fields = []
    
    for prop_name in properties.keys():
        if prop_name.endswith('_detected'):
            # Flat pattern: remove '_detected' suffix
            base_name = prop_name[:-9]
            schema_fields.append(base_name)
        elif prop_name in properties and isinstance(properties[prop_name], dict):
            # Check if it's a nested object with detected/rationale
            nested_props = properties[prop_name].get('properties', {})
            if 'detected' in nested_props and 'rationale' in nested_props:
                # This is a Prediction field
                schema_fields.append(prop_name)
        elif prop_name not in ['additional_notes']:  # Skip non-Prediction fields
            # Assume it's a direct field name
            schema_fields.append(prop_name)
    
    return set(schema_fields)


def test_solvability_schema_consistency():
    """Test that SolvabilityAnnotator schema matches SolvabilityRubrics fields."""
    print("Testing SolvabilityAnnotator schema consistency...")
    
    annotator = SolvabilityAnnotator()
    rubric_fields = get_prediction_fields(SolvabilityRubrics)
    schema_fields = get_tool_schema_fields(annotator)
    
    print(f"Rubric fields ({len(rubric_fields)}): {sorted(rubric_fields)}")
    print(f"Schema fields ({len(schema_fields)}): {sorted(schema_fields)}")
    
    missing_in_schema = rubric_fields - schema_fields
    extra_in_schema = schema_fields - rubric_fields
    
    if missing_in_schema:
        print(f"❌ Fields in rubric but missing in schema: {missing_in_schema}")
    
    if extra_in_schema:
        print(f"❌ Fields in schema but missing in rubric: {extra_in_schema}")
    
    if not missing_in_schema and not extra_in_schema:
        print("✅ SolvabilityAnnotator schema matches SolvabilityRubrics perfectly!")
        return True
    
    return False


def test_trajectory_schema_consistency():
    """Test that TrajectoryAnnotator schema matches TrajectoryRubrics fields."""
    print("\nTesting TrajectoryAnnotator schema consistency...")
    
    annotator = TrajectoryAnnotator()
    rubric_fields = get_prediction_fields(TrajectoryRubrics)
    schema_fields = get_tool_schema_fields(annotator)
    
    print(f"Rubric fields ({len(rubric_fields)}): {sorted(rubric_fields)}")
    print(f"Schema fields ({len(schema_fields)}): {sorted(schema_fields)}")
    
    missing_in_schema = rubric_fields - schema_fields
    extra_in_schema = schema_fields - rubric_fields
    
    if missing_in_schema:
        print(f"❌ Fields in rubric but missing in schema: {missing_in_schema}")
    
    if extra_in_schema:
        print(f"❌ Fields in schema but missing in rubric: {extra_in_schema}")
    
    if not missing_in_schema and not extra_in_schema:
        print("✅ TrajectoryAnnotator schema matches TrajectoryRubrics perfectly!")
        return True
    
    return False


def main():
    """Run all consistency tests."""
    print("=" * 60)
    print("SCHEMA CONSISTENCY TESTS")
    print("=" * 60)
    
    solvability_ok = test_solvability_schema_consistency()
    trajectory_ok = test_trajectory_schema_consistency()
    
    print("\n" + "=" * 60)
    if solvability_ok and trajectory_ok:
        print("🎉 ALL TESTS PASSED! Schemas are consistent with rubrics.")
        return 0
    else:
        print("❌ TESTS FAILED! Schemas do not match rubrics.")
        return 1


if __name__ == "__main__":
    exit(main())