"""
Example usage of the unified rubrics system.
"""

import os
from typing import List, Dict, Any

from .annotators import (
    create_solvability_annotator,
    create_conversation_annotator,
    create_multi_sample_solvability_annotator,
    CustomAnnotator
)
from .rubrics import create_custom_rubric
from .core import RubricItem, RubricCategory


def example_solvability_analysis():
    """Example: Analyze issue solvability."""
    
    # Create annotator
    annotator = create_solvability_annotator(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Example issue
    issue_text = """
    Bug: Login form doesn't work on mobile Safari
    
    When I try to log in using Safari on my iPhone 13, the login button doesn't respond when tapped.
    This works fine on desktop Chrome and Firefox.
    
    Steps to reproduce:
    1. Open the app in Safari on iPhone
    2. Enter valid credentials
    3. Tap the login button
    4. Nothing happens
    
    Expected: Should log in successfully
    Actual: Button doesn't respond, no error message
    
    Environment: iPhone 13, iOS 16.2, Safari
    """
    
    # Annotate
    result = annotator.annotate(issue_text)
    
    print("Solvability Analysis Results:")
    print(f"Detected features: {result.get_detected_items()}")
    print(f"Detection rate: {result.get_detection_rate():.2%}")
    print(f"Tokens used: {result.prompt_tokens} prompt, {result.completion_tokens} completion")
    
    return result


def example_conversation_analysis():
    """Example: Analyze agent-user conversation."""
    
    # Create annotator
    annotator = create_conversation_annotator(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Example conversation
    conversation_text = """
    User: Can you help me create a Python script to parse CSV files?
    
    Agent: I'll create a CSV parser script for you. Let me start by creating a basic parser:
    
    ```python
    import csv
    
    def parse_csv(filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                print(row)
    
    parse_csv('data.csv')
    ```
    
    This script will read and print all rows from a CSV file.
    
    User: That's not what I need. I need to parse CSV files with headers and convert them to JSON format, not just print them.
    """
    
    # Annotate
    result = annotator.annotate(conversation_text)
    
    print("\nConversation Analysis Results:")
    print(f"Follow-up timing: {result.additional_data.get('follow_up_timing')}")
    print(f"Task type: {result.additional_data.get('task_type')}")
    print(f"Detected issues: {result.get_detected_items()}")
    
    # Print rationales for detected issues
    for item in result.get_detected_items():
        rationale = result.rationales.get(item, "No rationale provided")
        print(f"  {item}: {rationale}")
    
    return result


def example_multi_sample_analysis():
    """Example: Multi-sample analysis for robust results."""
    
    # Create multi-sample annotator
    annotator = create_multi_sample_solvability_annotator(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Example issue (ambiguous case)
    issue_text = """
    Feature request: Better error handling
    
    The application should handle errors better. Sometimes it crashes and sometimes it shows weird messages.
    Can you improve this?
    """
    
    # Generate multiple samples
    multi_result = annotator.annotate_with_samples(
        issue_text, 
        samples=5, 
        temperature=0.7
    )
    
    print("\nMulti-Sample Analysis Results:")
    print(f"Total samples: {len(multi_result.samples)}")
    
    # Show detection rates
    detection_rates = multi_result.get_detection_rates()
    print("Detection rates across samples:")
    for feature, rate in detection_rates.items():
        print(f"  {feature}: {rate:.1%}")
    
    # Show entropy (measure of uncertainty)
    entropies = multi_result.get_detection_entropy()
    print("Detection entropy (uncertainty):")
    for feature, entropy in entropies.items():
        print(f"  {feature}: {entropy:.3f}")
    
    # Get consensus result
    consensus = multi_result.get_consensus_result(threshold=0.6)
    print(f"Consensus detections (60% threshold): {consensus.get_detected_items()}")
    
    return multi_result


def example_custom_rubric():
    """Example: Create and use a custom rubric."""
    
    # Define custom rubric items
    code_quality_items = [
        RubricItem(
            identifier="has_tests",
            description="The code includes unit tests or test cases"
        ),
        RubricItem(
            identifier="has_documentation",
            description="The code includes docstrings or comments explaining functionality"
        ),
        RubricItem(
            identifier="follows_pep8",
            description="The code follows PEP 8 style guidelines"
        ),
        RubricItem(
            identifier="has_error_handling",
            description="The code includes appropriate error handling (try/except blocks)"
        ),
        RubricItem(
            identifier="uses_type_hints",
            description="The code uses Python type hints for function parameters and return values"
        ),
    ]
    
    # Create custom rubric
    code_quality_rubric = create_custom_rubric(
        name="code_quality",
        description="Rubric for evaluating Python code quality",
        items=code_quality_items,
        additional_fields={
            "complexity_level": {
                "type": "string",
                "enum": ["simple", "moderate", "complex"],
                "description": "Overall complexity level of the code",
                "required": True
            }
        }
    )
    
    # Create custom annotator
    annotator = CustomAnnotator(
        rubric_set=code_quality_rubric,
        system_prompt="You are a code quality analyzer. Evaluate the provided Python code for quality characteristics.",
        instruction_prompt="Analyze the code and identify which quality characteristics are present.",
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Example code to analyze
    code_text = '''
    def calculate_average(numbers: List[float]) -> float:
        """
        Calculate the average of a list of numbers.
        
        Args:
            numbers: List of numbers to average
            
        Returns:
            The average value
            
        Raises:
            ValueError: If the list is empty
        """
        if not numbers:
            raise ValueError("Cannot calculate average of empty list")
        
        return sum(numbers) / len(numbers)
    
    
    # Test the function
    def test_calculate_average():
        assert calculate_average([1, 2, 3, 4, 5]) == 3.0
        assert calculate_average([10]) == 10.0
        
        try:
            calculate_average([])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
    '''
    
    # Annotate
    result = annotator.annotate(code_text)
    
    print("\nCustom Code Quality Analysis:")
    print(f"Complexity level: {result.additional_data.get('complexity_level')}")
    print(f"Quality features detected: {result.get_detected_items()}")
    
    # Print rationales
    for item in result.get_detected_items():
        rationale = result.rationales.get(item, "No rationale provided")
        print(f"  {item}: {rationale}")
    
    return result


def example_batch_processing():
    """Example: Process multiple items in batch."""
    
    annotator = create_solvability_annotator(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Multiple issues to analyze
    issues = [
        "Bug: App crashes when clicking save button. No error message shown.",
        "Feature request: Add dark mode support to the application.",
        "Issue: Database connection timeout after 30 seconds. Need to increase timeout or add retry logic.",
        "Bug: User profile images not loading on mobile devices. Works fine on desktop.",
    ]
    
    # Process in batch
    results = annotator.annotate_batch(issues, max_workers=2)
    
    print("\nBatch Processing Results:")
    for i, result in enumerate(results):
        print(f"Issue {i+1}: {len(result.get_detected_items())} features detected")
        print(f"  Detection rate: {result.get_detection_rate():.1%}")
    
    # Aggregate statistics
    total_detections = sum(len(r.get_detected_items()) for r in results)
    avg_detection_rate = sum(r.get_detection_rate() for r in results) / len(results)
    
    print(f"Total detections across all issues: {total_detections}")
    print(f"Average detection rate: {avg_detection_rate:.1%}")
    
    return results


def run_all_examples():
    """Run all examples."""
    print("=== Unified Rubrics Examples ===\n")
    
    try:
        example_solvability_analysis()
        example_conversation_analysis()
        example_multi_sample_analysis()
        example_custom_rubric()
        example_batch_processing()
        
        print("\n=== All examples completed successfully! ===")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set OPENAI_API_KEY environment variable")


if __name__ == "__main__":
    run_all_examples()