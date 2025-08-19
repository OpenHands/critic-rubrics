"""
Example usage of the critic rubrics system.

This module demonstrates how to use the different annotators and rubric types
for analyzing issues, conversations, and trajectories.
"""

import os
from typing import List

from .annotators import SolvabilityAnnotator, TrajectoryAnnotator
from .rubrics import SolvabilityRubrics, TrajectoryRubrics


def example_solvability_analysis():
    """Example of analyzing issue solvability."""
    
    # Sample issue text
    issue_text = """
    Bug: Application crashes when loading large files
    
    When I try to load a CSV file larger than 100MB, the application crashes with a memory error.
    
    Steps to reproduce:
    1. Open the application
    2. Go to File -> Open
    3. Select a CSV file larger than 100MB
    4. Click Open
    
    Expected behavior: The file should load successfully
    Actual behavior: Application crashes with "MemoryError: Unable to allocate array"
    
    Environment:
    - OS: Windows 10
    - Python version: 3.9.7
    - Application version: 2.1.3
    - RAM: 8GB
    
    I've tried with multiple large files and the issue is consistent.
    """
    
    # Create annotator
    annotator = SolvabilityAnnotator(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Analyze the issue
    result = annotator.annotate(issue_text)
    
    print("=== Solvability Analysis Results ===")
    print(f"Detection rate: {result.get_detection_rate():.2%}")
    print(f"Detected features: {result.get_detected_features()}")
    print(f"Missing features: {result.get_missing_features()}")
    
    # Show some specific results
    print(f"\nClear problem statement: {result.has_clear_problem_statement.detected}")
    print(f"Rationale: {result.has_clear_problem_statement.rationale}")
    
    print(f"\nReproduction steps: {result.has_reproduction_steps.detected}")
    print(f"Rationale: {result.has_reproduction_steps.rationale}")
    
    return result


def example_trajectory_analysis():
    """Example of analyzing agent conversation trajectory."""
    
    # Sample conversation
    conversation_text = """
    User: I need help setting up a Python virtual environment for my project.
    
    Agent: I'll help you set up a Python virtual environment. Let me create one for you.
    
    ```bash
    python -m venv myproject
    source myproject/bin/activate
    ```
    
    User: I'm on Windows, will that work?
    
    Agent: Oh, you're right! On Windows, the activation command is different. Let me correct that:
    
    ```bash
    python -m venv myproject
    myproject\\Scripts\\activate
    ```
    
    User: Perfect! Now how do I install packages?
    
    Agent: Once your virtual environment is activated, you can install packages using pip:
    
    ```bash
    pip install package_name
    ```
    
    For example, to install requests:
    ```bash
    pip install requests
    ```
    
    You can also install from a requirements.txt file:
    ```bash
    pip install -r requirements.txt
    ```
    
    User: Great, that's exactly what I needed. Thank you!
    
    Agent: You're welcome! Your virtual environment is now set up and ready to use. Is there anything else you'd like to know about managing Python environments?
    
    User: No, that covers everything. Thanks again!
    """
    
    # Create annotator
    annotator = TrajectoryAnnotator(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Analyze the conversation
    result = annotator.annotate(conversation_text)
    
    print("=== Trajectory Analysis Results ===")
    print(f"Issue count: {result.get_issue_count()}")
    print(f"Positive indicators: {result.get_positive_indicators_count()}")
    print(f"Quality score: {result.get_quality_score():.2f}")
    print(f"User follow-up indicators: {result.get_user_followup_indicators()}")
    
    # Show some specific results
    print(f"\nTask completed successfully: {result.task_completed_successfully.detected}")
    print(f"Rationale: {result.task_completed_successfully.rationale}")
    
    print(f"\nAgent adapted approach: {result.agent_adapted_approach.detected}")
    print(f"Rationale: {result.agent_adapted_approach.rationale}")
    
    print(f"\nUser satisfied: {result.user_satisfied_with_result.detected}")
    print(f"Rationale: {result.user_satisfied_with_result.rationale}")
    
    return result


def example_multi_sample_analysis():
    """Example of multi-sample analysis for statistical reliability."""
    
    issue_text = """
    Feature Request: Add dark mode support
    
    It would be great if the application supported a dark mode theme.
    Many users prefer dark interfaces, especially when working late.
    
    This could be implemented as a toggle in the settings menu.
    """
    
    # Create annotator
    annotator = SolvabilityAnnotator(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Analyze with multiple samples
    multi_result = annotator.annotate_with_samples(
        issue_text,
        samples=3,
        temperature=0.7
    )
    
    print("=== Multi-Sample Analysis Results ===")
    print(f"Sample count: {multi_result.sample_count}")
    print(f"Sample diversity: {multi_result.get_sample_diversity():.3f}")
    
    # Get detection rates across samples
    if hasattr(multi_result, 'get_detection_rates'):
        detection_rates = multi_result.get_detection_rates()
        print("\nDetection rates across samples:")
        for feature, rate in detection_rates.items():
            print(f"  {feature}: {rate:.1%}")
    
    # Get consensus result
    consensus = multi_result.get_consensus_result()
    if consensus:
        print(f"\nConsensus detection rate: {consensus.get_detection_rate():.2%}")
    
    return multi_result


def example_batch_processing():
    """Example of processing multiple items in batch."""
    
    issues = [
        "Bug: Login button doesn't work",
        "Feature request: Add export functionality with CSV and JSON formats",
        "Issue: Application crashes on startup with error code 0x80004005",
    ]
    
    # Create annotator
    annotator = SolvabilityAnnotator(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Process in batch
    results = annotator.annotate_batch(issues, max_workers=2)
    
    print("=== Batch Processing Results ===")
    for i, result in enumerate(results):
        if result:
            print(f"\nIssue {i+1}: {result.get_detection_rate():.2%} detection rate")
            print(f"  Detected: {len(result.get_detected_features())} features")
            print(f"  Missing: {len(result.get_missing_features())} features")
    
    return results


def run_all_examples():
    """Run all examples."""
    
    print("Running Critic Rubrics Examples")
    print("=" * 50)
    
    try:
        print("\n1. Solvability Analysis")
        print("-" * 30)
        example_solvability_analysis()
        
        print("\n\n2. Trajectory Analysis")
        print("-" * 30)
        example_trajectory_analysis()
        
        print("\n\n3. Multi-Sample Analysis")
        print("-" * 30)
        example_multi_sample_analysis()
        
        print("\n\n4. Batch Processing")
        print("-" * 30)
        example_batch_processing()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set your API key in the OPENAI_API_KEY environment variable")


if __name__ == "__main__":
    run_all_examples()