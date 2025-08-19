"""
Simple usage examples for critic_rubrics.
"""

from critic_rubrics import create_solvability_annotator, create_trajectory_annotator


def example_solvability_analysis():
    """Example of analyzing issue solvability."""
    
    # Sample issue text
    issue_text = """
    Bug: Application crashes when clicking submit button
    
    Steps to reproduce:
    1. Open the application
    2. Fill out the form with any data
    3. Click the submit button
    
    Expected: Form should submit successfully
    Actual: Application crashes with error "NullPointerException"
    
    Environment: Windows 10, Chrome 91, App version 2.1.0
    """
    
    # Create annotator
    annotator = create_solvability_annotator(
        model="gpt-4o-mini",
        api_key="your-api-key"
    )
    
    # Analyze the issue
    result = annotator.annotate(issue_text)
    
    print("Solvability Analysis Results:")
    print(f"Clear problem statement: {result.has_clear_problem_statement.detected}")
    print(f"Has reproduction steps: {result.has_reproduction_steps.detected}")
    print(f"Has error messages: {result.has_error_messages.detected}")


def example_trajectory_analysis():
    """Example of analyzing conversation trajectory."""
    
    # Sample conversation
    conversation = """
    User: Can you help me fix this Python error?
    
    Agent: I'd be happy to help! Could you please share the error message you're seeing?
    
    User: Here it is: "NameError: name 'x' is not defined"
    
    Agent: This error occurs because the variable 'x' is being used before it's defined. You need to define 'x' before using it in your code.
    
    User: Oh I see, that makes sense. Let me try defining it first.
    
    Agent: Great! That should resolve the issue. Let me know if you need any other help.
    
    User: Perfect, it works now. Thank you!
    """
    
    # Create annotator
    annotator = create_trajectory_annotator(
        model="gpt-4o-mini", 
        api_key="your-api-key"
    )
    
    # Analyze the conversation
    result = annotator.annotate(conversation)
    
    print("Trajectory Analysis Results:")
    print(f"Agent misunderstood intention: {result.misunderstood_intention.detected}")
    print(f"User requested clarification: {result.clarification_or_restatement.detected}")
    print(f"Agent issues detected: {result.get_agent_issue_count()}")
    print(f"User follow-ups: {result.get_user_followup_count()}")
    print(f"Total issues: {result.get_issue_count()}")
    print(f"Quality score: {result.get_quality_score():.2f}")


if __name__ == "__main__":
    print("=== Solvability Analysis Example ===")
    example_solvability_analysis()
    
    print("\n=== Trajectory Analysis Example ===")
    example_trajectory_analysis()