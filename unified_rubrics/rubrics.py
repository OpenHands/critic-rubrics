"""
Pre-defined rubric sets and builders for common annotation tasks.
"""

from typing import List, Dict, Any, Optional
from .core import RubricItem, RubricCategory, RubricSet


# =============================================================================
# AGENT BEHAVIORAL RUBRICS
# =============================================================================

AGENT_BEHAVIORAL_ITEMS = [
    RubricItem(
        identifier="misunderstood_intention",
        description="Agent misunderstood the user's goal/intent. Examples: User asked for a summary; agent produced a rewrite; user wanted high-level bullets; agent delivered full code."
    ),
    RubricItem(
        identifier="did_not_follow_instruction", 
        description="Agent ignored or failed to comply with explicit instructions/system constraints. Examples: User: 'Do NOT push to main.' Agent pushes; System says not to create a PR unless the user asks and the user didn't ask; agent creates a PR; user asked for bullet points only, agent gives long prose."
    ),
    RubricItem(
        identifier="insufficient_analysis",
        description="Didn't explore existing materials (prior code/docs/examples) before acting. Examples: User points to an existing function/file that is relevant or already solves it; agent reinvents it."
    ),
    RubricItem(
        identifier="insufficient_clarification",
        description="Failed to ask necessary questions before acting when requirements were ambiguous. Examples: Agent proceeds despite unclear acceptance criteria (locales, time zones, error thresholds) then is corrected later."
    ),
    RubricItem(
        identifier="improper_tool_use_or_setup",
        description="Misused tools/commands or used inappropriate tools; missing/incorrect dependencies/setup. Examples: wrong command syntax; using an inappropriate tool; import errors; wrong API URL; malformed auth header."
    ),
    RubricItem(
        identifier="loop_behavior",
        description="Repeats the same failed action 3+ times without strategy change."
    ),
    RubricItem(
        identifier="insufficient_testing",
        description="Skipped reasonable verification/tests for non-trivial or risky changes (trivial edits may be acceptable). Examples: No run/validation for a new parser; no check that a migration applies cleanly; no sanity check of output."
    ),
    RubricItem(
        identifier="insufficient_debugging",
        description="Did not investigate or reduce failing behavior when needed to make progress. Examples: Ignores stack trace; no isolation of failure; proceeds while errors persist."
    ),
    RubricItem(
        identifier="incomplete_implementation",
        description="Delivered unfinished or non-functioning work. Examples: TODO/FIXME left; stub methods; code that cannot run."
    ),
    RubricItem(
        identifier="file_management_errors",
        description="Wrong paths, overwrites, misplaced/extra (unnecessary) files. Examples: writes into wrong directory; overwrites config; creates unwanted artifacts."
    ),
    RubricItem(
        identifier="scope_creep",
        description="Implemented unrequested features without approval. Examples: adds a dashboard or endpoint not asked for."
    ),
    RubricItem(
        identifier="risky_actions_or_permission",
        description="Risky steps without the user's explicit consent. Examples: git push to main; deleting existing files in a repo (deleting files created by the agent itself is fine); altering credentials."
    ),
    RubricItem(
        identifier="other_agent_issue",
        description="Any other agent-side problem not covered above."
    ),
]

AGENT_BEHAVIORAL_RUBRICS = RubricCategory(
    name="agent_behavioral",
    description="Issues related to agent behavior and decision-making",
    items=AGENT_BEHAVIORAL_ITEMS
)


# =============================================================================
# USER FOLLOW-UP RUBRICS
# =============================================================================

USER_FOLLOWUP_ITEMS = [
    RubricItem(
        identifier="clarification_or_restatement",
        description="User clarifies/restates or corrects interpretation. Examples: 'That's not what I meant…', 'I meant X, not Y.', 'Let me clarify…'"
    ),
    RubricItem(
        identifier="correction",
        description="Agent broadly understood the intention but executed it incorrectly (technique/parameters/details). Examples: 'Use DESC not ASC.', 'Right table, wrong WHERE clause.', 'Same approach, wrong sort key.'"
    ),
    RubricItem(
        identifier="direction_change",
        description="User adds new constraints/intent not previously specified; scope/goal evolves. Examples: 'Also handle time zones.', 'We actually need streaming, not batch.', 'Support Windows too.'"
    ),
    RubricItem(
        identifier="vcs_update_requests",
        description="User instructs forward-moving VCS updates: commit, create branch, push, open/merge PR, tag. (Revert/reset/remove → use removal_or_reversion_request.)"
    ),
    RubricItem(
        identifier="progress_or_scope_concern",
        description="User flags slowness, overcomplexity, or scope bloat. Examples: 'This is taking too long.', 'Try a simpler approach.', 'This goes beyond what I asked.'"
    ),
    RubricItem(
        identifier="frustration_or_complaint",
        description="User expresses dissatisfaction or irritation. Examples: 'This is wrong.', 'You're not listening.', excessive caps or punctuation ('!!!', '???')."
    ),
    RubricItem(
        identifier="removal_or_reversion_request",
        description="User asks to remove or revert code/files/changes. Examples: 'Delete the new script.', 'Undo that migration.', 'Remove these outputs.', 'git revert'."
    ),
    RubricItem(
        identifier="other_user_issue",
        description="Any other notable user concern not covered above."
    ),
]

USER_FOLLOWUP_RUBRICS = RubricCategory(
    name="user_followup",
    description="Patterns in user follow-up messages and interventions",
    items=USER_FOLLOWUP_ITEMS,
    mutually_exclusive=True,  # Core follow-up types are mutually exclusive by default
    max_selections=1
)


# =============================================================================
# INFRASTRUCTURE RUBRICS
# =============================================================================

INFRASTRUCTURE_ITEMS = [
    RubricItem(
        identifier="infrastructure_external_issue",
        description="Environment/platform limits outside agent control. Examples: provider outage; disk full on a managed runner; missing enterprise API key; network failure not caused by agent."
    ),
    RubricItem(
        identifier="infrastructure_agent_caused_issue",
        description="Infrastructure faults introduced by the agent's prior actions. Examples: agent leaves server on port 8000 → later start on 8000 fails; agent fills disk with logs → later writes fail."
    ),
]

INFRASTRUCTURE_RUBRICS = RubricCategory(
    name="infrastructure",
    description="Infrastructure-related issues both external and agent-caused",
    items=INFRASTRUCTURE_ITEMS
)


# =============================================================================
# SOLVABILITY RUBRICS (for Calvin-style analysis)
# =============================================================================

def create_solvability_rubric(features: List[Dict[str, str]]) -> RubricSet:
    """
    Create a solvability rubric set from a list of feature definitions.
    
    Args:
        features: List of dicts with 'identifier' and 'description' keys
        
    Returns:
        RubricSet configured for solvability analysis
    """
    items = [
        RubricItem(
            identifier=feature["identifier"],
            description=feature["description"],
            requires_rationale=False  # Solvability features typically don't need rationales
        )
        for feature in features
    ]
    
    return RubricSet(
        name="solvability",
        description="Solvability analysis rubric for issue classification",
        standalone_items=items
    )


# Example solvability features (can be customized)
DEFAULT_SOLVABILITY_FEATURES = [
    {"identifier": "has_clear_requirements", "description": "The issue has clear, well-defined requirements"},
    {"identifier": "has_reproduction_steps", "description": "The issue includes steps to reproduce the problem"},
    {"identifier": "has_expected_behavior", "description": "The issue describes the expected behavior"},
    {"identifier": "has_actual_behavior", "description": "The issue describes the actual (incorrect) behavior"},
    {"identifier": "has_error_messages", "description": "The issue includes relevant error messages or logs"},
    {"identifier": "has_environment_info", "description": "The issue includes environment/system information"},
    {"identifier": "is_bug_report", "description": "This is a bug report rather than a feature request"},
    {"identifier": "is_feature_request", "description": "This is a feature request rather than a bug report"},
    {"identifier": "requires_domain_knowledge", "description": "Solving this requires specialized domain knowledge"},
    {"identifier": "requires_external_resources", "description": "Solving this requires external APIs, databases, or resources"},
    {"identifier": "is_well_scoped", "description": "The issue is well-scoped and not overly broad"},
    {"identifier": "has_acceptance_criteria", "description": "The issue has clear acceptance criteria or definition of done"},
]

SOLVABILITY_RUBRICS = create_solvability_rubric(DEFAULT_SOLVABILITY_FEATURES)


# =============================================================================
# CONVERSATION RUBRICS (combines multiple categories)
# =============================================================================

def create_conversation_rubric(include_timing: bool = True, include_task_type: bool = True) -> RubricSet:
    """
    Create a comprehensive conversation analysis rubric.
    
    Args:
        include_timing: Whether to include follow-up timing analysis
        include_task_type: Whether to include task type classification
        
    Returns:
        RubricSet configured for conversation analysis
    """
    rubric_set = RubricSet(
        name="conversation_analysis",
        description="Comprehensive rubric for analyzing agent-user conversations"
    )
    
    # Add categories
    rubric_set.add_category(AGENT_BEHAVIORAL_RUBRICS)
    rubric_set.add_category(USER_FOLLOWUP_RUBRICS)
    rubric_set.add_category(INFRASTRUCTURE_RUBRICS)
    
    # Add additional fields
    additional_fields = {}
    
    if include_timing:
        additional_fields["follow_up_timing"] = {
            "type": "string",
            "enum": ["mid_conversation", "post_completion", "no_follow_up"],
            "description": "WHEN did the user follow up? • mid_conversation: agent hadn't clearly finished/handed off • post_completion: agent signaled completion/hand-off • no_follow_up: no user message after the last agent message.",
            "required": True
        }
        additional_fields["follow_up_timing_rationale"] = {
            "type": "string", 
            "description": "Evidence for timing with a brief quote (≤25 words). Examples: 'Agent: Here's the final script.' → post_completion; 'Agent: I'll start tests…' and user replied → mid_conversation.",
            "required": True
        }
    
    if include_task_type:
        additional_fields["task_type"] = {
            "type": "string",
            "enum": ["coding", "debugging", "research", "file_management", "configuration", "documentation", "analysis", "other"],
            "description": "Primary task the user asked for. Examples: implement a script → coding; fix a stack trace → debugging; find sources → research; move/rename files → file_management; change settings/keys → configuration; write a README → documentation; assess a design → analysis; otherwise → other.",
            "required": True
        }
    
    rubric_set.additional_fields = additional_fields
    
    return rubric_set


CONVERSATION_RUBRICS = create_conversation_rubric()


# =============================================================================
# CUSTOM RUBRIC BUILDER
# =============================================================================

def create_custom_rubric(name: str, description: str, 
                        categories: Optional[List[RubricCategory]] = None,
                        items: Optional[List[RubricItem]] = None,
                        additional_fields: Optional[Dict[str, Dict[str, Any]]] = None) -> RubricSet:
    """
    Create a custom rubric set.
    
    Args:
        name: Name of the rubric set
        description: Description of what this rubric evaluates
        categories: List of rubric categories to include
        items: List of standalone rubric items
        additional_fields: Additional non-rubric fields to include
        
    Returns:
        Custom RubricSet
    """
    rubric_set = RubricSet(
        name=name,
        description=description,
        categories=categories or [],
        standalone_items=items or [],
        additional_fields=additional_fields or {}
    )
    
    return rubric_set


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

# Export commonly used rubric sets
__all__ = [
    # Individual categories
    "AGENT_BEHAVIORAL_RUBRICS",
    "USER_FOLLOWUP_RUBRICS", 
    "INFRASTRUCTURE_RUBRICS",
    
    # Complete rubric sets
    "SOLVABILITY_RUBRICS",
    "CONVERSATION_RUBRICS",
    
    # Builders
    "create_solvability_rubric",
    "create_conversation_rubric",
    "create_custom_rubric",
    
    # Feature lists for customization
    "DEFAULT_SOLVABILITY_FEATURES",
    "AGENT_BEHAVIORAL_ITEMS",
    "USER_FOLLOWUP_ITEMS",
    "INFRASTRUCTURE_ITEMS",
]