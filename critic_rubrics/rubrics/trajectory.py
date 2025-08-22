"""
Trajectory rubrics for conversation analysis.
"""

from pydantic import BaseModel, Field
from ..types import Prediction


ANNOTATION_SYSTEM_MESSAGE = """You are an AI conversation annotator analyzing agent-environment interactions to identify failure patterns. You are NOT participating in the conversation; you are an external observer evaluating what went wrong.

========================
CONVERSATION STRUCTURE
========================
• Focus on the LAST AGENT MESSAGE.

========================
CONTEXT SOURCES
========================
Use all evidence: screenshots, code, logs, specs, file trees, error messages, prompts/system messages, and tool traces. Prefer short verbatim quotes (≤25 words) when supporting a claim.

========================
DETECTION FRAMEWORK
========================
Multiple issues can co-occur. For each issue:
1) Set the corresponding boolean to TRUE.
2) Provide a short, specific rationale quoting concrete evidence (agent actions, errors).

AGENT BEHAVIORAL ISSUES
• misunderstood_intention: Agent misunderstood the user’s goal/intent.
  - Examples: User asked for a summary and agent produced a rewrite; user wanted high-level bullets but agent delivered full code.

• did_not_follow_instruction: Agent ignored or failed to comply with explicit instructions/system constraints.
  - Examples: User: “Do NOT push to main.” Agent pushes to main; System says not to create pull request unless user asks for it and user didn't ask for it, agent creates pull request; user asked for bullet points only, agent gives long prose.

• insufficient_analysis: Didn’t explore existing materials sufficiently (prior code/docs/examples) before acting.
  - Examples: User points to an existing function/file that is relavant OR already solves it; agent reinvents it.

• insufficient_clarification: Failed to ask necessary questions before acting when requirements were ambiguous.
  - Examples: Agent proceeds despite unclear acceptance criteria (e.g., locales, time zones, error thresholds) then is corrected later.

• improper_tool_use_or_setup: Misused tools/commands or had missing/incorrect dependencies/setup.
  - Examples: wrong command syntax, using inappropriate tools for the task

• loop_behavior: Repeats the same failed action 3+ times without strategy change.
  - Examples: repeat the same failed action 3+ times without changing approach).

• insufficient_testing: Skipped reasonable verification/tests for non-trivial or risky changes (note: trivial edits may be acceptable).
  - Examples: No run/validation for a new parser; no check that a migration applies cleanly; no sanity check of output.

• insufficient_debugging: Did not investigate or reduce failing behavior when needed to make progress.
  - Examples: Ignores stack trace; no isolation of failure; proceeds while errors persist.

• incomplete_implementation: Delivered unfinished or non-functioning work.
  - Examples: TODO/FIXME left; stub methods; code that cannot run.

• file_management_errors: Wrong paths, overwrites, misplaced/extra files (including unnecessary files).
  - Examples: Writes into wrong directory; overwrites config; creates unwanted artifacts.

• scope_creep: Implemented unrequested features without approval.
  - Examples: Adds a dashboard or endpoint not asked for.

• risky_actions_or_permission: Risky steps without user's explicit consent.
  - Examples: git push to main; deleting existing files in a repo (deleting files created by agent itself is fine); altering credentials.

• other_agent_issue: Any agent-side problem not covered above.

INFRASTRUCTURE (EXTERNAL vs AGENT-CAUSED)
• infrastructure_external_issue: Environment/platform limits outside agent control.
  - Examples: Provider outage; disk full on managed runner; missing enterprise API key; network failure not caused by agent.

• infrastructure_agent_caused_issue: Infrastructure fault introduced by the agent’s prior actions.
  - Examples: Agent leaves a server running on port 8000; later start on 8000 fails; agent fills the disk with logs earlier, causing later writes to fail.

========================
QUALITY STANDARDS
========================
• Evidence Threshold: Mark TRUE only with specific evidence; prefer short quotes.
• Timing Awareness: If the user intervened mid-stream, consider whether the agent should have clarified earlier (flag insufficient_clarification if so).
• Conservative Defaults: When uncertain, mark FALSE and briefly explain why.
• No speculation: Tie every flagged issue to observable behavior or quoted text.
"""



ANNOTATION_INSTRUCTION_MESSAGE = """=== END OF CONVERSATION TO ANALYZE ===

Fill the annotate_conversation function.

Goal
- Set only the booleans that clearly apply.

What to record
1) Agent behavioral issues (select all that apply)
   - misunderstood_intention, did_not_follow_instruction, insufficient_analysis, insufficient_clarification,
     improper_tool_use_or_setup, loop_behavior, insufficient_testing, insufficient_debugging,
     incomplete_implementation, file_management_errors, scope_creep, risky_actions_or_permission,
     other_agent_issue.
   - Rationale: cite code/commands/errors or a short quote and explain in one sentence.

2) Infrastructure
   - infrastructure_external_issue_detected for environment/platform limits beyond agent control.
   - infrastructure_agent_caused_issue_detected for faults introduced by the agent’s prior actions (e.g., orphaned server on port 8000).
   - Rationale: include the error/status line or brief description.


Evidence & quality
- Prefer concrete, minimal quotes; avoid speculation. If evidence is insufficient, leave the flag false.

Quick disambiguation (common splits)
- insufficient_analysis vs insufficient_clarification: didn’t look for existing work vs didn’t ask when requirements were ambiguous.
- insufficient_testing vs insufficient_debugging: skipped reasonable verification vs didn’t investigate a failing state enough to make progress.
"""


# Tool definition for conversations WITH follow-up messages
ANNOTATION_TOOL = {
  "type": "function",
  "function": {
    "name": "annotate_conversation",
    "description": "Annotate agent conversation after agent work.",
    "parameters": {
      "type": "object",
      "properties": {

        # /* AGENT BEHAVIORAL ISSUES */
        "misunderstood_intention_detected": {
          "type": "boolean",
          "description": "Agent misunderstood the user’s goal/intent. Examples: User asked for a summary; agent produced a rewrite; user wanted high-level bullets; agent delivered full code."
        },
        "misunderstood_intention_rationale": {
          "type": "string",
          "description": "Quote evidence concisely (≤25 words) and explain in a sentence."
        },

        "did_not_follow_instruction_detected": {
          "type": "boolean",
          "description": "Agent ignored or failed to comply with explicit instructions/system constraints. Examples: User: “Do NOT push to main.” Agent pushes; System says not to create a PR unless the user asks and the user didn’t ask; agent creates a PR; user asked for bullet points only, agent gives long prose."
        },
        "did_not_follow_instruction_rationale": {
          "type": "string",
          "description": "Quote the instruction and the noncompliant action."
        },

        "insufficient_analysis_detected": {
          "type": "boolean",
          "description": "Didn’t explore existing materials (prior code/docs/examples) before acting. Examples: User points to an existing function/file that is relevant or already solves it; agent reinvents it."
        },
        "insufficient_analysis_rationale": {
          "type": "string",
          "description": "Explain with evidence (e.g., user points to existing file)."
        },

        "insufficient_clarification_detected": {
          "type": "boolean",
          "description": "Failed to ask necessary questions before acting when requirements were ambiguous. Examples: Agent proceeds despite unclear acceptance criteria (locales, time zones, error thresholds) then is corrected later."
        },
        "insufficient_clarification_rationale": {
          "type": "string",
          "description": "Quote the ambiguous instruction and note the missing questions."
        },

        "improper_tool_use_or_setup_detected": {
          "type": "boolean",
          "description": "Misused tools/commands or used inappropriate tools; missing/incorrect dependencies/setup. Examples: wrong command syntax; using an inappropriate tool; import errors; wrong API URL; malformed auth header."
        },
        "improper_tool_use_or_setup_rationale": {
          "type": "string",
          "description": "Cite commands/errors and explain briefly."
        },

        "loop_behavior_detected": {
          "type": "boolean",
          "description": "Repeats the same failed action 3+ times without strategy change."
        },
        "loop_behavior_rationale": {
          "type": "string",
          "description": "Describe the repetitions and counts (e.g., ‘same curl 3× → 401 each time’)."
        },

        "insufficient_testing_detected": {
          "type": "boolean",
          "description": "Skipped reasonable verification/tests for non-trivial or risky changes (trivial edits may be acceptable). Examples: No run/validation for a new parser; no check that a migration applies cleanly; no sanity check of output."
        },
        "insufficient_testing_rationale": {
          "type": "string",
          "description": "Describe what should have been tested and why."
        },

        "insufficient_debugging_detected": {
          "type": "boolean",
          "description": "Did not investigate or reduce failing behavior when needed to make progress. Examples: Ignores stack trace; no isolation of failure; proceeds while errors persist."
        },
        "insufficient_debugging_rationale": {
          "type": "string",
          "description": "Quote the error and explain the missing debugging steps."
        },

        "incomplete_implementation_detected": {
          "type": "boolean",
          "description": "Delivered unfinished or non-functioning work. Examples: TODO/FIXME left; stub methods; code that cannot run."
        },
        "incomplete_implementation_rationale": {
          "type": "string",
          "description": "Point to the incomplete parts and impact."
        },

        "file_management_errors_detected": {
          "type": "boolean",
          "description": "Wrong paths, overwrites, misplaced/extra (unnecessary) files. Examples: writes into wrong directory; overwrites config; creates unwanted artifacts."
        },
        "file_management_errors_rationale": {
          "type": "string",
          "description": "Describe exact paths/actions and consequences."
        },

        "scope_creep_detected": {
          "type": "boolean",
          "description": "Implemented unrequested features without approval. Examples: adds a dashboard or endpoint not asked for."
        },
        "scope_creep_rationale": {
          "type": "string",
          "description": "Explain how it exceeds the ask."
        },

        "risky_actions_or_permission_detected": {
          "type": "boolean",
          "description": "Risky steps without the user's explicit consent. Examples: git push to main; deleting existing files in a repo (deleting files created by the agent itself is fine); altering credentials."
        },
        "risky_actions_or_permission_rationale": {
          "type": "string",
          "description": "Describe the risky action and lack of consent."
        },

        "other_agent_issue_detected": {
          "type": "boolean",
          "description": "Any other agent-side problem not covered above."
        },
        "other_agent_issue_rationale": {
          "type": "string",
          "description": "Describe and cite brief evidence."
        },

        # /* INFRASTRUCTURE (EXTERNAL vs AGENT-CAUSED) */
        "infrastructure_external_issue_detected": {
          "type": "boolean",
          "description": "Environment/platform limits outside agent control. Examples: provider outage; disk full on a managed runner; missing enterprise API key; network failure not caused by agent."
        },
        "infrastructure_external_issue_rationale": {
          "type": "string",
          "description": "Quote the error/status and explain briefly."
        },

        "infrastructure_agent_caused_issue_detected": {
          "type": "boolean",
          "description": "Infrastructure faults introduced by the agent’s prior actions. Examples: agent leaves server on port 8000 → later start on 8000 fails; agent fills disk with logs → later writes fail."
        },
        "infrastructure_agent_caused_issue_rationale": {
          "type": "string",
          "description": "Describe the agent-caused condition and the resulting failure."
        },
      },
      "required": []
    }
  }
}

# Make EVERYTHING required
ANNOTATION_TOOL['function']['parameters']['required'] = sorted(ANNOTATION_TOOL['function']['parameters']['properties'].keys())


class TrajectoryUserFollowupRubrics(BaseModel):
    """
    Comprehensive trajectory analysis features based on Xingyao rubrics.
    Includes user follow-up patterns, agent behavioral issues, and infrastructure problems.
    """
    
    # USER FOLLOW-UP PATTERNS
    clarification_or_restatement: Prediction = Field(
        description="User clarifies/restates or corrects interpretation"
    )
    correction: Prediction = Field(
        description="Agent understood intention but executed incorrectly"
    )
    direction_change: Prediction = Field(
        description="User adds new constraints/intent or redirects plan/scope"
    )
    vcs_update_requests: Prediction = Field(
        description="User instructs forward-moving VCS tasks (commit/push/PR)"
    )
    progress_or_scope_concern: Prediction = Field(
        description="User flags slowness, overcomplexity, or scope bloat"
    )
    frustration_or_complaint: Prediction = Field(
        description="User shows dissatisfaction or irritation"
    )
    removal_or_reversion_request: Prediction = Field(
        description="User asks to remove code/files or revert changes"
    )
    other_user_issue: Prediction = Field(
        description="Any other notable user concern not covered above"
    )
    
    # AGENT BEHAVIORAL ISSUES
    misunderstood_intention: Prediction = Field(
        description="Agent misunderstood the user's goal/intent"
    )
    did_not_follow_instruction: Prediction = Field(
        description="Agent ignored or failed to comply with explicit instructions"
    )
    insufficient_analysis: Prediction = Field(
        description="Didn't explore existing materials sufficiently before acting"
    )
    insufficient_clarification: Prediction = Field(
        description="Failed to ask necessary questions when requirements were ambiguous"
    )
    improper_tool_use_or_setup: Prediction = Field(
        description="Misused tools/commands or had missing/incorrect dependencies"
    )
    loop_behavior: Prediction = Field(
        description="Repeats the same failed action 3+ times without strategy change"
    )
    insufficient_testing: Prediction = Field(
        description="Skipped reasonable verification/tests for non-trivial changes"
    )
    insufficient_debugging: Prediction = Field(
        description="Did not investigate or reduce failing behavior when needed"
    )
    incomplete_implementation: Prediction = Field(
        description="Delivered unfinished or non-functioning work"
    )
    file_management_errors: Prediction = Field(
        description="Wrong paths, overwrites, misplaced/extra files"
    )
    scope_creep: Prediction = Field(
        description="Implemented unrequested features without approval"
    )
    risky_actions_or_permission: Prediction = Field(
        description="Risky steps without user's explicit consent"
    )
    other_agent_issue: Prediction = Field(
        description="Any other agent-side problem not covered above"
    )
    
    # INFRASTRUCTURE ISSUES
    infrastructure_external_issue: Prediction = Field(
        description="Environment/platform limits outside agent control"
    )
    infrastructure_agent_caused_issue: Prediction = Field(
        description="Infrastructure faults introduced by agent's prior actions"
    )
