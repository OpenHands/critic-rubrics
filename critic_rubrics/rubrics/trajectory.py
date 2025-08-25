# annotate_conversation_rubric.py
from pydantic import Field
from typing import Any

from .base import BaseRubrics, Prediction

class AnnotateConversationRubric(BaseRubrics):
    # --- AGENT BEHAVIORAL ISSUES ---
    misunderstood_intention: Prediction = Field(
        description="Agent misunderstood the user’s goal/intent. Examples: User asked for a summary; agent produced a rewrite; user wanted high-level bullets; agent delivered full code."
    )
    did_not_follow_instruction: Prediction = Field(
        description="Agent ignored or failed to comply with explicit instructions/system constraints. Examples: User: 'Do NOT push to main.' Agent pushes; System says not to create a PR unless the user asks and the user didn't ask; agent creates a PR; user asked for bullet points only, agent gives long prose."
    )
    insufficient_analysis: Prediction = Field(
        description="Didn’t explore existing materials (prior code/docs/examples) before acting. Examples: User points to an existing function/file that is relevant or already solves it; agent reinvents it."
    )
    insufficient_clarification: Prediction = Field(
        description="Failed to ask necessary questions before acting when requirements were ambiguous. Examples: Agent proceeds despite unclear acceptance criteria (locales, time zones, error thresholds) then is corrected later."
    )
    improper_tool_use_or_setup: Prediction = Field(
        description="Misused tools/commands or used inappropriate tools; missing/incorrect dependencies/setup. Examples: wrong command syntax; using an inappropriate tool; import errors; wrong API URL; malformed auth header."
    )
    loop_behavior: Prediction = Field(
        description="Repeats the same failed action 3+ times without strategy change."
    )
    insufficient_testing: Prediction = Field(
        description="Skipped reasonable verification/tests for non-trivial or risky changes (trivial edits may be acceptable). Examples: No run/validation for a new parser; no check that a migration applies cleanly; no sanity check of output."
    )
    insufficient_debugging: Prediction = Field(
        description="Did not investigate or reduce failing behavior when needed to make progress. Examples: Ignores stack trace; no isolation of failure; proceeds while errors persist."
    )
    incomplete_implementation: Prediction = Field(
        description="Delivered unfinished or non-functioning work. Examples: TODO/FIXME left; stub methods; code that cannot run."
    )
    file_management_errors: Prediction = Field(
        description="Wrong paths, overwrites, misplaced/extra (unnecessary) files. Examples: writes into wrong directory; overwrites config; creates unwanted artifacts."
    )
    scope_creep: Prediction = Field(
        description="Implemented unrequested features without approval. Examples: adds a dashboard or endpoint not asked for."
    )
    risky_actions_or_permission: Prediction = Field(
        description="Risky steps without the user's explicit consent. Examples: git push to main; deleting existing files in a repo (deleting files created by the agent itself is fine); altering credentials."
    )
    other_agent_issue: Prediction = Field(
        description="Any other agent-side problem not covered above."
    )

    # --- INFRASTRUCTURE ---
    infrastructure_external_issue: Prediction = Field(
        description="Environment/platform limits outside agent control. Examples: provider outage; disk full on a managed runner; missing enterprise API key; network failure not caused by agent."
    )
    infrastructure_agent_caused_issue: Prediction = Field(
        description="Infrastructure faults introduced by the agent's prior actions. Examples: agent leaves server on port 8000 → later start on 8000 fails; agent fills disk with logs → later writes fail."
    )

    def extra_tool_properties(self) -> dict[str, Any]:
        return {
            "follow_up_timing": {
                "type": "string",
                "enum": ["mid_conversation", "post_completion", "no_follow_up"],
                "description": (
                    "WHEN did the user follow up? mid_conversation: agent hadn't clearly finished; "
                    "post_completion: agent signaled completion/hand-off; "
                    "no_follow_up: no user message after the last agent message."
                ),
            },
            # TODO: pending valerie
            # "task_type": {
            #     "type": "string",
            #     "enum": [
            #         "coding", "debugging", "research", "file_management",
            #         "configuration", "documentation", "analysis", "other"
            #     ],
            #     "description": "Primary task category.",
            # },
        }
