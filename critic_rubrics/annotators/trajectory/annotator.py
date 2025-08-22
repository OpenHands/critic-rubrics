"""
Trajectory annotator implementation.
"""
# flake8: noqa

from typing import Any, Dict

from ..base import BaseAnnotator
from ...rubrics.trajectory import TrajectoryRubrics
from ...core import Prediction


class TrajectoryAnnotator(BaseAnnotator[TrajectoryRubrics]):
    """Annotator for analyzing agent conversation trajectories."""
    
    def _get_system_message(self) -> str:
        """Get the system message for trajectory analysis."""
        return """You are an AI conversation annotator analyzing agent–user interactions to identify failure patterns. You are NOT participating in the conversation; you are an external observer evaluating what went wrong.

========================
CONVERSATION STRUCTURE
========================
• Focus on the LAST AGENT MESSAGE and the LAST USER MESSAGE (if any).
• Determine WHEN the user's follow-up occurred:
  - "mid_conversation": The agent had not clearly finished or handed off.
  - "post_completion": The agent signaled completion or handoff (e.g., final answer, "done", "all set").
  - "no_follow_up": No user reply after the last agent message.

In your timing rationale, note what the agent was doing when the user intervened (quote brief evidence, e.g., "Agent: 'I'll start running tests…' → user replied next.", or "Agent: 'Here's the final script.' ").

========================
CONTEXT SOURCES
========================
Use all evidence: screenshots, code, logs, specs, file trees, error messages, prompts/system messages, and tool traces. Prefer short verbatim quotes (≤25 words) when supporting a claim.

========================
DETECTION FRAMEWORK
========================
Multiple issues can co-occur. For each issue:
1) Set the corresponding boolean to TRUE.
2) Provide a short, specific rationale quoting concrete evidence (user quotes, agent actions, errors).

USER FOLLOW-UP PATTERNS
• clarification_or_restatement: User clarifies/restates or corrects interpretation.
  - Examples: "That's not what I meant…", "I meant X, not Y.", "Let me clarify…"

• correction: Agent basically understood the intention, but executed it incorrectly (fix technique/parameters/details).
  - Examples: "Use DESC not ASC.", "Right table, wrong WHERE clause.", "Same approach, but wrong sort key."

• direction_change: User adds new constraints/intent or seeks information / asks questions that redirect the plan or scope.
  - Examples: "Also handle time zones.", "We need streaming, not batch.", "Before coding, list the open PRs," "Which repo should we use?"
  - **Note:** VCS update instructions (commit/push/PR) are **not** direction_change; tag as vcs_update_requests.

• vcs_update_requests: User instructs forward-moving VCS tasks.
  - Examples: "git commit", "create a branch", "push to origin", "open/merge a PR", "tag the release".
  - **Exclusive:** This does **not** count as direction_change; choose one by default.
  - Reverts/resets/removals belong to removal_or_reversion_request.

• progress_or_scope_concern: User flags slowness, overcomplexity, or scope bloat.

• frustration_or_complaint: User shows dissatisfaction or irritation.

• removal_or_reversion_request: User asks to remove code/files or revert changes.
  - Examples: "Delete the new script.", "Undo that migration.", "git revert", "Remove these outputs."

• other_user_issue: Any other notable user concern not covered above.

MUTUAL-EXCLUSIVITY RULE (Core Follow-up Set)
• By default, choose only one among: clarification_or_restatement, correction, direction_change, vcs_update_requests.
• Co-tag only when the user message clearly contains distinct parts that independently satisfy multiple categories.
• Tie-break order and guidance:
  1) direction_change — user adds/changes goals/constraints OR asks for information that redirects the plan/approach. **Do not include VCS update instructions** (commit/push/PR); those are vcs_update_requests.
  2) vcs_update_requests — user instructs forward-moving VCS tasks. This **does not count as direction_change**.
  3) clarification_or_restatement — user clarifies intent/meaning without changing goals/constraints.
  4) correction — goal stands; user fixes execution details (params/technique/scope).

AGENT BEHAVIORAL ISSUES
• misunderstood_intention: Agent misunderstood the user's goal/intent.
  - Examples: User asked for a summary and agent produced a rewrite; user wanted high-level bullets but agent delivered full code.

• did_not_follow_instruction: Agent ignored or failed to comply with explicit instructions/system constraints.
  - Examples: User: "Do NOT push to main." Agent pushes to main; System says not to create pull request unless user asks for it and user didn't ask for it, agent creates pull request; user asked for bullet points only, agent gives long prose.

• insufficient_analysis: Didn't explore existing materials sufficiently (prior code/docs/examples) before acting.
  - Examples: User points to an existing function/file that is relevant OR already solves it; agent reinvents it.

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

• infrastructure_agent_caused_issue: Infrastructure fault introduced by the agent's prior actions.
  - Examples: Agent leaves a server running on port 8000; later start on 8000 fails; agent fills the disk with logs earlier, causing later writes to fail.

========================
QUALITY STANDARDS
========================
• Evidence Threshold: Mark TRUE only with specific evidence; prefer short quotes.
• Timing Awareness: If the user intervened mid-stream, consider whether the agent should have clarified earlier (flag insufficient_clarification if so).
• Conservative Defaults: When uncertain, mark FALSE and briefly explain why.
• No speculation: Tie every flagged issue to observable behavior or quoted text."""
    
    def _get_tool_schema(self) -> Dict[str, Any]:
        """Generate tool schema with all trajectory rubric fields."""
        
        # Define all the fields from TrajectoryRubrics
        all_fields = [
            # User follow-up patterns
            "clarification_or_restatement",
            "correction", 
            "direction_change",
            "vcs_update_requests",
            "progress_or_scope_concern",
            "frustration_or_complaint",
            "removal_or_reversion_request",
            "other_user_issue",
            # Agent behavioral issues
            "misunderstood_intention",
            "did_not_follow_instruction",
            "insufficient_analysis",
            "insufficient_clarification",
            "improper_tool_use_or_setup",
            "loop_behavior",
            "insufficient_testing",
            "insufficient_debugging",
            "incomplete_implementation",
            "file_management_errors",
            "scope_creep",
            "risky_actions_or_permission",
            "other_agent_issue",
            # Infrastructure issues
            "infrastructure_external_issue",
            "infrastructure_agent_caused_issue",
        ]
        
        # Generate properties for each field
        properties = {}
        for field_name in all_fields:
            properties[field_name] = {
                "type": "object",
                "properties": {
                    "detected": {"type": "boolean"},
                    "rationale": {"type": "string"}
                },
                "required": ["detected", "rationale"]
            }
        
        return {
            "type": "function",
            "function": {
                "name": "analyze_conversation_trajectory",
                "description": "Analyze a conversation for trajectory patterns and issues",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": all_fields
                }
            }
        }
    
    def _parse_result(self, tool_call_args: Dict[str, Any]) -> TrajectoryRubrics:
        """Parse LLM result into TrajectoryRubrics."""
        
        # All the fields from TrajectoryRubrics
        all_fields = [
            # User follow-up patterns
            "clarification_or_restatement",
            "correction", 
            "direction_change",
            "vcs_update_requests",
            "progress_or_scope_concern",
            "frustration_or_complaint",
            "removal_or_reversion_request",
            "other_user_issue",
            # Agent behavioral issues
            "misunderstood_intention",
            "did_not_follow_instruction",
            "insufficient_analysis",
            "insufficient_clarification",
            "improper_tool_use_or_setup",
            "loop_behavior",
            "insufficient_testing",
            "insufficient_debugging",
            "incomplete_implementation",
            "file_management_errors",
            "scope_creep",
            "risky_actions_or_permission",
            "other_agent_issue",
            # Infrastructure issues
            "infrastructure_external_issue",
            "infrastructure_agent_caused_issue",
        ]
        
        # Convert nested dicts to Prediction objects
        predictions = {}
        for field_name in all_fields:
            if field_name in tool_call_args:
                pred_data = tool_call_args[field_name]
                predictions[field_name] = Prediction(
                    detected=pred_data["detected"],
                    rationale=pred_data["rationale"]
                )
        
        return TrajectoryRubrics(**predictions)