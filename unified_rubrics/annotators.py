"""
Concrete annotator implementations for different LLM providers and use cases.
"""

import json
from typing import Any, Dict, List, Optional

from .core import RubricAnnotator, MultiSampleAnnotator, RubricSet
from .rubrics import SOLVABILITY_RUBRICS, CONVERSATION_RUBRICS


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SOLVABILITY_SYSTEM_PROMPT = """You are an AI issue analyzer that evaluates the solvability and characteristics of software issues, bug reports, and feature requests.

Your task is to analyze the provided issue description and identify which features or characteristics are present. Focus on objective, observable aspects of the issue description itself.

Guidelines:
- Base your analysis only on the information provided in the issue
- Look for explicit mentions or clear implications of each feature
- Be conservative - only mark features as present when there is clear evidence
- Consider the completeness and clarity of the information provided"""

SOLVABILITY_INSTRUCTION_PROMPT = """Analyze the above issue description and identify which features are present. Use the provided function to record your findings."""


CONVERSATION_SYSTEM_PROMPT = """You are an AI conversation annotator analyzing agent-user interactions to identify failure patterns. You are NOT participating in the conversation; you are an external observer evaluating what went wrong.

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

• infrastructure_agent_caused_issue: Infrastructure fault introduced by the agent's prior actions.
  - Examples: Agent leaves a server running on port 8000; later start on 8000 fails; agent fills the disk with logs earlier, causing later writes to fail.

========================
QUALITY STANDARDS
========================
• Evidence Threshold: Mark TRUE only with specific evidence; prefer short quotes.
• Timing Awareness: If the user intervened mid-stream, consider whether the agent should have clarified earlier (flag insufficient_clarification if so).
• Conservative Defaults: When uncertain, mark FALSE and briefly explain why.
• No speculation: Tie every flagged issue to observable behavior or quoted text."""

CONVERSATION_INSTRUCTION_PROMPT = """=== END OF CONVERSATION TO ANALYZE ===

Fill the annotate function.

Goal
- Identify when the user followed up (mid_conversation, post_completion, or no_follow_up) and what issues occurred.
- Set only the booleans that clearly apply. For the **exclusive set** (direction_change, clarification_or_restatement, correction, vcs_update_requests), choose one by default using the tie-break rules; only co-tag if the message clearly contains independent parts for multiple categories.

What to record
1) Follow-up timing
   - Choose the timing value and, in follow_up_timing_rationale, state what the agent was doing when the user replied and include a short quote.

2) User follow-up patterns (select all that apply)
   - clarification_or_restatement, correction, direction_change, vcs_update_requests, progress_or_scope_concern,
     frustration_or_complaint, removal_or_reversion_request, other_user_issue.
   - Rationale: quote the user and explain in one sentence.

3) Agent behavioral issues (select all that apply)
   - misunderstood_intention, did_not_follow_instruction, insufficient_analysis, insufficient_clarification,
     improper_tool_use_or_setup, loop_behavior, insufficient_testing, insufficient_debugging,
     incomplete_implementation, file_management_errors, scope_creep, risky_actions_or_permission,
     other_agent_issue.
   - Rationale: cite code/commands/errors or a short quote and explain in one sentence.

4) Infrastructure
   - infrastructure_external_issue_detected for environment/platform limits beyond agent control.
   - infrastructure_agent_caused_issue_detected for faults introduced by the agent's prior actions (e.g., orphaned server on port 8000).
   - Rationale: include the error/status line or brief description.

5) Task type
   - Pick one: coding, debugging, research, file_management, configuration, documentation, analysis, other.

Evidence & quality
- Prefer concrete, minimal quotes; avoid speculation. If evidence is insufficient, leave the flag false.
- If the user intervened mid-stream and the request was ambiguous, consider insufficient_clarification.

Quick disambiguation (common splits)
- correction vs misunderstood_intention: right goal, wrong details vs wrong goal altogether.
- did_not_follow_instruction vs direction_change: ignored a clear instruction vs user adds new requirement later.
- insufficient_analysis vs insufficient_clarification: didn't look for existing work vs didn't ask when requirements were ambiguous.
- insufficient_testing vs insufficient_debugging: skipped reasonable verification vs didn't investigate a failing state enough to make progress.
- direction_change includes information seeking / question asking that redirects scope/approach.
- vcs_update_requests is not direction_change; it covers forward-moving VCS steps (commit, branch, push, open/merge PR, tag).
- Requests to revert/reset/remove belong to removal_or_reversion_request.
- For the **exclusive set** (direction_change, clarification_or_restatement, correction, vcs_update_requests), choose one by default using the tie-break rules; only co-tag if the message clearly contains independent parts for multiple categories."""


# =============================================================================
# LITELLM ANNOTATOR (supports multiple providers)
# =============================================================================

class LiteLLMAnnotator(RubricAnnotator):
    """
    Annotator using LiteLLM for multi-provider support.
    
    Supports OpenAI, Anthropic, and other providers through LiteLLM.
    """
    
    def __init__(self, rubric_set: RubricSet, system_prompt: str, instruction_prompt: str,
                 model: str = "gpt-4o-mini", api_key: Optional[str] = None, **llm_kwargs):
        super().__init__(rubric_set, system_prompt, instruction_prompt)
        self.model = model
        self.api_key = api_key
        self.llm_kwargs = llm_kwargs
        
        # Import litellm here to avoid hard dependency
        try:
            import litellm
            self.litellm = litellm
        except ImportError:
            raise ImportError("litellm is required for LiteLLMAnnotator. Install with: pip install litellm")
    
    def _call_llm(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], 
                  tool_choice: Dict[str, Any], temperature: float = 0.0, **kwargs) -> Any:
        """Call LiteLLM with the given parameters."""
        call_kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            **self.llm_kwargs,
            **kwargs
        }
        
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        
        return self.litellm.completion(**call_kwargs)


class LiteLLMMultiSampleAnnotator(MultiSampleAnnotator, LiteLLMAnnotator):
    """Multi-sample annotator using LiteLLM."""
    pass


# =============================================================================
# OPENAI ANNOTATOR
# =============================================================================

class OpenAIAnnotator(RubricAnnotator):
    """
    Annotator using OpenAI's API directly.
    """
    
    def __init__(self, rubric_set: RubricSet, system_prompt: str, instruction_prompt: str,
                 model: str = "gpt-4o-mini", api_key: Optional[str] = None, **client_kwargs):
        super().__init__(rubric_set, system_prompt, instruction_prompt)
        self.model = model
        
        # Import OpenAI here to avoid hard dependency
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key, **client_kwargs)
        except ImportError:
            raise ImportError("openai is required for OpenAIAnnotator. Install with: pip install openai")
    
    def _call_llm(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], 
                  tool_choice: Dict[str, Any], temperature: float = 0.0, **kwargs) -> Any:
        """Call OpenAI API with the given parameters."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            **kwargs
        )


class OpenAIMultiSampleAnnotator(MultiSampleAnnotator, OpenAIAnnotator):
    """Multi-sample annotator using OpenAI."""
    pass


# =============================================================================
# SPECIALIZED ANNOTATORS
# =============================================================================

class SolvabilityAnnotator(LiteLLMAnnotator):
    """
    Specialized annotator for solvability analysis.
    
    Uses the solvability rubric and appropriate prompts by default.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, 
                 custom_features: Optional[List[Dict[str, str]]] = None, **kwargs):
        from .rubrics import create_solvability_rubric, DEFAULT_SOLVABILITY_FEATURES
        
        features = custom_features or DEFAULT_SOLVABILITY_FEATURES
        rubric_set = create_solvability_rubric(features)
        
        super().__init__(
            rubric_set=rubric_set,
            system_prompt=SOLVABILITY_SYSTEM_PROMPT,
            instruction_prompt=SOLVABILITY_INSTRUCTION_PROMPT,
            model=model,
            api_key=api_key,
            **kwargs
        )


class ConversationAnnotator(LiteLLMAnnotator):
    """
    Specialized annotator for conversation analysis.
    
    Uses the conversation rubric and appropriate prompts by default.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 include_timing: bool = True, include_task_type: bool = True, **kwargs):
        from .rubrics import create_conversation_rubric
        
        rubric_set = create_conversation_rubric(
            include_timing=include_timing,
            include_task_type=include_task_type
        )
        
        super().__init__(
            rubric_set=rubric_set,
            system_prompt=CONVERSATION_SYSTEM_PROMPT,
            instruction_prompt=CONVERSATION_INSTRUCTION_PROMPT,
            model=model,
            api_key=api_key,
            **kwargs
        )


class CustomAnnotator(LiteLLMAnnotator):
    """
    Fully customizable annotator.
    
    Allows complete control over rubric set, prompts, and model configuration.
    """
    
    def __init__(self, rubric_set: RubricSet, system_prompt: str, instruction_prompt: str,
                 model: str = "gpt-4o-mini", api_key: Optional[str] = None, **kwargs):
        super().__init__(
            rubric_set=rubric_set,
            system_prompt=system_prompt,
            instruction_prompt=instruction_prompt,
            model=model,
            api_key=api_key,
            **kwargs
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_solvability_annotator(model: str = "gpt-4o-mini", **kwargs) -> SolvabilityAnnotator:
    """Create a solvability annotator with sensible defaults."""
    return SolvabilityAnnotator(model=model, **kwargs)


def create_conversation_annotator(model: str = "gpt-4o-mini", **kwargs) -> ConversationAnnotator:
    """Create a conversation annotator with sensible defaults.""" 
    return ConversationAnnotator(model=model, **kwargs)


def create_multi_sample_solvability_annotator(model: str = "gpt-4o-mini", **kwargs) -> LiteLLMMultiSampleAnnotator:
    """Create a multi-sample solvability annotator."""
    from .rubrics import create_solvability_rubric, DEFAULT_SOLVABILITY_FEATURES
    
    features = kwargs.pop('custom_features', DEFAULT_SOLVABILITY_FEATURES)
    rubric_set = create_solvability_rubric(features)
    
    return LiteLLMMultiSampleAnnotator(
        rubric_set=rubric_set,
        system_prompt=SOLVABILITY_SYSTEM_PROMPT,
        instruction_prompt=SOLVABILITY_INSTRUCTION_PROMPT,
        model=model,
        **kwargs
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base annotators
    "LiteLLMAnnotator",
    "LiteLLMMultiSampleAnnotator",
    "OpenAIAnnotator", 
    "OpenAIMultiSampleAnnotator",
    
    # Specialized annotators
    "SolvabilityAnnotator",
    "ConversationAnnotator",
    "CustomAnnotator",
    
    # Convenience functions
    "create_solvability_annotator",
    "create_conversation_annotator",
    "create_multi_sample_solvability_annotator",
    
    # Prompts (for customization)
    "SOLVABILITY_SYSTEM_PROMPT",
    "SOLVABILITY_INSTRUCTION_PROMPT",
    "CONVERSATION_SYSTEM_PROMPT",
    "CONVERSATION_INSTRUCTION_PROMPT",
]