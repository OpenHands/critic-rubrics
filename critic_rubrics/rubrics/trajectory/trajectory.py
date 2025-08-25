from typing import Any, ClassVar, Literal

from litellm import ChatCompletionRequest
from pydantic import Field

from ...prediction import (
    BinaryPrediction,
    ClassificationPrediction,
    TextPrediction,
)
from ..base import BaseRubrics
from .converter import transform_for_annotator


SentimentPrediction = ClassificationPrediction[Literal["Positive", "Negative", "Neutral"]]

TaskTypePrediction = ClassificationPrediction[
    Literal[
        "Fix Bugs",
        "Implement Features",
        "Create Programs from Scratch",
        "Fix Failing Continuous Integration",
        "Fix Merge Conflicts",
        "Write Documentation",
        "Perform Deployments",
        "Perform Data Analysis",
    ]
]

DevClusterPrediction = ClassificationPrediction[
    Literal[
        "Web Development",
        "DevOps & Infrastructure",
        "AI Integration",
        "Code Management",
    ]
]


# %%
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
"""  # noqa: E501


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


class AnnotateConversationRubric(BaseRubrics):
    TOOL_NAME: ClassVar[str] = "annotate_conversation"
    TOOL_DESCRIPTION: ClassVar[str] = "Annotate agent conversation."
    SYSTEM_MESSAGE: ClassVar[str] = ANNOTATION_SYSTEM_MESSAGE
    USER_MESSAGE: ClassVar[str | None] = ANNOTATION_INSTRUCTION_MESSAGE


    # --- Generic Questions ---
    user_goal_summary: TextPrediction = Field(description="One sentence describing what the user is trying to accomplish.")
    overall_sentiment: SentimentPrediction = Field(description="Classify the overall sentiment of the user's messages.")
    task_type: TaskTypePrediction = Field(
        description=(
            "Classify the type of task into exactly one category."
            "Choose from: Fix Bugs, Implement Features, Create Programs from Scratch, "
            "Fix Failing Continuous Integration, Fix Merge Conflicts, Write Documentation, "
            "Perform Deployments, Perform Data Analysis."
        )
    )
    dev_cluster: DevClusterPrediction = Field(
        description=(
            "Choose the best-fitting development cluster: "
            "Web Development (frontend/backend, UI/UX, e-commerce), "
            "DevOps & Infrastructure (CI/CD, Docker/Kubernetes, cloud, env config), "
            "AI Integration (OpenAI/Anthropic/Gemini APIs, ML systems), "
            "Code Management (Git ops, PRs, docs, bug fixes, features)."
        )
    )

    # --- AGENT BEHAVIORAL ISSUES ---
    misunderstood_intention: BinaryPrediction = Field(
        description="Agent misunderstood the user’s goal/intent. Examples: User asked for a summary; agent produced a rewrite; user wanted high-level bullets; agent delivered full code."
    )
    did_not_follow_instruction: BinaryPrediction = Field(
        description=(
            "Agent ignored or failed to comply with explicit instructions/system constraints. "
            "Examples: User: 'Do NOT push to main.' Agent pushes; System says not to create a PR unless the user asks and the user didn't ask; "
            "agent creates a PR; user asked for bullet points only, agent gives long prose."
        )
    )
    insufficient_analysis: BinaryPrediction = Field(
        description=(
            "Didn’t explore existing materials (prior code/docs/examples) before acting. Examples: User points to an existing function/file that is relevant or already solves it; agent reinvents it."
        )
    )
    insufficient_clarification: BinaryPrediction = Field(
        description=(
            "Failed to ask necessary questions before acting when requirements were ambiguous. "
            "Examples: Agent proceeds despite unclear acceptance criteria (locales, time zones, error thresholds) then is corrected later."
        )
    )
    improper_tool_use_or_setup: BinaryPrediction = Field(
        description=(
            "Misused tools/commands or used inappropriate tools; missing/incorrect dependencies/setup. "
            "Examples: wrong command syntax; using an inappropriate tool; import errors; wrong API URL; malformed auth header."
        )
    )
    loop_behavior: BinaryPrediction = Field(description="Repeats the same failed action 3+ times without strategy change.")
    insufficient_testing: BinaryPrediction = Field(
        description=(
            "Skipped reasonable verification/tests for non-trivial or risky changes (trivial edits may be acceptable). "
            "Examples: No run/validation for a new parser; no check that a migration applies cleanly; no sanity check of output."
        )
    )
    insufficient_debugging: BinaryPrediction = Field(
        description="Did not investigate or reduce failing behavior when needed to make progress. Examples: Ignores stack trace; no isolation of failure; proceeds while errors persist."
    )
    incomplete_implementation: BinaryPrediction = Field(description="Delivered unfinished or non-functioning work. Examples: TODO/FIXME left; stub methods; code that cannot run.")
    file_management_errors: BinaryPrediction = Field(
        description="Wrong paths, overwrites, misplaced/extra (unnecessary) files. Examples: writes into wrong directory; overwrites config; creates unwanted artifacts."
    )
    scope_creep: BinaryPrediction = Field(description="Implemented unrequested features without approval. Examples: adds a dashboard or endpoint not asked for.")
    risky_actions_or_permission: BinaryPrediction = Field(
        description=(
            "Risky steps without the user's explicit consent. Examples: git push to main; deleting existing files in a repo (deleting files created by the agent itself is fine); altering credentials."
        )
    )
    other_agent_issue: BinaryPrediction = Field(description="Any other agent-side problem not covered above.")

    # --- INFRASTRUCTURE ---
    infrastructure_external_issue: BinaryPrediction = Field(
        description="Environment/platform limits outside agent control. Examples: provider outage; disk full on a managed runner; missing enterprise API key; network failure not caused by agent."
    )
    infrastructure_agent_caused_issue: BinaryPrediction = Field(
        description=(
            "Infrastructure faults introduced by the agent's prior actions. Examples: agent leaves server on port 8000 → later start on 8000 fails; agent fills disk with logs → later writes fail."
        )
    )

    @classmethod
    def create_annotation_request(
        cls,
        inputs: dict[str, Any],
        model: str = "openai/o3-2025-04-16",
    ) -> ChatCompletionRequest | None:
        assert cls.USER_MESSAGE is not None, "user_message must be defined for this rubrics"
        messages = transform_for_annotator(
            inputs,
            system_message=cls.SYSTEM_MESSAGE,
            annotation_instruction_message=cls.USER_MESSAGE,
        )
        if messages is None:
            return None
        return ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=0.0,
            tools=cls.tools(),
            tool_choice=cls.tool_choice(),
        )
