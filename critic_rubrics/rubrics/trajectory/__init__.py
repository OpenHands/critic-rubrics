from .rubric_impl import AnnotateConversationRubric
from .trajectory import (
    ANNOTATION_INSTRUCTION_MESSAGE as TRAJ_USER_MSG,
    ANNOTATION_SYSTEM_MESSAGE as TRAJ_SYS_MSG,
    FEATURES as TRAJ_FEATURES,
)
from .trajectory_with_user import (
    ANNOTATION_INSTRUCTION_MESSAGE as TRAJ_W_USER_USER_MSG,
    ANNOTATION_SYSTEM_MESSAGE as TRAJ_W_USER_SYS_MSG,
    FEATURES as TRAJ_USER_FEATURES,
)


TOOL_NAME = "annotate_conversation"
TOOL_DESCRIPTION = "Annotate agent conversation."
annotate_conversation_rubrics = AnnotateConversationRubric(
    tool_name=TOOL_NAME,
    tool_description=TOOL_DESCRIPTION,
    features=TRAJ_FEATURES,
    system_message=TRAJ_SYS_MSG,
    user_message=TRAJ_USER_MSG,
)

annotate_conversation_with_user_rubrics = AnnotateConversationRubric(
    tool_name=TOOL_NAME,
    tool_description=TOOL_DESCRIPTION,
    features=TRAJ_FEATURES + TRAJ_USER_FEATURES,
    system_message=TRAJ_W_USER_SYS_MSG,
    user_message=TRAJ_W_USER_USER_MSG,
)

def get_trajectory_level_rubrics(has_user_follow_up: bool) -> AnnotateConversationRubric:
    if has_user_follow_up:
        return annotate_conversation_with_user_rubrics
    return annotate_conversation_rubrics

__all__ = [
    "AnnotateConversationRubric",
    "annotate_conversation_rubrics",
    "annotate_conversation_with_user_rubrics",
    "get_trajectory_level_rubrics"
]
