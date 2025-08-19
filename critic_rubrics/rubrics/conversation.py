"""
Conversation rubrics dataclass for general conversation analysis.
"""

from typing import Optional
from pydantic import BaseModel, Field

from ..core import Prediction


class ConversationRubrics(BaseModel):
    """
    Dataclass for general conversation analysis results.
    
    Focuses on conversation quality, flow, and effectiveness
    rather than specific task execution.
    """
    
    # === CONVERSATION FLOW ===
    
    natural_conversation_flow: Prediction = Field(
        description="Conversation flows naturally without awkward transitions"
    )
    
    appropriate_turn_taking: Prediction = Field(
        description="Participants take turns appropriately in the conversation"
    )
    
    maintains_context: Prediction = Field(
        description="Conversation maintains context throughout the interaction"
    )
    
    # === COMMUNICATION QUALITY ===
    
    clear_communication: Prediction = Field(
        description="Communication is clear and easy to understand"
    )
    
    appropriate_tone: Prediction = Field(
        description="Tone is appropriate for the context and relationship"
    )
    
    active_listening: Prediction = Field(
        description="Participants demonstrate active listening and engagement"
    )
    
    # === UNDERSTANDING AND COMPREHENSION ===
    
    mutual_understanding: Prediction = Field(
        description="Participants demonstrate mutual understanding"
    )
    
    clarification_when_needed: Prediction = Field(
        description="Participants seek clarification when needed"
    )
    
    acknowledges_responses: Prediction = Field(
        description="Participants acknowledge and respond to each other appropriately"
    )
    
    # === ENGAGEMENT AND RAPPORT ===
    
    engaged_participation: Prediction = Field(
        description="All participants are engaged and actively participating"
    )
    
    builds_rapport: Prediction = Field(
        description="Conversation builds rapport and positive relationship"
    )
    
    respectful_interaction: Prediction = Field(
        description="Interaction is respectful and professional"
    )
    
    # === EFFECTIVENESS ===
    
    achieves_communication_goals: Prediction = Field(
        description="Conversation achieves its intended communication goals"
    )
    
    efficient_information_exchange: Prediction = Field(
        description="Information is exchanged efficiently without unnecessary repetition"
    )
    
    productive_outcome: Prediction = Field(
        description="Conversation leads to a productive outcome or resolution"
    )
    
    # === ISSUES AND PROBLEMS ===
    
    communication_breakdown: Prediction = Field(
        description="Communication breakdown or misunderstanding occurred"
    )
    
    repetitive_or_circular: Prediction = Field(
        description="Conversation became repetitive or went in circles"
    )
    
    off_topic_drift: Prediction = Field(
        description="Conversation drifted significantly off topic"
    )
    
    # === ADDITIONAL CONTEXT ===
    
    conversation_length: Optional[str] = Field(
        default=None,
        description="Length category of the conversation (short, medium, long)"
    )
    
    conversation_type: Optional[str] = Field(
        default=None,
        description="Type of conversation (informational, problem-solving, social, etc.)"
    )
    
    participant_count: Optional[int] = Field(
        default=None,
        description="Number of participants in the conversation"
    )
    
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any additional observations about the conversation"
    )
    
    def get_quality_score(self) -> float:
        """Calculate overall conversation quality score."""
        positive_indicators = [
            self.natural_conversation_flow,
            self.appropriate_turn_taking,
            self.maintains_context,
            self.clear_communication,
            self.appropriate_tone,
            self.active_listening,
            self.mutual_understanding,
            self.clarification_when_needed,
            self.acknowledges_responses,
            self.engaged_participation,
            self.builds_rapport,
            self.respectful_interaction,
            self.achieves_communication_goals,
            self.efficient_information_exchange,
            self.productive_outcome,
        ]
        
        negative_indicators = [
            self.communication_breakdown,
            self.repetitive_or_circular,
            self.off_topic_drift,
        ]
        
        positive_count = sum(1 for p in positive_indicators if p.detected)
        negative_count = sum(1 for p in negative_indicators if p.detected)
        
        # Score based on positive indicators minus negative ones
        total_possible = len(positive_indicators)
        score = (positive_count - negative_count) / total_possible
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
    def get_issues(self) -> list[str]:
        """Get list of conversation issues that were detected."""
        issues = []
        if self.communication_breakdown.detected:
            issues.append("communication_breakdown")
        if self.repetitive_or_circular.detected:
            issues.append("repetitive_or_circular")
        if self.off_topic_drift.detected:
            issues.append("off_topic_drift")
        return issues
    
    def get_strengths(self) -> list[str]:
        """Get list of conversation strengths that were detected."""
        strengths = []
        strength_mapping = [
            ("natural_conversation_flow", self.natural_conversation_flow),
            ("appropriate_turn_taking", self.appropriate_turn_taking),
            ("maintains_context", self.maintains_context),
            ("clear_communication", self.clear_communication),
            ("appropriate_tone", self.appropriate_tone),
            ("active_listening", self.active_listening),
            ("mutual_understanding", self.mutual_understanding),
            ("clarification_when_needed", self.clarification_when_needed),
            ("acknowledges_responses", self.acknowledges_responses),
            ("engaged_participation", self.engaged_participation),
            ("builds_rapport", self.builds_rapport),
            ("respectful_interaction", self.respectful_interaction),
            ("achieves_communication_goals", self.achieves_communication_goals),
            ("efficient_information_exchange", self.efficient_information_exchange),
            ("productive_outcome", self.productive_outcome),
        ]
        
        for name, prediction in strength_mapping:
            if prediction.detected:
                strengths.append(name)
        
        return strengths