"""
Minimal base annotator class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic
import json

T = TypeVar('T')


class BaseAnnotator(ABC, Generic[T]):
    """Minimal base class for LLM-based rubric annotators."""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
    
    @abstractmethod
    def _get_tool_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling."""
        pass
    
    @abstractmethod
    def _parse_result(self, tool_call_args: Dict[str, Any]) -> T:
        """Parse LLM tool call result into rubric dataclass."""
        pass
    
    def _get_system_message(self) -> Optional[str]:
        """Get system message for the annotator. Override in subclasses."""
        return None
    
    def annotate(self, content: str) -> T:
        """Annotate content and return rubric result."""
        try:
            import litellm
            
            # Build messages
            messages = []
            system_message = self._get_system_message()
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": content})
            
            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=[self._get_tool_schema()],
                tool_choice="required",
                temperature=0.1,
                api_key=self.api_key,
            )
            
            tool_call = response.choices[0].message.tool_calls[0]
            tool_call_args = json.loads(tool_call.function.arguments)
            return self._parse_result(tool_call_args)
            
        except ImportError:
            raise ImportError("'litellm' package is required for LLM calls")