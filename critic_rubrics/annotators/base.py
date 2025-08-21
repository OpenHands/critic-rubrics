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
    
    def get_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the raw litellm completion request format.
        
        Args:
            request_data: Dict with 'messages_for_annotator' and 'tools_for_annotator' fields,
                         OR a simple string content for backward compatibility
        
        Returns:
            Dict in litellm completion format
        """
        # Handle backward compatibility - if it's a string, convert to message format
        if isinstance(request_data, str):
            messages = []
            system_message = self._get_system_message()
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": request_data})
            tools = [self._get_tool_schema()]
        else:
            # Use the provided messages and tools from the request data
            messages = request_data['messages_for_annotator']
            tools = [request_data['tools_for_annotator']]
        
        return {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "temperature": 0.1,
            "api_key": self.api_key,
        }
    
    def annotate(self, request_data) -> T:
        """Annotate content and return rubric result.
        
        Args:
            request_data: Dict with 'messages_for_annotator' and 'tools_for_annotator' fields,
                         OR a simple string content for backward compatibility
        """
        try:
            import litellm
            
            request = self.get_request(request_data)
            response = litellm.completion(**request)
            
            tool_call = response.choices[0].message.tool_calls[0]
            tool_call_args = json.loads(tool_call.function.arguments)
            return self._parse_result(tool_call_args)
            
        except ImportError:
            raise ImportError("'litellm' package is required for LLM calls")