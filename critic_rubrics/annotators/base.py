"""
Minimal base annotator class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic, Union
import json
import litellm
import logging

T = TypeVar('T')
logger = logging.getLogger(__name__)


class BaseAnnotator(ABC, Generic[T]):
    """Minimal base class for LLM-based rubric annotators."""
    
    def __init__(self, model: str = "openai/o3-2025-04-16", api_key: Optional[str] = None, *, temperature: float = 0.0, max_tokens: int = 8192, request_timeout: Optional[float] = None):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
    
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
    
    def get_request(self, request_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
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
            
            # Validate provided tool schema against annotator's schema
            provided_tool = request_data.get('tools_for_annotator')
            expected_tool = self._get_tool_schema()
            try:
                expected_props = set(expected_tool['function']['parameters']['properties'].keys())
                if provided_tool:
                    provided_props = set(provided_tool['function']['parameters']['properties'].keys())
                    if expected_props != provided_props:
                        logger.warning("Provided tool schema does not match annotator schema; using internal schema.")
                        tools = [expected_tool]
                    else:
                        tools = [provided_tool]
                else:
                    # No tool provided; quietly use internal schema
                    tools = [expected_tool]
            except Exception:
                logger.warning("Invalid provided tool schema; falling back to internal schema.")
                tools = [expected_tool]
        
        request = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key": self.api_key,
        }
        if self.request_timeout is not None:
            request["timeout"] = self.request_timeout
        return request

    
    def annotate(self, request_data: Union[str, Dict[str, Any]]) -> T:
        """Annotate content and return rubric result.
        
        Args:
            request_data: Dict with 'messages_for_annotator' and 'tools_for_annotator' fields,
                         OR a simple string content for backward compatibility
        """
        try:
            request = self.get_request(request_data)
            response = litellm.completion(**request)
            
            # Robust parsing with actionable errors
            choices = getattr(response, 'choices', None)
            if not choices:
                raise ValueError("No choices in response; ensure model supports tool calls and request is valid")
            message = getattr(choices[0], 'message', None)
            if message is None:
                raise ValueError("Malformed response: missing message in first choice")
            tool_calls = getattr(message, 'tool_calls', None)
            if not tool_calls:
                # Provide hint about tool usage
                raise ValueError("Model did not call any tools; ensure tool_choice='required' and schema matches expected")
            first_call = tool_calls[0]
            try:
                args_str = first_call.function.arguments
            except Exception:
                raise ValueError("Malformed tool call: missing function.arguments")
            try:
                tool_call_args = json.loads(args_str)
            except Exception as e:
                raise ValueError(f"Tool arguments are not valid JSON: {e}")
            
            return self._parse_result(tool_call_args)
            
        except ImportError:
            raise ImportError("'litellm' package is required for LLM calls")
