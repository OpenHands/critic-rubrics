"""
Base annotator class for all rubric types.
"""

import json
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic

from pydantic import BaseModel

# Type variable for rubric result types
T = TypeVar('T', bound=BaseModel)


class BaseAnnotator(ABC, Generic[T]):
    """
    Base class for all rubric annotators.
    
    Provides common functionality for LLM-based annotation including:
    - Multi-sample analysis
    - Batch processing
    - Error handling
    - Result aggregation
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        instruction_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the base annotator.
        
        Args:
            model: LLM model to use
            api_key: API key for the LLM service
            base_url: Base URL for the LLM service
            system_prompt: System prompt for the LLM
            instruction_prompt: Instruction prompt for the LLM
            **kwargs: Additional arguments passed to LLM
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.instruction_prompt = instruction_prompt or self._get_default_instruction_prompt()
        self.llm_kwargs = kwargs
        
        # Initialize LLM client
        self._init_llm_client()
    
    def _init_llm_client(self):
        """Initialize the LLM client. Override in subclasses if needed."""
        try:
            import litellm
            self._use_litellm = True
        except ImportError:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                self._use_litellm = False
            except ImportError:
                raise ImportError(
                    "Either 'litellm' or 'openai' package is required. "
                    "Install with: pip install litellm or pip install openai"
                )
    
    @abstractmethod
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for this annotator type."""
        pass
    
    @abstractmethod
    def _get_default_instruction_prompt(self) -> str:
        """Get the default instruction prompt for this annotator type."""
        pass
    
    @abstractmethod
    def _get_tool_schema(self) -> Dict[str, Any]:
        """Get the tool schema for this annotator type."""
        pass
    
    @abstractmethod
    def _parse_result(self, tool_call_args: Dict[str, Any]) -> T:
        """Parse the LLM result into the appropriate rubric dataclass."""
        pass
    
    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: Dict[str, Any],
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call the LLM with the given parameters.
        
        Returns:
            The parsed tool call arguments
        """
        if self._use_litellm:
            import litellm
            
            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                api_key=self.api_key,
                base_url=self.base_url,
                **{**self.llm_kwargs, **kwargs}
            )
        else:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                **{**self.llm_kwargs, **kwargs}
            )
        
        # Extract tool call arguments
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    
    def annotate(self, content: str, **kwargs) -> T:
        """
        Annotate a single piece of content.
        
        Args:
            content: The content to annotate
            **kwargs: Additional arguments passed to LLM
            
        Returns:
            The rubric result dataclass
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{self.instruction_prompt}\n\n{content}"}
        ]
        
        tools = [self._get_tool_schema()]
        tool_choice = {"type": "function", "function": {"name": tools[0]["function"]["name"]}}
        
        try:
            result_args = self._call_llm(messages, tools, tool_choice, **kwargs)
            return self._parse_result(result_args)
        except Exception as e:
            raise RuntimeError(f"Failed to annotate content: {str(e)}")
    
    def annotate_batch(
        self,
        contents: List[str],
        max_workers: int = 3,
        **kwargs
    ) -> List[T]:
        """
        Annotate multiple pieces of content in parallel.
        
        Args:
            contents: List of content to annotate
            max_workers: Maximum number of parallel workers
            **kwargs: Additional arguments passed to LLM
            
        Returns:
            List of rubric result dataclasses
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.annotate, content, **kwargs): i
                for i, content in enumerate(contents)
            }
            
            # Collect results in order
            results = [None] * len(contents)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    print(f"Error processing item {index}: {str(e)}")
                    results[index] = None
        
        return [r for r in results if r is not None]
    
    def annotate_with_samples(
        self,
        content: str,
        samples: int = 5,
        temperature: float = 0.7,
        **kwargs
    ) -> 'MultiSampleResult[T]':
        """
        Annotate content multiple times for statistical analysis.
        
        Args:
            content: The content to annotate
            samples: Number of samples to generate
            temperature: Temperature for sampling diversity
            **kwargs: Additional arguments passed to LLM
            
        Returns:
            MultiSampleResult containing all samples and statistics
        """
        sample_results = []
        
        for i in range(samples):
            try:
                result = self.annotate(content, temperature=temperature, **kwargs)
                sample_results.append(result)
                
                # Small delay to avoid rate limiting
                if i < samples - 1:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error in sample {i+1}: {str(e)}")
                continue
        
        if not sample_results:
            raise RuntimeError("Failed to generate any successful samples")
        
        return MultiSampleResult(sample_results)


class MultiSampleResult(Generic[T]):
    """
    Container for multi-sample annotation results with statistical analysis.
    """
    
    def __init__(self, samples: List[T]):
        """
        Initialize with a list of sample results.
        
        Args:
            samples: List of rubric result dataclasses
        """
        self.samples = samples
        self.sample_count = len(samples)
    
    def get_consensus_result(self, threshold: float = 0.5) -> Optional[T]:
        """
        Get a consensus result based on majority voting.
        
        Args:
            threshold: Minimum agreement threshold (0.0-1.0)
            
        Returns:
            Consensus result or None if no consensus reached
        """
        if not self.samples:
            return None
        
        # This is a simplified implementation - subclasses should override
        # for more sophisticated consensus logic
        return self.samples[0]
    
    def get_sample_diversity(self) -> float:
        """
        Calculate diversity/entropy across samples.
        
        Returns:
            Diversity score (higher = more diverse)
        """
        # Simplified implementation - subclasses should override
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sample_count": self.sample_count,
            "samples": [sample.model_dump() for sample in self.samples],
            "consensus": self.get_consensus_result().model_dump() if self.get_consensus_result() else None,
            "diversity": self.get_sample_diversity()
        }