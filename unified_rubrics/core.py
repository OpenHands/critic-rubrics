"""
Core classes for the unified rubrics system.
"""

import json
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union
from collections import Counter
from math import log2

from pydantic import BaseModel, Field


class RubricItem(BaseModel):
    """
    A single rubric item that can be evaluated.
    
    This is the basic unit of evaluation - it represents one specific
    aspect or feature that can be detected and annotated.
    """
    identifier: str = Field(..., description="Unique identifier for this rubric item")
    description: str = Field(..., description="Human-readable description of what this item detects")
    category: Optional[str] = Field(None, description="Category this item belongs to")
    requires_rationale: bool = Field(True, description="Whether this item requires a rationale explanation")
    
    @property
    def detection_field_name(self) -> str:
        """Name of the boolean detection field in the tool schema."""
        return f"{self.identifier}_detected"
    
    @property 
    def rationale_field_name(self) -> str:
        """Name of the rationale field in the tool schema."""
        return f"{self.identifier}_rationale"
    
    def to_tool_schema_properties(self) -> Dict[str, Dict[str, Any]]:
        """Convert this rubric item to OpenAI tool schema properties."""
        properties = {
            self.detection_field_name: {
                "type": "boolean",
                "description": self.description
            }
        }
        
        if self.requires_rationale:
            properties[self.rationale_field_name] = {
                "type": "string", 
                "description": f"Rationale for {self.identifier} detection. Provide specific evidence and brief explanation."
            }
            
        return properties


class RubricCategory(BaseModel):
    """
    A category of related rubric items.
    
    Categories help organize rubrics and can have special behaviors
    like mutual exclusivity rules.
    """
    name: str = Field(..., description="Category name")
    description: str = Field(..., description="Description of this category")
    items: List[RubricItem] = Field(default_factory=list, description="Items in this category")
    mutually_exclusive: bool = Field(False, description="Whether items in this category are mutually exclusive")
    max_selections: Optional[int] = Field(None, description="Maximum number of items that can be selected from this category")
    
    def add_item(self, item: RubricItem) -> None:
        """Add an item to this category."""
        item.category = self.name
        self.items.append(item)
    
    def get_item_identifiers(self) -> List[str]:
        """Get all item identifiers in this category."""
        return [item.identifier for item in self.items]


class RubricSet(BaseModel):
    """
    A complete set of rubrics for a specific annotation task.
    
    This combines multiple categories and individual items into a cohesive
    evaluation framework.
    """
    name: str = Field(..., description="Name of this rubric set")
    description: str = Field(..., description="Description of what this rubric set evaluates")
    categories: List[RubricCategory] = Field(default_factory=list, description="Categories in this rubric set")
    standalone_items: List[RubricItem] = Field(default_factory=list, description="Items not belonging to any category")
    additional_fields: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Additional non-rubric fields (e.g., task_type, timing)")
    
    def add_category(self, category: RubricCategory) -> None:
        """Add a category to this rubric set."""
        self.categories.append(category)
    
    def add_item(self, item: RubricItem, category_name: Optional[str] = None) -> None:
        """Add an item to this rubric set, optionally to a specific category."""
        if category_name:
            category = self.get_category(category_name)
            if category:
                category.add_item(item)
            else:
                raise ValueError(f"Category '{category_name}' not found")
        else:
            self.standalone_items.append(item)
    
    def get_category(self, name: str) -> Optional[RubricCategory]:
        """Get a category by name."""
        for category in self.categories:
            if category.name == name:
                return category
        return None
    
    def get_all_items(self) -> List[RubricItem]:
        """Get all rubric items across all categories and standalone items."""
        items = list(self.standalone_items)
        for category in self.categories:
            items.extend(category.items)
        return items
    
    def get_all_identifiers(self) -> List[str]:
        """Get all rubric item identifiers."""
        return [item.identifier for item in self.get_all_items()]
    
    def to_tool_schema(self, function_name: str = "annotate", function_description: str = "Annotate the content using the rubric") -> Dict[str, Any]:
        """Convert this rubric set to an OpenAI function calling tool schema."""
        properties = {}
        required_fields = []
        
        # Add additional fields first
        for field_name, field_schema in self.additional_fields.items():
            properties[field_name] = field_schema
            if field_schema.get("required", False):
                required_fields.append(field_name)
        
        # Add rubric items
        for item in self.get_all_items():
            item_properties = item.to_tool_schema_properties()
            properties.update(item_properties)
            
            # All detection fields are required
            required_fields.append(item.detection_field_name)
            if item.requires_rationale:
                required_fields.append(item.rationale_field_name)
        
        return {
            "type": "function",
            "function": {
                "name": function_name,
                "description": function_description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": sorted(required_fields)
                }
            }
        }


class AnnotationResult(BaseModel):
    """
    Result of a rubric annotation.
    
    Contains the detected items, rationales, and metadata about the annotation process.
    """
    rubric_set_name: str = Field(..., description="Name of the rubric set used")
    detections: Dict[str, bool] = Field(default_factory=dict, description="Boolean detections for each rubric item")
    rationales: Dict[str, str] = Field(default_factory=dict, description="Rationales for detected items")
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Additional fields like task_type, timing, etc.")
    
    # Metadata
    prompt_tokens: Optional[int] = Field(None, description="Number of prompt tokens used")
    completion_tokens: Optional[int] = Field(None, description="Number of completion tokens used")
    response_latency: Optional[float] = Field(None, description="Response latency in seconds")
    
    def get_detected_items(self) -> List[str]:
        """Get list of detected rubric item identifiers."""
        return [identifier for identifier, detected in self.detections.items() if detected]
    
    def get_detection_rate(self) -> float:
        """Get the overall detection rate (fraction of items detected)."""
        if not self.detections:
            return 0.0
        return sum(self.detections.values()) / len(self.detections)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a flat dictionary suitable for analysis."""
        result = {
            "rubric_set_name": self.rubric_set_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "response_latency": self.response_latency,
            **self.detections,
            **self.additional_data
        }
        
        # Add rationales with _rationale suffix
        for identifier, rationale in self.rationales.items():
            result[f"{identifier}_rationale"] = rationale
            
        return result


class RubricAnnotator(ABC):
    """
    Abstract base class for rubric-based annotators.
    
    This defines the interface that all annotators must implement,
    while allowing for different LLM backends and annotation strategies.
    """
    
    def __init__(self, rubric_set: RubricSet, system_prompt: str, instruction_prompt: str):
        self.rubric_set = rubric_set
        self.system_prompt = system_prompt
        self.instruction_prompt = instruction_prompt
        self._tool_schema = rubric_set.to_tool_schema()
    
    @abstractmethod
    def _call_llm(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], 
                  tool_choice: Dict[str, Any], temperature: float = 0.0, **kwargs) -> Any:
        """
        Call the LLM with the given parameters.
        
        This method must be implemented by subclasses to handle the specific
        LLM API being used (OpenAI, Anthropic, etc.).
        """
        pass
    
    def _extract_annotation_from_response(self, response: Any) -> Dict[str, Any]:
        """
        Extract the annotation data from the LLM response.
        
        This method should be overridden by subclasses if they need custom
        response parsing logic.
        """
        # Default implementation assumes OpenAI-style response
        try:
            tool_call = response.choices[0].message.tool_calls[0]
            return json.loads(tool_call.function.arguments)
        except (AttributeError, IndexError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to extract annotation from response: {e}")
    
    def _get_usage_info(self, response: Any) -> tuple[Optional[int], Optional[int]]:
        """Extract token usage information from the response."""
        try:
            usage = response.usage
            return usage.prompt_tokens, usage.completion_tokens
        except AttributeError:
            return None, None
    
    def annotate(self, content: str, temperature: float = 0.0, **kwargs) -> AnnotationResult:
        """
        Annotate content using the rubric set.
        
        Args:
            content: The content to annotate
            temperature: Sampling temperature for the LLM
            **kwargs: Additional arguments passed to the LLM
            
        Returns:
            AnnotationResult containing the annotation
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{content}\n\n{self.instruction_prompt}"}
        ]
        
        tool_choice = {
            "type": "function",
            "function": {"name": self._tool_schema["function"]["name"]}
        }
        
        start_time = time.time()
        response = self._call_llm(
            messages=messages,
            tools=[self._tool_schema],
            tool_choice=tool_choice,
            temperature=temperature,
            **kwargs
        )
        end_time = time.time()
        
        # Extract annotation data
        annotation_data = self._extract_annotation_from_response(response)
        
        # Parse into detections, rationales, and additional data
        detections = {}
        rationales = {}
        additional_data = {}
        
        all_identifiers = self.rubric_set.get_all_identifiers()
        
        for key, value in annotation_data.items():
            if key.endswith("_detected"):
                identifier = key[:-9]  # Remove "_detected" suffix
                if identifier in all_identifiers:
                    detections[identifier] = value
            elif key.endswith("_rationale"):
                identifier = key[:-10]  # Remove "_rationale" suffix
                if identifier in all_identifiers:
                    rationales[identifier] = value
            else:
                # Additional field (like task_type, timing, etc.)
                additional_data[key] = value
        
        # Get usage info
        prompt_tokens, completion_tokens = self._get_usage_info(response)
        
        return AnnotationResult(
            rubric_set_name=self.rubric_set.name,
            detections=detections,
            rationales=rationales,
            additional_data=additional_data,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            response_latency=end_time - start_time
        )
    
    def annotate_batch(self, contents: List[str], temperature: float = 0.0, 
                      max_workers: int = 5, **kwargs) -> List[AnnotationResult]:
        """
        Annotate multiple contents in parallel.
        
        Args:
            contents: List of contents to annotate
            temperature: Sampling temperature for the LLM
            max_workers: Maximum number of parallel workers
            **kwargs: Additional arguments passed to the LLM
            
        Returns:
            List of AnnotationResults in the same order as input
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.annotate, content, temperature, **kwargs): i
                for i, content in enumerate(contents)
            }
            
            # Collect results in order
            results = [None] * len(contents)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
                
            return results


class MultiSampleAnnotator(RubricAnnotator):
    """
    An annotator that generates multiple samples for more robust annotations.
    
    This is similar to the Calvin featurizer approach, where multiple samples
    are generated with temperature > 0 to get a distribution of annotations.
    """
    
    def annotate_with_samples(self, content: str, samples: int = 10, temperature: float = 1.0, 
                            **kwargs) -> 'MultiSampleResult':
        """
        Generate multiple annotation samples for more robust results.
        
        Args:
            content: The content to annotate
            samples: Number of samples to generate
            temperature: Sampling temperature (should be > 0 for diversity)
            **kwargs: Additional arguments passed to the LLM
            
        Returns:
            MultiSampleResult containing all samples and aggregated statistics
        """
        sample_results = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency = 0.0
        
        for _ in range(samples):
            result = self.annotate(content, temperature=temperature, **kwargs)
            sample_results.append(result)
            
            if result.prompt_tokens:
                total_prompt_tokens += result.prompt_tokens
            if result.completion_tokens:
                total_completion_tokens += result.completion_tokens
            if result.response_latency:
                total_latency += result.response_latency
        
        return MultiSampleResult(
            rubric_set_name=self.rubric_set.name,
            samples=sample_results,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_latency=total_latency
        )


class MultiSampleResult(BaseModel):
    """
    Result of multi-sample annotation.
    
    Contains multiple annotation samples and provides methods to aggregate
    and analyze the results.
    """
    rubric_set_name: str
    samples: List[AnnotationResult]
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency: float = 0.0
    
    def get_detection_rates(self) -> Dict[str, float]:
        """Get detection rate for each rubric item across all samples."""
        if not self.samples:
            return {}
        
        all_identifiers = set()
        for sample in self.samples:
            all_identifiers.update(sample.detections.keys())
        
        rates = {}
        for identifier in all_identifiers:
            detections = [
                sample.detections.get(identifier, False) 
                for sample in self.samples
            ]
            rates[identifier] = sum(detections) / len(detections)
        
        return rates
    
    def get_detection_entropy(self) -> Dict[str, float]:
        """Calculate entropy of detections for each rubric item."""
        if not self.samples:
            return {}
        
        all_identifiers = set()
        for sample in self.samples:
            all_identifiers.update(sample.detections.keys())
        
        entropies = {}
        for identifier in all_identifiers:
            detections = [
                sample.detections.get(identifier, False)
                for sample in self.samples
            ]
            
            counts = Counter(detections)
            total = len(detections)
            
            if total == 0:
                entropies[identifier] = 0.0
                continue
                
            entropy = -sum(
                (count / total) * log2(count / total)
                for count in counts.values()
                if count > 0
            )
            entropies[identifier] = entropy
        
        return entropies
    
    def get_consensus_result(self, threshold: float = 0.5) -> AnnotationResult:
        """
        Get a consensus annotation result based on detection rates.
        
        Args:
            threshold: Minimum detection rate to consider an item as detected
            
        Returns:
            AnnotationResult representing the consensus
        """
        detection_rates = self.get_detection_rates()
        
        # Determine consensus detections
        consensus_detections = {
            identifier: rate >= threshold
            for identifier, rate in detection_rates.items()
        }
        
        # For rationales, use the most common one for detected items
        consensus_rationales = {}
        for identifier, detected in consensus_detections.items():
            if detected:
                rationales = [
                    sample.rationales.get(identifier, "")
                    for sample in self.samples
                    if sample.detections.get(identifier, False)
                ]
                if rationales:
                    # Use the most common rationale (simple approach)
                    rationale_counts = Counter(rationales)
                    consensus_rationales[identifier] = rationale_counts.most_common(1)[0][0]
        
        # For additional data, use the most common values
        consensus_additional = {}
        if self.samples:
            for key in self.samples[0].additional_data.keys():
                values = [sample.additional_data.get(key) for sample in self.samples]
                value_counts = Counter(values)
                consensus_additional[key] = value_counts.most_common(1)[0][0]
        
        return AnnotationResult(
            rubric_set_name=self.rubric_set_name,
            detections=consensus_detections,
            rationales=consensus_rationales,
            additional_data=consensus_additional,
            prompt_tokens=self.total_prompt_tokens,
            completion_tokens=self.total_completion_tokens,
            response_latency=self.total_latency
        )
    
    def to_dataframe(self):
        """Convert samples to a pandas DataFrame for analysis."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")
        
        rows = []
        for i, sample in enumerate(self.samples):
            row = {"sample_id": i, **sample.to_dict()}
            rows.append(row)
        
        return pd.DataFrame(rows)