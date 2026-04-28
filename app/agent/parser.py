"""
Parser for LLM responses with strict error handling and repair mechanism.
"""
import json
import re
from typing import Optional
from app.schemas.messages import LLMResponse


def repair_llm_output(raw_text: str) -> str:
    """
    Attempt to repair malformed JSON output from LLM.
    
    This function applies several repair strategies:
    1. Extract JSON from markdown code blocks
    2. Fix common JSON syntax errors (missing quotes, trailing commas)
    3. Balance brackets/braces
    4. Remove comments
    
    Args:
        raw_text: Raw text response from LLM
        
    Returns:
        Repaired JSON string, or original if repair fails
    """
    text = raw_text.strip()
    
    # Strategy 1: Extract JSON from markdown code blocks
    if "```json" in text or "```" in text:
        # Find content between code blocks
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
    
    # Strategy 2: Fix common JSON syntax errors
    # Remove trailing commas before closing brackets/braces
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Add quotes around unquoted keys (simple pattern)
    text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    
    # Strategy 3: Remove comments (both // and /* */ style)
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # Strategy 4: Balance brackets and braces
    open_braces = text.count('{')
    close_braces = text.count('}')
    if open_braces > close_braces:
        text += '}' * (open_braces - close_braces)
    
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    if open_brackets > close_brackets:
        text += ']' * (open_brackets - close_brackets)
    
    return text


class ResponseParser:
    """
    Parses structured JSON responses from the LLM.
    """
    
    @staticmethod
    def parse(response_text: str) -> LLMResponse:
        """
        Parse LLM response text into structured format.
        
        Args:
            response_text: Raw text response from LLM
            
        Returns:
            LLMResponse object
            
        Raises:
            ValueError: If response is invalid JSON or missing required fields
        """
        try:
            # Try to parse as JSON
            data = json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code blocks
            if "```json" in response_text or "```" in response_text:
                try:
                    # Extract content between code blocks
                    start = response_text.find("{")
                    end = response_text.rfind("}") + 1
                    if start != -1 and end > start:
                        json_str = response_text[start:end]
                        data = json.loads(json_str)
                    else:
                        raise ValueError(f"Could not extract JSON from response: {e}")
                except Exception as extract_error:
                    raise ValueError(f"Failed to extract JSON from code blocks: {extract_error}")
            else:
                raise ValueError(f"Invalid JSON response: {e}")
        
        # Validate required fields
        response_type = data.get("type")
        if not response_type:
            raise ValueError("Response missing 'type' field")
        
        if response_type not in ["thought", "action", "finish"]:
            raise ValueError(f"Invalid response type: {response_type}")
        
        # Validate type-specific fields
        if response_type == "action":
            if not data.get("tool_name"):
                raise ValueError("Action response missing 'tool_name'")
            if not data.get("arguments"):
                raise ValueError("Action response missing 'arguments'")
        
        if response_type == "finish":
            if not data.get("final_answer"):
                raise ValueError("Finish response missing 'final_answer'")
        
        if response_type == "thought" and not data.get("thought"):
            raise ValueError("Thought response missing 'thought'")
        
        # Create and return LLMResponse
        return LLMResponse(
            type=response_type,
            thought=data.get("thought"),
            tool_name=data.get("tool_name"),
            arguments=data.get("arguments"),
            final_answer=data.get("final_answer")
        )
    
    @staticmethod
    def parse_with_fallback(response_text: str) -> LLMResponse:
        """
        Parse response with fallback handling for malformed JSON.
        
        Args:
            response_text: Raw text response from LLM
            
        Returns:
            LLMResponse object, or a fallback response if parsing fails
        """
        try:
            return ResponseParser.parse(response_text)
        except ValueError as e:
            # Fallback: treat as a thought if parsing fails
            return LLMResponse(
                type="thought",
                thought=response_text.strip(),
                tool_name=None,
                arguments=None,
                final_answer=None
            )
    
    @staticmethod
    def parse_with_retry(response_text: str, max_retries: int = 2) -> tuple[LLMResponse, bool, str]:
        """
        Parse response with retry mechanism using repair function.
        
        Implements the three-tier strategy:
        1. First attempt: strict JSON parse
        2. If fail: retry with "repair prompt"
        3. If still fail: return structured error observation
        
        Args:
            response_text: Raw text response from LLM
            max_retries: Maximum number of repair attempts
            
        Returns:
            Tuple of (LLMResponse, success, error_message)
        """
        # Attempt 1: Strict parse
        try:
            return ResponseParser.parse(response_text), True, ""
        except ValueError as e:
            pass
        
        # Attempt 2: Repair and parse (up to max_retries)
        for attempt in range(max_retries):
            try:
                repaired_text = repair_llm_output(response_text)
                return ResponseParser.parse(repaired_text), True, ""
            except ValueError as e:
                if attempt == max_retries - 1:
                    # Final attempt failed, return error observation
                    error_response = LLMResponse(
                        type="thought",
                        thought=f"Parsing error: Could not parse LLM response after {max_retries} repair attempts. Original error: {str(e)}",
                        tool_name=None,
                        arguments=None,
                        final_answer=None
                    )
                    return error_response, False, f"Parse failed after {max_retries} repair attempts: {str(e)}"
        
        # Should not reach here, but return error if we do
        error_response = LLMResponse(
            type="thought",
            thought="Parsing error: Unexpected failure in retry mechanism",
            tool_name=None,
            arguments=None,
            final_answer=None
        )
        return error_response, False, "Unexpected parse failure"
