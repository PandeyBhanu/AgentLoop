"""
Agent loop implementing the ReAct pattern: Thought → Action → Observation.
"""
import asyncio
import time
from typing import Optional, Callable
from app.agent.state import AgentState
from app.agent.parser import ResponseParser
from app.agent.guardrails import Guardrails
from app.tools.registry import ToolRegistry
from app.schemas.tools import validate_tool_inputs
from app.schemas.messages import LLMResponse
from app.llm import LLMClient
from app.utils.logger import logger


class AgentLoop:
    """
    Implements the ReAct agent loop with proper error handling and guardrails.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        max_steps: int = 10,
        tool_timeout: int = 30,
        max_token_budget: Optional[int] = None,
        allowed_tools: Optional[list[str]] = None
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.tool_timeout = tool_timeout
        self.max_token_budget = max_token_budget
        self.allowed_tools = allowed_tools
        self.guardrails = Guardrails(max_steps=max_steps, tool_timeout=tool_timeout)
        self.total_tokens_used = 0
    
    async def run(
        self,
        user_message: str,
        on_step: Optional[Callable] = None
    ) -> AgentState:
        """
        Run the agent loop for a user message.
        
        Args:
            user_message: The user's input message
            on_step: Optional callback called after each step
            
        Returns:
            AgentState with the final result
        """
        state = AgentState(max_steps=self.max_steps)
        
        # Add user message to conversation
        state.add_message("user", user_message)
        
        # Add system prompt
        system_prompt = self._get_system_prompt()
        state.add_message("system", system_prompt)
        
        logger.info(f"Starting agent run {state.run_id}")
        
        try:
            while state.should_continue():
                state.increment_step()
                logger.log_step(state.current_step, "Starting iteration")
                
                # Get tool schemas (filter by allowed_tools if specified)
                if self.allowed_tools:
                    tool_schemas = [
                        schema for schema in self.tool_registry.get_all_definitions()
                        if schema["name"] in self.allowed_tools
                    ]
                else:
                    tool_schemas = self.tool_registry.get_all_definitions()
                
                # Send conversation to LLM with retry mechanism
                response_text, token_metadata = await self._call_llm_with_retry(
                    state=state,
                    tool_schemas=tool_schemas,
                    max_retries=2
                )
                
                # Track token usage and check budget
                tokens_used = token_metadata.get("total_tokens", 0)
                self.total_tokens_used += tokens_used
                state.total_cost += token_metadata.get("estimated_cost", 0)
                
                # Check token budget
                if self.max_token_budget and self.total_tokens_used > self.max_token_budget:
                    logger.error(f"Token budget exceeded: {self.total_tokens_used} > {self.max_token_budget}")
                    state.fail(f"Token budget exceeded: {self.total_tokens_used} > {self.max_token_budget}")
                    break
                
                # Log token usage
                logger.log_token_usage(
                    step=state.current_step,
                    prompt_tokens=token_metadata.get("prompt_tokens", 0),
                    completion_tokens=token_metadata.get("completion_tokens", 0),
                    total_tokens=tokens_used,
                    latency_ms=token_metadata.get("latency_ms", 0),
                    cost=token_metadata.get("estimated_cost", 0),
                    provider=token_metadata.get("provider", "unknown"),
                    model=token_metadata.get("model", "unknown")
                )
                
                # Parse LLM response with repair and retry
                response, parse_success, parse_error = ResponseParser.parse_with_retry(
                    response_text,
                    max_retries=2
                )
                
                if not parse_success:
                    logger.warning(f"Parse error after repair attempts: {parse_error}")
                    state.add_trace_entry(
                        entry_type="error",
                        content=f"Parsing error: {parse_error}",
                        step_number=state.current_step,
                        tokens_input=token_metadata.get("prompt_tokens", 0),
                        tokens_output=token_metadata.get("completion_tokens", 0),
                        latency_ms=token_metadata.get("latency_ms", 0),
                        error_details={"stage": "parsing", "message": parse_error, "recoverable": True}
                    )
                
                # Handle based on response type
                if response.type == "thought":
                    await self._handle_thought(state, response, token_metadata)
                elif response.type == "action":
                    await self._handle_action(state, response, token_metadata)
                elif response.type == "finish":
                    await self._handle_finish(state, response, token_metadata)
                    break
                
                # Call step callback if provided
                if on_step:
                    await on_step(state)
                
                # Small delay to prevent tight loops
                await asyncio.sleep(0.1)
            
            # Check if we exited without finishing
            if not state.is_finished:
                state.fail("Agent stopped without finishing (max steps reached)")
            
            logger.info(f"Agent run {state.run_id} completed: {state.current_step} steps")
            
        except Exception as e:
            logger.error(f"Agent run failed: {str(e)}")
            state.fail(str(e))
        
        return state
    
    async def _call_llm_with_retry(
        self,
        state: AgentState,
        tool_schemas: list[dict],
        max_retries: int = 2
    ) -> tuple[str, Optional[dict]]:
        """
        Call LLM with retry mechanism for API failures.
        
        Retries up to max_retries times if:
        - JSON parsing fails
        - Required fields missing
        - API call fails
        
        Args:
            state: Current agent state
            tool_schemas: Tool schemas to include in the call
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (response_text, token_usage)
        """
        for attempt in range(max_retries + 1):
            try:
                response_text, token_usage = await self.llm_client.send_messages(
                    messages=state.conversation_history,
                    tool_schemas=tool_schemas,
                    json_mode=True
                )
                
                # Try to validate the response has required fields
                import json
                try:
                    data = json.loads(response_text.strip())
                    if "type" not in data:
                        if attempt < max_retries:
                            logger.warning(f"LLM response missing 'type' field, retrying (attempt {attempt + 1}/{max_retries})")
                            continue
                except json.JSONDecodeError:
                    if attempt < max_retries:
                        logger.warning(f"LLM response not valid JSON, retrying (attempt {attempt + 1}/{max_retries})")
                        continue
                
                return response_text, token_usage
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"LLM call failed, retrying (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"LLM call failed after {max_retries} retries: {str(e)}")
                    raise
        
        # Should not reach here, but return error if we do
        raise Exception(f"LLM call failed after {max_retries} retries")
    
    async def _handle_thought(
        self,
        state: AgentState,
        response: LLMResponse,
        token_metadata: Optional[dict]
    ) -> None:
        """Handle a thought response from the LLM."""
        logger.info(f"Thought: {response.thought}")
        
        # Check for thought loops
        if response.thought:
            should_continue, loop_error = self.guardrails.check_thought_loop(response.thought)
            if not should_continue:
                logger.error(loop_error)
                state.add_trace_entry(
                    entry_type="error",
                    content=loop_error,
                    step_number=state.current_step,
                    tokens_input=token_metadata.get("prompt_tokens", 0),
                    tokens_output=token_metadata.get("completion_tokens", 0),
                    latency_ms=token_metadata.get("latency_ms", 0),
                    error_details={"stage": "guardrail", "message": loop_error, "recoverable": False}
                )
                state.fail(loop_error)
                return
        
        state.add_trace_entry(
            entry_type="thought",
            content=response.thought or "",
            step_number=state.current_step,
            tokens_input=token_metadata.get("prompt_tokens", 0),
            tokens_output=token_metadata.get("completion_tokens", 0),
            latency_ms=token_metadata.get("latency_ms", 0)
        )
        
        state.add_message("assistant", response.thought or "")
    
    async def _handle_action(
        self,
        state: AgentState,
        response: LLMResponse,
        token_metadata: Optional[dict]
    ) -> None:
        """Handle an action response from the LLM."""
        tool_name = response.tool_name
        arguments = response.arguments or {}
        
        logger.log_tool_call(tool_name, arguments)
        
        # Validate tool exists
        if not self.tool_registry.has_tool(tool_name):
            error_msg = f"Unknown tool: {tool_name}"
            logger.error(error_msg)
            state.add_trace_entry(
                entry_type="error",
                content=error_msg,
                step_number=state.current_step,
                tool_name=tool_name,
                tokens_input=token_metadata.get("prompt_tokens", 0),
                tokens_output=token_metadata.get("completion_tokens", 0),
                latency_ms=token_metadata.get("latency_ms", 0),
                error_details={"stage": "tool_execution", "message": error_msg, "recoverable": False}
            )
            state.add_message("assistant", f"Error: {error_msg}")
            return
        
        # Validate arguments against schema
        tool = self.tool_registry.get(tool_name)
        validation = validate_tool_inputs(arguments, tool.input_schema)
        
        if not validation.valid:
            error_msg = f"Invalid arguments: {', '.join(validation.errors)}"
            logger.error(error_msg)
            state.add_trace_entry(
                entry_type="error",
                content=error_msg,
                step_number=state.current_step,
                tool_name=tool_name,
                tokens_input=token_metadata.get("prompt_tokens", 0),
                tokens_output=token_metadata.get("completion_tokens", 0),
                latency_ms=token_metadata.get("latency_ms", 0),
                error_details={"stage": "tool_execution", "message": error_msg, "recoverable": True}
            )
            state.add_message("assistant", f"Error: {error_msg}")
            return
        
        # Check guardrails
        is_valid, guardrail_error = self.guardrails.validate_tool_call(tool_name, arguments)
        if not is_valid:
            logger.error(guardrail_error)
            state.add_trace_entry(
                entry_type="error",
                content=guardrail_error,
                step_number=state.current_step,
                tool_name=tool_name,
                tokens_input=token_metadata.get("prompt_tokens", 0),
                tokens_output=token_metadata.get("completion_tokens", 0),
                latency_ms=token_metadata.get("latency_ms", 0),
                error_details={"stage": "guardrail", "message": guardrail_error, "recoverable": False}
            )
            state.add_message("assistant", f"Error: {guardrail_error}")
            state.fail(guardrail_error)
            return
        
        # Execute tool with timeout
        state.add_trace_entry(
            entry_type="action",
            content=f"Calling {tool_name}",
            step_number=state.current_step,
            tool_name=tool_name,
            tokens_input=token_metadata.get("prompt_tokens", 0),
            tokens_output=token_metadata.get("completion_tokens", 0),
            latency_ms=token_metadata.get("latency_ms", 0)
        )
        state.add_message("assistant", f"I will use {tool_name} with arguments {arguments}")
        
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                tool.execute(arguments),
                timeout=self.tool_timeout
            )
            execution_time = time.time() - start_time
            
            # Convert result to string for observation
            observation = str(result)
            logger.log_observation(tool_name, observation)
            
            state.add_trace_entry(
                entry_type="observation",
                content=observation,
                step_number=state.current_step,
                tool_name=tool_name,
                execution_time=execution_time
            )
            
            state.add_message("assistant", f"Observation: {observation}")
            
        except asyncio.TimeoutError:
            error_msg = f"Tool execution timed out after {self.tool_timeout}s"
            logger.error(error_msg)
            state.add_trace_entry(
                entry_type="error",
                content=error_msg,
                step_number=state.current_step,
                tool_name=tool_name,
                execution_time=self.tool_timeout,
                error_details={"stage": "timeout", "message": error_msg, "recoverable": True}
            )
            state.add_message("assistant", f"Error: {error_msg}")
        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            logger.error(error_msg)
            state.add_trace_entry(
                entry_type="error",
                content=error_msg,
                step_number=state.current_step,
                tool_name=tool_name,
                error_details={"stage": "tool_execution", "message": error_msg, "recoverable": True}
            )
            state.add_message("assistant", f"Error: {error_msg}")
    
    async def _handle_finish(
        self,
        state: AgentState,
        response: LLMResponse,
        token_metadata: Optional[dict]
    ) -> None:
        """Handle a finish response from the LLM."""
        final_answer = response.final_answer or ""
        logger.info(f"Finish: {final_answer}")
        
        state.add_trace_entry(
            entry_type="finish",
            content=final_answer,
            step_number=state.current_step,
            tokens_input=token_metadata.get("prompt_tokens", 0),
            tokens_output=token_metadata.get("completion_tokens", 0),
            latency_ms=token_metadata.get("latency_ms", 0)
        )
        
        state.add_message("assistant", final_answer)
        state.finish(final_answer)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """You are a helpful AI assistant with access to tools. Follow the ReAct pattern:

1. Think about what you need to do
2. If you need information, use a tool
3. Observe the tool's result
4. Continue until you can answer the user's question

Your responses must be valid JSON with one of these types:

- thought: Share your reasoning
- action: Call a tool with arguments
- finish: Provide the final answer

Example responses:
{"type": "thought", "thought": "I need to calculate something"}
{"type": "action", "tool_name": "Calculator", "arguments": {"operation": "add", "a": 5, "b": 3}}
{"type": "finish", "final_answer": "The answer is 8"}

Available tools:
- Calculator: Perform arithmetic operations
- WebSearch: Search the web for information
- FileReader: Read text files from the local filesystem
- PythonExec: Execute Python code

Always return valid JSON. Think step by step."""
