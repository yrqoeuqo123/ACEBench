# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import re
import sys
from dataclasses import asdict, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra

from nemo_skills.inference.generate import (
    GenerateSolutionsConfig,
    GenerationTask,
    InferenceConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.inference.model.base import EndpointType
from nemo_skills.inference.model.utils import is_context_window_exceeded_error
from nemo_skills.prompt.utils import get_token_count
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class ACEBenchGenerationConfig(GenerateSolutionsConfig):
    """ACEBench benchmark generation."""

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    # Number of attempts for generation (for best-of-N sampling)
    attempts: int = 1
    
    # Language for prompts ("en" for English, "zh" for Chinese)
    language: str = "en"

    def _post_init_validate_params(self):
        """Validate that certain parameters are restricted to certain values"""
        if self.prompt_format not in ["ns", "openai"]:
            raise ValueError(f"prompt_format must be either 'ns' or 'openai', got '{self.prompt_format}'")

        if self.prompt_format == "openai":
            assert self.prompt_config is None, "prompt_config is not supported for prompt_format == 'openai'"

        for param, default_value in self._get_disallowed_params():
            if getattr(self, param) != default_value:
                raise ValueError(f"{param} must be {default_value}")

    def _get_disallowed_params(self):
        """Returns a list of parameters with their default values to check that they are not changed from the defaults"""
        return [
            ("prompt_config", None),
        ]


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_acebench_generation_config", node=ACEBenchGenerationConfig)


def normalize_schema(schema: Dict) -> Dict:
    """Normalize ACEBench schema to be OpenAI-compatible."""
    if not isinstance(schema, dict):
        return schema
    
    normalized = {}
    for key, value in schema.items():
        if key == "type" and value == "dict":
            # OpenAI requires 'object' not 'dict'
            normalized[key] = "object"
        elif isinstance(value, dict):
            # Recursively normalize nested schemas
            normalized[key] = normalize_schema(value)
        elif isinstance(value, list):
            # Normalize list items if they're dicts
            normalized[key] = [normalize_schema(item) if isinstance(item, dict) else item for item in value]
        else:
            normalized[key] = value
    
    return normalized


def convert_functions_to_tools(functions: List[Dict]) -> List[Dict]:
    """Convert ACEBench function format to OpenAI tools format."""
    tools = []
    for func in functions:
        # Normalize function name (max 64 chars for OpenAI)
        func_name = func.get("name", "")
        if len(func_name) > 64:
            func_name = func_name[:64]
        
        # Normalize parameters schema
        parameters = normalize_schema(func.get("parameters", {}))
        
        tool = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": func.get("description", ""),
                "parameters": parameters,
            },
        }
        tools.append(tool)
    return tools


def format_acebench_messages(
    data_point: Dict[str, Any],
    use_text_format: bool = True,
    language: str = "en",
) -> tuple[List[Dict[str, Any]], Optional[List[Dict]]]:
    """
    Format ACEBench data point into chat messages.
    
    Args:
        data_point: Data point containing question, functions, etc.
        use_text_format: Whether to use text-based format (True) or OpenAI tools format (False)
        language: Language for prompts ("en" or "zh")
    
    Returns:
        tuple: (messages, tools) where tools is None for text format
    """
    from nemo_skills.inference.eval.acebench_prompts import get_system_prompt, get_user_prompt
    
    question = data_point.get("question", "")
    functions = data_point.get("function", [])
    time = data_point.get("time", "")
    profile = data_point.get("profile", {})
    sample_id = data_point.get("id", "")
    
    # Determine category from ID
    category = ""
    if sample_id:
        sample_id_lower = sample_id.lower()
        if "special" in sample_id_lower:
            category = "special"
        elif "agent" in sample_id_lower:
            category = "agent"
        elif "preference" in sample_id_lower:
            category = "preference"
        else:
            category = "normal"
    
    if use_text_format:
        system_prompt = get_system_prompt(category, functions, time=time, profile=profile, language=language)
        user_prompt = get_user_prompt(question, language=language)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        return messages, None  # No tools for text format
    
    else:
        # OpenAI tools format (original approach)
        messages = []
        
        # Add conversation history if available
        if "previous_conversation_history" in data_point and data_point["previous_conversation_history"]:
            history = data_point["previous_conversation_history"]
            if isinstance(history, str):
                try:
                    history_obj = json.loads(history)
                    if isinstance(history_obj, list):
                        messages.extend(history_obj)
                    else:
                        messages.append({"role": "user", "content": history})
                except json.JSONDecodeError:
                    messages.append({"role": "user", "content": history})
            elif isinstance(history, list):
                messages.extend(history)
        
        # Add the current question/user input
        if question:
            if question.startswith("user: "):
                question = question[6:].strip()
            messages.append({"role": "user", "content": question})
        
        # Convert functions to tools
        tools = convert_functions_to_tools(functions) if functions else None
        
        return messages, tools


class ACEBenchGenerationTask(GenerationTask):
    """Generation task for ACEBench benchmark."""

    def __init__(self, cfg: ACEBenchGenerationConfig):
        super().__init__(cfg)
        # Set up tokenizer for prompt formatting if needed
        if cfg.tokenizer:
            try:
                from transformers import AutoTokenizer
                self.hf_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
            except (ImportError, OSError) as e:
                LOG.warning(f"Could not load tokenizer: {e}")
                self.hf_tokenizer = None
        else:
            self.hf_tokenizer = None

    def log_example_prompt(self, data):
        """Log an example prompt for debugging."""
        # data might be a list of samples, get the first one
        if isinstance(data, list):
            if not data:
                LOG.warning("Empty data provided to log_example_prompt")
                return
            data_point = data[0]
        else:
            data_point = data
            
        messages = format_acebench_messages(data_point, self.hf_tokenizer)
        functions = data_point.get("function", [])
        tools = convert_functions_to_tools(functions) if functions else None

        LOG.info("Example ACEBench prompt:")
        LOG.info(f"Messages: {json.dumps(messages, indent=2)}")
        if tools:
            LOG.info(f"Tools: {json.dumps(tools[:1], indent=2)}...")  # Log first tool only
        return

    def setup_prompt(self):
        return None

    async def process_agent_datapoint(self, data_point, language):
        """Process agent task using integrated simulation infrastructure."""
        import asyncio
        import os
        import time
        
        # Start timing
        start_time = time.time()
        
        def run_agent_simulation():
            try:
                from nemo_skills.inference.eval.agent_simulation.apimodel_inference import APIModelInference
                
                # Set environment variables for agent simulation
                os.environ["GPT_API_KEY"] = self.cfg.server.get("api_key", "")
                os.environ["GPT_AGENT_API_KEY"] = self.cfg.server.get("api_key", "")
                os.environ["GPT_BASE_URL"] = self.cfg.server.get("base_url", "")
                
                # Get full model name (keep openai/ prefix for server compatibility)
                model_name = self.cfg.server.get("model", "")
                
                LOG.info(f"Running agent simulation for {data_point.get('id', '')} with model {model_name}")
                
                # Create agent
                agent = APIModelInference(
                    model_name=model_name,
                    temperature=float(self.cfg.inference.temperature),
                    top_p=float(self.cfg.inference.top_p),
                    max_tokens=int(self.cfg.inference.tokens_to_generate or 8192),
                    max_dialog_turns=40,
                    user_model=model_name,
                    language=language,
                )
                
                # Run agent inference
                result = agent.inference(
                    question=data_point.get("question", ""),
                    functions=data_point.get("function", []),
                    time=data_point.get("time", ""),
                    profile=data_point.get("profile", {}),
                    test_case=data_point,
                    id=data_point.get("id", ""),
                )
                
                LOG.info(f"Agent simulation completed for {data_point.get('id', '')}")
                return result
                
            except Exception as e:
                LOG.error(f"Agent simulation error: {e}")
                import traceback
                LOG.error(traceback.format_exc())
                raise
        
        try:
            # Run synchronous agent simulation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_agent_simulation)
            
            end_time = time.time()
            
            # Format result for NeMo-Skills pipeline
            return_dict = {
                "generation_start_time": start_time,
                "generation_end_time": end_time,
                "generation_time": end_time - start_time,
            }
            
            if isinstance(result, tuple) and len(result) == 2:
                final_states, process_list = result
                # Agent tasks return final class states
                return_dict["result"] = final_states
                return_dict["process"] = process_list
                # Convert to string for generation field (required by pipeline)
                return_dict["generation"] = json.dumps(final_states, ensure_ascii=False)
            else:
                # Fallback
                return_dict["result"] = result
                return_dict["process"] = []
                return_dict["generation"] = str(result) if result else ""
            
            LOG.info(f"Agent result formatted: generation length = {len(return_dict.get('generation', ''))}")
            return return_dict
            
        except Exception as e:
            LOG.error(f"Failed to process agent task: {e}")
            import traceback
            LOG.error(traceback.format_exc())
            # Return empty generation to avoid breaking pipeline
            return {
                "generation": "",
                "error": str(e),
            }

    async def process_single_datapoint(self, data_point, all_data):
        """Process a single ACEBench data point."""
        # Determine language from data point or use config default
        language = data_point.get("language", self.cfg.language)
        
        # Check if this is an agent task
        if 'initial_config' in data_point and 'involved_classes' in data_point:
            return await self.process_agent_datapoint(data_point, language)
        
        messages, tools = format_acebench_messages(data_point, use_text_format=True, language=language)
        
        # Add system message if configured
        if self.cfg.system_message:
            messages = [{"role": "system", "content": self.cfg.system_message}] + messages
        
        # Prepare generation request
        return_dict = {}
        
        if self.cfg.count_prompt_tokens and self.hf_tokenizer:
            try:
                if tools:
                    prompt_str = self.hf_tokenizer.apply_chat_template(
                        messages, tools=tools, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt_str = self.hf_tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                num_input_tokens = get_token_count(self.hf_tokenizer, messages=prompt_str)
                return_dict["num_input_tokens"] = num_input_tokens
            except Exception as e:
                LOG.warning(f"Could not count prompt tokens: {e}")

        # Try multiple attempts if configured
        best_response = None
        for attempt in range(self.cfg.attempts):
            try:
                # Generate response using text-based chat
                if self.cfg.prompt_format == "openai":
                    # Use prompt parameter for OpenAI
                    response = await self.generate_with_semaphore(
                        prompt=messages,
                        include_response=True,
                        **asdict(self.cfg.inference),
                    )
                else:
                    if self.hf_tokenizer:
                        prompt = self.hf_tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            **self.cfg.chat_template_kwargs,
                        )
                        response = await self.generate_with_semaphore(
                            prompt=prompt,
                            endpoint_type=EndpointType.text,
                            **asdict(self.cfg.inference),
                        )
                    else:
                        response = await self.generate_with_semaphore(
                            prompt=messages,
                            include_response=True,
                            **asdict(self.cfg.inference),
                        )
                
                # Parse response
                if isinstance(response, dict):
                    # Check for tool_calls in the response (OpenAI format)
                    if "response" in response and hasattr(response["response"], 'choices'):
                        # OpenAI response object
                        message = response["response"].choices[0].message
                        tool_calls = message.tool_calls if hasattr(message, 'tool_calls') and message.tool_calls else None
                        
                        if tool_calls:
                            # Format as ACEBench expects: [ApiName(key1='value1', ...)]
                            func_calls = []
                            for tc in tool_calls:
                                func_name = tc.function.name
                                try:
                                    args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                                    arg_parts = []
                                    for k, v in args.items():
                                        if isinstance(v, str):
                                            arg_parts.append(f"{k}='{v}'")
                                        else:
                                            arg_parts.append(f"{k}={repr(v)}")
                                    arg_str = ', '.join(arg_parts)
                                    func_calls.append(f"{func_name}({arg_str})")
                                except:
                                    func_calls.append(f"{func_name}()")
                            generation = f"[{', '.join(func_calls)}]"
                        else:
                            # No tool calls, just text
                            generation = message.content if hasattr(message, 'content') else ""
                    else:
                        generation = response.get("generation", response.get("response", ""))
                else:
                    generation = str(response)
                
                if self.cfg.attempts == 1:
                    best_response = generation
                    break
                else:
                    # For best-of-N, we'd need to evaluate quality, but for now just use first valid response
                    if generation and (best_response is None or len(generation) > len(best_response or "")):
                        best_response = generation
                
            except Exception as error:
                if is_context_window_exceeded_error(error):
                    error_str = str(error)
                    LOG.warning(f"ACEBench generation failed due to running out of context: {error_str}")
                    if attempt == self.cfg.attempts - 1:
                        return_dict.update({"generation": "", "result": None})
                        return return_dict
                    continue
                else:
                    if attempt == self.cfg.attempts - 1:
                        raise error
                    LOG.warning(f"Generation attempt {attempt + 1} failed: {error}, retrying...")
                    continue

        # Store the generation
        return_dict["generation"] = best_response or ""
        
        # Extract function calls from generation if in ACEBench format [ApiName(...)]
        # This helps with evaluation later
        function_calls = self._extract_function_calls(best_response or "")
        if function_calls:
            return_dict["function_calls"] = function_calls
        
        return return_dict

    def _extract_function_calls(self, generation: str) -> List[Dict]:
        """Extract function calls from ACEBench format: [ApiName(key1='value1', ...)]"""
        function_calls = []
        # Pattern to match [ApiName(...)]
        pattern = r'\[(\w+)\((.*?)\)\]'
        matches = re.findall(pattern, generation)
        for func_name, params_str in matches:
            func_call = {"name": func_name, "arguments": {}}
            # Try to parse parameters
            if params_str.strip():
                # Simple parameter parsing (can be improved)
                param_pattern = r"(\w+)=['\"]([^'\"]+)['\"]"
                param_matches = re.findall(param_pattern, params_str)
                for param_name, param_value in param_matches:
                    func_call["arguments"][param_name] = param_value
            function_calls.append(func_call)
        return function_calls


GENERATION_TASK_CLASS = ACEBenchGenerationTask


@hydra.main(version_base=None, config_name="base_acebench_generation_config")
def acebench_generation(cfg: ACEBenchGenerationConfig):
    cfg = ACEBenchGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = ACEBenchGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    ACEBenchGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        acebench_generation()
