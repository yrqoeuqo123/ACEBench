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
from transformers import AutoTokenizer

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


def convert_functions_to_tools(functions: List[Dict]) -> List[Dict]:
    """Convert ACEBench function format to OpenAI tools format."""
    tools = []
    for func in functions:
        tool = {
            "type": "function",
            "function": {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            },
        }
        tools.append(tool)
    return tools


def format_acebench_messages(
    data_point: Dict[str, Any],
    hf_tokenizer: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Format ACEBench data point into chat messages.
    
    ACEBench data contains:
    - question: user query/conversation history
    - function: available functions
    - agent_system_prompt: system prompt for the agent (optional, from loader)
    - previous_conversation_history: previous conversation (optional)
    - user_system_prompt: system prompt for user simulation (optional, for agent tasks)
    """
    messages = []
    
    # Add system prompt if available (from processed data)
    if "agent_system_prompt" in data_point and data_point["agent_system_prompt"]:
        messages.append({
            "role": "system",
            "content": data_point["agent_system_prompt"],
        })
    
    # Add conversation history if available
    if "previous_conversation_history" in data_point and data_point["previous_conversation_history"]:
        # Parse conversation history if it's a string, otherwise use as-is
        history = data_point["previous_conversation_history"]
        if isinstance(history, str):
            # Try to parse if it's JSON, otherwise treat as plain text
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
    question = data_point.get("question", "")
    if question:
        # Question might include "user: " prefix
        if question.startswith("user: "):
            question = question[6:].strip()
        messages.append({"role": "user", "content": question})
    
    return messages


class ACEBenchGenerationTask(GenerationTask):
    """Generation task for ACEBench benchmark."""

    def __init__(self, cfg: ACEBenchGenerationConfig):
        super().__init__(cfg)
        # Set up tokenizer for prompt formatting if needed
        if cfg.tokenizer:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        else:
            self.hf_tokenizer = None

    def log_example_prompt(self, data):
        """Log an example prompt for debugging."""
        messages = format_acebench_messages(data, self.hf_tokenizer)
        functions = data.get("function", [])
        tools = convert_functions_to_tools(functions) if functions else None
        
        LOG.info("Example ACEBench prompt:")
        LOG.info(f"Messages: {json.dumps(messages, indent=2)}")
        if tools:
            LOG.info(f"Tools: {json.dumps(tools[:1], indent=2)}...")  # Log first tool only
        return

    def setup_prompt(self):
        return None

    async def process_single_datapoint(self, data_point, all_data):
        """Process a single ACEBench data point."""
        functions = data_point.get("function", [])
        tools = convert_functions_to_tools(functions) if functions else None
        
        # Format messages
        messages = format_acebench_messages(data_point, self.hf_tokenizer)
        
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
                # Generate response
                if tools:
                    # Use function calling format
                    if self.cfg.prompt_format == "openai":
                        # OpenAI format with tools
                        response = await self.generate_with_semaphore(
                            messages=messages,
                            tools=tools,
                            **asdict(self.cfg.inference),
                        )
                    else:
                        # Nemo-Skills format - format prompt with tokenizer
                        if self.hf_tokenizer:
                            prompt = self.hf_tokenizer.apply_chat_template(
                                messages,
                                tools=tools,
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
                            # Fallback to OpenAI format
                            response = await self.generate_with_semaphore(
                                messages=messages,
                                tools=tools,
                                **asdict(self.cfg.inference),
                            )
                else:
                    # No tools, regular chat
                    if self.cfg.prompt_format == "openai":
                        response = await self.generate_with_semaphore(
                            messages=messages,
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
                                messages=messages,
                                **asdict(self.cfg.inference),
                            )
                
                # Parse response
                if isinstance(response, dict):
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
