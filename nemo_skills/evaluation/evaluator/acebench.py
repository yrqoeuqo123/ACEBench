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
from pathlib import Path
from typing import Any, Dict, List, Optional

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))

try:
    from nemo_skills.dataset.acebench.checker import normal_checker, agent_checker
    from nemo_skills.dataset.acebench.acebench_utils import standardize_string, is_function_call_format_valid
    from nemo_skills.dataset.acebench.acebench_decode import decode_ast
    ACEBENCH_IMPORTS_AVAILABLE = True
except ImportError as e:
    LOG.warning(f"Could not import ACEBench evaluation modules: {e}")
    ACEBENCH_IMPORTS_AVAILABLE = False
    def normal_checker(*args, **kwargs):
        return {"valid": False, "error": ["Import error"]}
    def agent_checker(*args, **kwargs):
        return {"valid": False, "error": ["Import error"]}
    def is_function_call_format_valid(*args, **kwargs):
        return False
    def decode_ast(*args, **kwargs):
        return []
    def standardize_string(*args, **kwargs):
        return ""


def special_checker(model_result_item: str, possible_answer_item: Dict, test_category: str) -> Dict:
    """
    Special checker for incomplete, error_param, and irrelevant categories.
    This evaluates whether the model correctly identifies issues in the user's input.
    """
    result = {
        "valid": True,
        "error": [],
        "error_type": "",
    }
    
    if "incomplete" in test_category:
        for name, values in possible_answer_item.items():
            if "Missing necessary parameters" not in model_result_item:
                result["valid"] = False
                result["error"] = [f"The user's instruction is missing necessary parameters ({values}) for the ({name}), but the model failed to correctly point it out"]
                result["error_type"] = "error_detection"
                break
            elif name not in model_result_item:
                result["valid"] = False
                result["error"] = [f"The user's instruction is missing necessary parameters ({values}) for the ({name}), but the model failed to correctly point it out"]
                result["error_type"] = "error_correction"
                break
            else:
                for value in values:
                    if value not in model_result_item:
                        result["valid"] = False
                        result["error"] = [f"The user's instruction is missing necessary parameters ({value}) for the ({name}), but the model failed to correctly point it out"]
                        result["error_type"] = "error_correction"
                        break
    
    elif "error" in test_category:
        for name, values in possible_answer_item.items():
            if "There is incorrect value" not in model_result_item:
                result["valid"] = False
                result["error"] = [f"The user's instruction contains incorrect values ({values}) of the parameters ({name}), but the model failed to correctly point it out"]
                result["error_type"] = "error_detection"
                break
            else:
                for value in values:
                    if value not in model_result_item:
                        result["valid"] = False
                        result["error"] = [f"The user's instruction contains incorrect values ({values}) of the parameters ({name}), but the model failed to correctly point it out"]
                        result["error_type"] = "error_correction"
                        break
    
    elif "irrelevant" in test_category:
        if "the limitations of the function" not in model_result_item:
            result["valid"] = False
            result["error"] = [f"The model cannot solve this problem, due to the limitations of the function"]
            result["error_type"] = "error_detection"
    
    return result


@nested_dataclass(kw_only=True)
class ACEBenchEvaluatorConfig(BaseEvaluatorConfig):
    model: str = ""
    timeout: int = 300


def extract_qwen_answer(result):
    """For Qwen models, extract the answer after the </think> tag."""
    if isinstance(result, str):
        match = re.search(r'</think>\s*(.*)$', result, re.DOTALL)
        if match:
            return match.group(1).strip()
    return result


def is_qwen_or_ds_r1_model(model_name):
    """Check if model is Qwen or DeepSeek-R1."""
    return "Qwen" in model_name or "qwen" in model_name or "DeepSeek-R1" in model_name


def evaluate_single_sample(
    data_point: Dict[str, Any],
    generation: str,
    model_name: str,
    test_category: str,
) -> Dict[str, Any]:
    """Evaluate a single ACEBench sample using AST parsing and checker."""
    id = data_point.get("id", "")
    question = data_point.get("question", "")
    functions = data_point.get("function", [])
    ground_truth = data_point.get("ground_truth", [])
    
    # For Qwen models, extract the answer after </think>
    if is_qwen_or_ds_r1_model(model_name):
        generation = extract_qwen_answer(generation)
    
    generation_raw = generation
    generation_raw_no_whitespace = "".join(generation_raw.split())
    
    # For special categories, skip AST decoding as they expect error messages, not function calls
    if "special" not in test_category:
        # Decode AST from generation for normal and agent categories
        try:
            decoded_output = decode_ast(model_name, generation_raw_no_whitespace)
        except Exception as e:
            return {
                "valid": False,
                "is_correct": False,
                "error": [f"Invalid syntax. Failed to decode AST. {str(e)}"],
                "error_type": "wrong_output_format",
                "model_result_raw": generation_raw,
                "possible_answer": ground_truth,
            }
        
        # Check if output format is valid
        if not is_function_call_format_valid(decoded_output):
            return {
                "valid": False,
                "is_correct": False,
                "error": ["The output format does not meet the specified requirements."],
                "error_type": "wrong_output_format",
                "model_result": str(generation_raw_no_whitespace),
                "possible_answer": ground_truth,
            }
    else:
        decoded_output = []
    
    # Normalize ground truth
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    
    # Check against all possible answers
    all_errors = []
    is_valid = False
    
    for possible_answer_item in ground_truth:
        if "special" in test_category:
            checker_result = special_checker(
                generation_raw,
                possible_answer_item,
                test_category,
            )
        elif "agent" in test_category:
            checker_result = agent_checker(
                decoded_output[0] if decoded_output else {},
                possible_answer_item,
            )
        else:
            checker_result = normal_checker(
                functions,
                decoded_output,
                possible_answer_item,
                question,
                test_category,
            )
        
        if checker_result.get("valid", False):
            is_valid = True
            break
        else:
            all_errors.append({
                "error": checker_result.get("error", []),
                "error_type": checker_result.get("error_type", "unknown"),
            })
    
    if is_valid:
        return {
            "valid": True,
            "is_correct": True,
            "error": [],
        }
    else:
        return {
            "valid": False,
            "is_correct": False,
            "error": all_errors[0]["error"] if all_errors else ["Unknown error"],
            "error_type": all_errors[0]["error_type"] if all_errors else "unknown",
            "model_result": str(generation_raw_no_whitespace),
            "possible_answer": ground_truth,
        }


def multiplt_turn_accuracy(score_list):
    """Calculate conversation-level and turn-level accuracy for multi-turn tasks.
    
    Matches agent_hard_benchmark/ACEBench/model_eval/evaluation_helper.py:multiplt_turn_accuracy
    """
    end_score_list = []
    process_score_list = []
    
    for score in score_list:
        if False in score["valid"]:
            end_score = 0
        else:
            end_score = 1
        
        process_score = score["valid"].count(True) / len(score["valid"])
        process_score = round(process_score, 3)
        
        end_score_list.append(end_score)
        process_score_list.append(process_score)
    
    end_score_total = round(sum(end_score_list) / len(end_score_list), 3)
    process_score_total = round(sum(process_score_list) / len(process_score_list), 3)
    
    return end_score_total, process_score_total


def eval_acebench(cfg: Dict[str, Any]):
    """Main evaluation function for ACEBench using AST-based checking."""
    eval_config = ACEBenchEvaluatorConfig(**cfg)
    
    # Get model name from config or environment
    model_name = eval_config.model or os.environ.get("NEMO_MODEL", "")
    if not model_name:
        LOG.warning("model not provided, using default decoding")
        model_name = "default"
    
    # Read input file
    input_file = Path(eval_config.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    test_category = ""
    path_parts = input_file.parts
    if "acebench" in path_parts:
        idx = path_parts.index("acebench")
        if idx + 1 < len(path_parts):
            test_category = path_parts[idx + 1]
    
    # Read and process samples
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                samples.append(sample)
    
    LOG.info(f"Evaluating {len(samples)} samples with AST-based checker")
    
    is_multi_turn = "multi_turn" in test_category.lower()
    
    results = []
    correct_count = 0
    score_list = []
    
    for sample in samples:
        generation = sample.get("generation", sample.get("result", ""))

        # Infer test category from sample ID
        sample_category = None
        sample_id = sample.get("id", "")
        if sample_id:
            sample_id_lower = sample_id.lower()
            if "special" in sample_id_lower:
                sample_category = "special"
            elif "agent" in sample_id_lower:
                sample_category = "agent"
            elif "normal" in sample_id_lower:
                if "multi_turn" in sample_id_lower:
                    sample_category = "normal_multi_turn"
                elif "atom" in sample_id_lower:
                    sample_category = "normal_atom"
                else:
                    sample_category = "normal"
        
        # Fallback to task type category if ID inference failed
        if not sample_category:
            sample_category = test_category
        
        eval_result = evaluate_single_sample(
            sample,
            generation,
            model_name,
            sample_category or "normal",
        )
        
        # Update sample with evaluation results
        sample["is_correct"] = eval_result.get("is_correct", False)
        sample["valid"] = eval_result.get("valid", False)
        if "error" in eval_result:
            sample["error"] = eval_result["error"]
        if "error_type" in eval_result:
            sample["error_type"] = eval_result["error_type"]
        
        results.append(sample)
        if eval_result.get("is_correct", False):
            correct_count += 1
        
        if is_multi_turn and sample_id:
            parts = sample_id.split("_")
            if len(parts) >= 2:
                turn = parts[-2]
                item = parts[-1]
                
                if len(score_list) > 0 and turn == score_list[-1]["turn"]:
                    score_list[-1]["valid"].append(eval_result.get("is_correct", False))
                    score_list[-1]["number"] = item
                else:
                    score_list.append({
                        "turn": turn,
                        "number": item,
                        "valid": [eval_result.get("is_correct", False)]
                    })
    
    total_count = len(samples)
    
    if is_multi_turn and len(score_list) > 0:
        end_accuracy, process_accuracy = multiplt_turn_accuracy(score_list)
        accuracy = end_accuracy
        LOG.info(f"Evaluation complete. End Accuracy (conv-level): {end_accuracy} | Process Accuracy (turn-level): {process_accuracy} ({correct_count}/{total_count} turns)")
    else:
        accuracy = round(correct_count / total_count, 3) if total_count else 0.0
        LOG.info(f"Evaluation complete. Accuracy: {accuracy} ({correct_count}/{total_count} samples)")
    
    with open(input_file, "w", encoding="utf-8") as f:
        for sample in results:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    LOG.info(f"Results written to {input_file}")


if __name__ == "__main__":
    import sys
    
    # Simple CLI for testing
    if len(sys.argv) < 2:
        print("Usage: python acebench.py <input_file> [model_name]")
        sys.exit(1)
    
    config = {
        "input_file": sys.argv[1],
        "model": sys.argv[2] if len(sys.argv) > 2 else os.environ.get("NEMO_MODEL", ""),
        "timeout": 300,
    }
    
    eval_acebench(config)
