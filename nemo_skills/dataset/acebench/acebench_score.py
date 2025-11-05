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

from typing import Any, Dict


def compute_score(combined_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute overall ACEBench score from combined metrics.
    
    Args:
        combined_metrics: Dictionary with metrics from all task types:
            - acebench.inference_memory
            - acebench.instruction_retention
            - acebench.reliable_version_editing
            - acebench.self_coherence
    
    Returns:
        Dictionary with overall metrics including per-task and averaged scores
    """
    final_metrics = {}
    
    # Task type names
    task_types = [
        "inference_memory",
        "instruction_retention",
        "reliable_version_editing",
        "self_coherence",
    ]
    
    # Extract metrics for each task type
    task_metrics = {}
    for task_type in task_types:
        task_key = f"acebench.{task_type}"
        if task_key in combined_metrics:
            task_metrics[task_type] = combined_metrics[task_key]
        elif f"acebench/{task_type}" in combined_metrics:
            task_metrics[task_type] = combined_metrics[f"acebench/{task_type}"]
        elif task_type in combined_metrics:
            task_metrics[task_type] = combined_metrics[task_type]
    
    # Compute per-task accuracy
    task_accuracies = {}
    for task_type, metrics in task_metrics.items():
        # Extract accuracy from metrics
        if isinstance(metrics, dict):
            if "accuracy" in metrics:
                task_accuracies[task_type] = metrics["accuracy"]
            elif "_all_" in metrics and isinstance(metrics["_all_"], dict):
                task_accuracies[task_type] = metrics["_all_"].get("accuracy", 0.0)
            else:
                # Try to find accuracy in nested structure
                for key, value in metrics.items():
                    if isinstance(value, dict) and "accuracy" in value:
                        task_accuracies[task_type] = value["accuracy"]
                        break
                else:
                    task_accuracies[task_type] = 0.0
        else:
            task_accuracies[task_type] = 0.0
    
    # Store per-task metrics
    final_metrics["per_task"] = {}
    for task_type in task_types:
        final_metrics["per_task"][task_type] = {
            "accuracy": task_accuracies.get(task_type, 0.0),
            "full_metrics": task_metrics.get(task_type, {}),
        }
    
    # Compute overall average accuracy
    valid_accuracies = [acc for acc in task_accuracies.values() if acc is not None]
    if valid_accuracies:
        overall_accuracy = sum(valid_accuracies) / len(valid_accuracies)
    else:
        overall_accuracy = 0.0
    
    final_metrics["overall"] = {
        "accuracy": overall_accuracy,
        "num_tasks": len(valid_accuracies),
    }
    
    # Store all raw metrics for reference
    final_metrics["raw_metrics"] = combined_metrics
    
    return final_metrics
