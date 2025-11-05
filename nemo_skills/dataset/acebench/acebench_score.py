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
        combined_metrics: Dictionary with metrics from all categories:
            - acebench.normal
            - acebench.special
            - acebench.agent
    
    Returns:
        Dictionary with overall metrics including per-category and averaged scores
    """
    final_metrics = {}
    
    # Category names
    categories = [
        "normal",
        "special",
        "agent",
    ]
    
    # Extract metrics for each category
    category_metrics = {}
    for category in categories:
        category_key = f"acebench.{category}"
        if category_key in combined_metrics:
            category_metrics[category] = combined_metrics[category_key]
        elif f"acebench/{category}" in combined_metrics:
            category_metrics[category] = combined_metrics[f"acebench/{category}"]
        elif category in combined_metrics:
            category_metrics[category] = combined_metrics[category]
    
    # Compute per-category accuracy
    category_accuracies = {}
    for category, metrics in category_metrics.items():
        # Extract accuracy from metrics
        if isinstance(metrics, dict):
            if "accuracy" in metrics:
                category_accuracies[category] = metrics["accuracy"]
            elif "_all_" in metrics and isinstance(metrics["_all_"], dict):
                category_accuracies[category] = metrics["_all_"].get("accuracy", 0.0)
            else:
                # Try to find accuracy in nested structure
                for key, value in metrics.items():
                    if isinstance(value, dict) and "accuracy" in value:
                        category_accuracies[category] = value["accuracy"]
                        break
                else:
                    category_accuracies[category] = 0.0
        else:
            category_accuracies[category] = 0.0

    # Store per-category metrics
    final_metrics["per_category"] = {}
    for category in categories:
        final_metrics["per_category"][category] = {
            "accuracy": category_accuracies.get(category, 0.0),
            "full_metrics": category_metrics.get(category, {}),
        }

    # Compute overall average accuracy
    valid_accuracies = [acc for acc in category_accuracies.values() if acc is not None]
    if valid_accuracies:
        overall_accuracy = sum(valid_accuracies) / len(valid_accuracies)
    else:
        overall_accuracy = 0.0

    final_metrics["overall"] = {
        "accuracy": overall_accuracy,
        "num_categories": len(valid_accuracies),
    }
    
    # Store all raw metrics for reference
    final_metrics["raw_metrics"] = combined_metrics
    
    return final_metrics
