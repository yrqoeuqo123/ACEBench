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
        combined_metrics: Dictionary with metrics from all categories and languages:
            - acebench.normal_en, acebench.normal_zh
            - acebench.special_en, acebench.special_zh
            - acebench.agent_en, acebench.agent_zh
    
    Returns:
        Dictionary with overall metrics including per-category, per-language, and averaged scores
    """
    final_metrics = {}
    
    # Category and language combinations
    categories = ["normal", "special", "agent"]
    languages = ["en", "zh"]
    
    # Extract metrics for each category-language combination
    category_metrics = {}
    for category in categories:
        for lang in languages:
            split_name = f"{category}_{lang}"
            # Try multiple key formats
            for key_fmt in [f"acebench.{split_name}", f"acebench/{split_name}", split_name]:
                if key_fmt in combined_metrics:
                    category_metrics[split_name] = combined_metrics[key_fmt]
                    break
    
    # Compute per-split accuracy
    split_accuracies = {}
    for split, metrics in category_metrics.items():
        # Extract accuracy from metrics
        if isinstance(metrics, dict):
            if "accuracy" in metrics:
                split_accuracies[split] = metrics["accuracy"]
            elif "_all_" in metrics and isinstance(metrics["_all_"], dict):
                split_accuracies[split] = metrics["_all_"].get("accuracy", 0.0)
            else:
                # Try to find accuracy in nested structure
                for key, value in metrics.items():
                    if isinstance(value, dict) and "accuracy" in value:
                        split_accuracies[split] = value["accuracy"]
                        break
                else:
                    split_accuracies[split] = 0.0
        else:
            split_accuracies[split] = 0.0

    # Store per-split metrics
    final_metrics["per_split"] = {}
    for category in categories:
        for lang in languages:
            split_name = f"{category}_{lang}"
            final_metrics["per_split"][split_name] = {
                "accuracy": split_accuracies.get(split_name, 0.0),
                "full_metrics": category_metrics.get(split_name, {}),
            }
    
    # Compute per-category average (across languages)
    final_metrics["per_category"] = {}
    for category in categories:
        cat_accs = [split_accuracies.get(f"{category}_{lang}", 0.0) for lang in languages]
        valid_accs = [acc for acc in cat_accs if acc is not None]
        if valid_accs:
            final_metrics["per_category"][category] = {
                "accuracy": sum(valid_accs) / len(valid_accs),
                "en_accuracy": split_accuracies.get(f"{category}_en", 0.0),
                "zh_accuracy": split_accuracies.get(f"{category}_zh", 0.0),
            }
        else:
            final_metrics["per_category"][category] = {
                "accuracy": 0.0,
                "en_accuracy": 0.0,
                "zh_accuracy": 0.0,
            }

    # Compute overall average accuracy (across all splits)
    valid_accuracies = [acc for acc in split_accuracies.values() if acc is not None]
    if valid_accuracies:
        overall_accuracy = sum(valid_accuracies) / len(valid_accuracies)
    else:
        overall_accuracy = 0.0

    final_metrics["overall"] = {
        "accuracy": overall_accuracy,
        "num_splits": len(valid_accuracies),
    }
    
    # Store all raw metrics for reference
    final_metrics["raw_metrics"] = combined_metrics
    
    return final_metrics
