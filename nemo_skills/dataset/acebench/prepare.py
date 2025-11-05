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

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

# Mapping of ACEBench data files to task types
# Based on the actual data files and their categorization
TASK_TYPE_MAPPING = {
    "inference_memory": [
        "data_normal_multi_turn_user_adjust",
        "data_normal_multi_turn_user_switch",
        "data_agent_multi_step",
        "data_agent_multi_turn",
    ],
    "instruction_retention": [
        "data_normal_preference",
        "data_normal_similar_api",
        "data_special_error_param",
        "data_special_incomplete",
        "data_special_irrelevant",
    ],
    "reliable_version_editing": [
        "data_normal_atom_bool",
        "data_normal_atom_enum",
        "data_normal_atom_list",
        "data_normal_atom_number",
        "data_normal_atom_object_deep",
        "data_normal_atom_object_short",
    ],
    "self_coherence": [
        "data_normal_single_turn_single_function",
        "data_normal_single_turn_parallel_function",
    ],
}

DEFAULT_SETTINGS = """
DATASET_GROUP = "tool"
METRICS_TYPE = "acebench"
GENERATION_ARGS = "++eval_type=acebench"
GENERATION_MODULE = "nemo_skills.inference.eval.acebench"
"""


def find_acebench_source_dir() -> Path:
    """Find the ACEBench source directory in agent_hard_benchmark."""
    # Try to find it relative to this file
    current_dir = Path(__file__).parent.absolute()
    # Navigate up to workspace root
    workspace_root = current_dir
    for _ in range(5):
        workspace_root = workspace_root.parent
        acebench_path = workspace_root / "agent_hard_benchmark" / "ACEBench"
        if acebench_path.exists():
            return acebench_path
    # Fallback: check common locations
    for base in [Path.home(), Path("/")]:
        acebench_path = base / "agent_hard_benchmark" / "ACEBench"
        if acebench_path.exists():
            return acebench_path
    raise FileNotFoundError(
        "Could not find ACEBench source directory. "
        "Expected: agent_hard_benchmark/ACEBench relative to workspace root"
    )


def load_question_file(file_path: Path) -> List[Dict]:
    """Load questions from a JSON file (one JSON object per line)."""
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def process_task_type(
    task_type: str,
    source_dir: Path,
    output_dir: Path,
    seed: Optional[int] = None,
):
    """Process all data files for a given task type and create test.jsonl."""
    task_output_dir = output_dir / task_type
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # Get the list of data files for this task type
    data_files = TASK_TYPE_MAPPING.get(task_type, [])
    if not data_files:
        LOG.warning(f"No data files mapped for task type: {task_type}")
        return

    # Load questions from all data files for this task type
    all_questions = []
    data_en_dir = source_dir / "data_all" / "data_en"
    possible_answers_dir = data_en_dir / "possible_answer"

    for data_file_base in data_files:
        data_file = data_en_dir / f"{data_file_base}.json"
        if not data_file.exists():
            LOG.warning(f"Data file not found: {data_file}")
            continue

        questions = load_question_file(data_file)
        LOG.info(f"Loaded {len(questions)} questions from {data_file.name}")

        # Load ground truth if available
        ground_truth_file = possible_answers_dir / f"{data_file_base}.json"
        ground_truths = {}
        if ground_truth_file.exists():
            with open(ground_truth_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        gt = json.loads(line)
                        if "id" in gt:
                            ground_truths[gt["id"]] = gt.get("ground_truth", [])

        # Add ground truth to questions
        for q in questions:
            q_id = q.get("id", "")
            if q_id in ground_truths:
                q["ground_truth"] = ground_truths[q_id]
            # Preserve original field names from AgentHard
            # The fields we need: id, question, function, time, profile, initial_config, path, involved_classes

        all_questions.extend(questions)

    if not all_questions:
        LOG.warning(f"No questions found for task type: {task_type}")
        return

    # Write to test.jsonl
    output_file = task_output_dir / "test.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    LOG.info(f"Wrote {len(all_questions)} questions to {output_file}")

    # Write __init__.py
    init_file = task_output_dir / "__init__.py"
    with open(init_file, "w", encoding="utf-8") as f:
        f.write(DEFAULT_SETTINGS)


def main(args):
    """Main entry point for ACEBench dataset preparation."""
    source_dir = find_acebench_source_dir()
    LOG.info(f"Using ACEBench source directory: {source_dir}")

    # Output directory is the acebench dataset directory
    output_dir = Path(__file__).parent

    # Process each task type
    task_types = TASK_TYPE_MAPPING.keys()
    for task_type in task_types:
        LOG.info(f"Processing task type: {task_type}")
        process_task_type(task_type, source_dir, output_dir, seed=args.seed)

    LOG.info("ACEBench dataset preparation completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ACEBench dataset")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling (if needed)",
    )
    args = parser.parse_args()

    main(args)
