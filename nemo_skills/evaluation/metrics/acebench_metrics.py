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

from nemo_skills.evaluation.metrics.base import BaseMetrics


class ACEBenchMetrics(BaseMetrics):
    """Metrics for ACEBench evaluation using AST-based checking."""

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Extract scores from prediction."""
        # ACEBench evaluation sets is_correct based on AST checking
        return {"accuracy": prediction.get("is_correct", False)}

    def update(self, predictions):
        """Update metrics with new predictions."""
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)
