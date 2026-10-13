# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model routing configuration for the Noogler Onboarding Agent.

Matches each agent in the hierarchy to a Gemini model tier suited to its task:
- `FLASH_MODEL` (`gemini-3.8-flash`): Used by `noogler_onboarding_agent` (Coordinator),
  `admin_onboarding_agent`, and `tech_logistics_agent` for low-latency intent routing
  and transactional tool execution, as well as background summarization and self-eval.
- `PRO_MODEL` (`gemini-3.8-pro`): Used by `role_knowledge_agent` for deeper
  multi-document synthesis and tailored 30-60-90 day onboarding plan generation.

When the Coordinator delegates a turn via `sub_agents` (`transfer_to_agent`), ADK
automatically executes the turn on the target specialist's configured model tier.
"""

from __future__ import annotations

from google.adk.models import Gemini
from google.genai import types

FLASH_MODEL = "gemini-3.8-flash"
PRO_MODEL = "gemini-3.8-pro"

AGENT_MODEL_TIERS: dict[str, str] = {
    "noogler_onboarding_agent": FLASH_MODEL,
    "admin_onboarding_agent": FLASH_MODEL,
    "tech_logistics_agent": FLASH_MODEL,
    "role_knowledge_agent": PRO_MODEL,
}


def build_gemini_model(model_name: str) -> Gemini:
    """Creates a configured ADK Gemini model instance with automatic HTTP retry."""
    return Gemini(
        model=model_name,
        retry_options=types.HttpRetryOptions(attempts=3),
    )
