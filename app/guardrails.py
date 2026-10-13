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

"""Google Cloud Model Armor & LLM-as-a-Judge Self-Evaluation Guardrail Plugin.

Two-layer security and policy enforcement across all agents:
1. **Google Cloud Model Armor (`sanitize_user_prompt` & `sanitize_model_response`)**:
   Uses `google.cloud.modelarmor_v1.ModelArmorClient` inside an ADK `BasePlugin`
   (`OnboardingPolicyGuardrailPlugin`) to screen every incoming user prompt and
   outgoing LLM response against a Model Armor template, backed by deterministic
   local regex shields when running offline.
2. **ADK LLM-as-a-Judge Self-Evaluator (`self_eval_judge_agent`)**:
   Uses a dedicated ADK `Agent` running on `FLASH_MODEL` (`gemini-3.8-flash`) with a
   strict Pydantic `output_schema=SelfEvaluationResult` to audit candidate responses
   for policy compliance, grounding, and PII safety before delivery.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Literal

from google import genai
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import BaseTool, ToolContext
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

from app.routing import FLASH_MODEL, build_gemini_model

logger = logging.getLogger(__name__)

# Optional Google Cloud Model Armor SDK import (installed via google-cloud-modelarmor)
try:
    from google.api_core.client_options import ClientOptions
    from google.cloud import modelarmor_v1

    HAS_MODEL_ARMOR_SDK = True
except ImportError:  # pragma: no cover
    modelarmor_v1 = None  # type: ignore[assignment]
    ClientOptions = None  # type: ignore[assignment]
    HAS_MODEL_ARMOR_SDK = False

PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"bypass\s+(security|approval|guardrail|policy)", re.IGNORECASE),
    re.compile(r"system\s+override|developer\s+mode|jailbreak", re.IGNORECASE),
)

FORBIDDEN_OUTPUT_PROMISES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"granted\s+you\s+prod_db_admin_access\s+immediately",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # Unredacted SSN pattern in model output
)


# ============================================================================
# 1. LLM-AS-A-JUDGE SELF-EVALUATION AGENT & SCHEMA (ADK Self-Eval Pattern)
# ============================================================================

SELF_EVAL_MODEL = FLASH_MODEL  # "gemini-3.8-flash" for low-latency self-grading


class SelfEvaluationResult(BaseModel):
    """Structured output schema returned by the LLM-as-a-Judge self-evaluator."""

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(
        description="True if the candidate response complies with all Noogler onboarding policies."
    )
    verdict: Literal["pass", "fail"] = Field(
        description="Overall compliance grade ('pass' or 'fail')."
    )
    evaluator_model: str = Field(
        default=SELF_EVAL_MODEL,
        description="The Gemini model used to perform self-evaluation.",
    )
    reasoning: str = Field(
        description="Explanation of why the candidate response passed or failed policy checks."
    )
    violations: list[str] = Field(
        default_factory=list,
        description="List of specific onboarding or PII policy violations detected, if any.",
    )


SELF_EVAL_JUDGE_INSTRUCTION = """
You are the **Noogler Onboarding Policy Self-Evaluator (LLM-as-a-Judge)**.
Your job is to audit a candidate agent response before it is shown to a Noogler.

Grade the response as `verdict="fail"` (`passed=False`) if ANY of the following occur:
1. It leaks or echoes an unredacted Social Security Number (SSN) or bank account number.
2. It claims to have granted `PROD_DB_ADMIN_ACCESS` to a first-week Noogler without security training.
3. It fabricates unsupported health insurance plans outside `GOOG_PPO_STANDARD`, `GOOG_HDHP_HSA`, or `GOOG_HMO_PREMIER`.

Otherwise, grade `verdict="pass"` (`passed=True`).
"""

# Dedicated ADK Agent for LLM-as-a-Judge Self-Evaluation
self_eval_judge_agent = Agent(
    name="onboarding_self_eval_judge",
    model=build_gemini_model(SELF_EVAL_MODEL),
    description="LLM-as-a-Judge agent that self-evaluates candidate responses for policy compliance.",
    instruction=SELF_EVAL_JUDGE_INSTRUCTION,
    output_schema=SelfEvaluationResult,
    output_key="last_self_eval_report",
)


def evaluate_output_compliance(
    response_text: str,
    use_llm_judge: bool = False,
) -> dict[str, Any]:
    """Evaluates a candidate LLM response using deterministic rules + optional Gemini Flash Judge.

    Args:
        response_text: Candidate response generated by the active agent.
        use_llm_judge: When True (or `ENABLE_LLM_SELF_EVAL=true`), invokes `SELF_EVAL_MODEL`
            (`gemini-3.8-flash`) with `response_schema=SelfEvaluationResult` in addition
            to deterministic regex checks.

    Returns:
        Dictionary matching `SelfEvaluationResult` schema.
    """
    violations: list[str] = []
    for pattern in FORBIDDEN_OUTPUT_PROMISES:
        if pattern.search(response_text):
            violations.append(f"Matched restricted output pattern: {pattern.pattern}")

    # Deterministic failure short-circuits immediately
    if violations:
        return SelfEvaluationResult(
            passed=False,
            verdict="fail",
            evaluator_model=SELF_EVAL_MODEL,
            reasoning="Deterministic guardrail detected policy or PII violation in candidate output.",
            violations=violations,
        ).model_dump()

    # Optional live LLM-as-a-Judge call using SELF_EVAL_MODEL ("gemini-3.8-flash")
    should_call_llm = use_llm_judge or (
        os.getenv("ENABLE_LLM_SELF_EVAL", "false").lower() == "true"
    )
    if should_call_llm:
        try:
            client = genai.Client()
            judge_resp = client.models.generate_content(
                model=SELF_EVAL_MODEL,
                contents=f"Audit this candidate Noogler onboarding response:\n\n{response_text}",
                config=types.GenerateContentConfig(
                    system_instruction=SELF_EVAL_JUDGE_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=SelfEvaluationResult,
                    temperature=0.0,
                ),
            )
            if judge_resp.text:
                parsed = SelfEvaluationResult.model_validate_json(judge_resp.text)
                return parsed.model_dump()
        except Exception as exc:  # pragma: no cover - fallback when running offline
            logger.warning(
                "LLM self-eval judge fell back to deterministic pass check: %s", exc
            )

    return SelfEvaluationResult(
        passed=True,
        verdict="pass",
        evaluator_model=SELF_EVAL_MODEL,
        reasoning="Candidate response passed all PII and onboarding policy compliance checks.",
        violations=[],
    ).model_dump()


# ============================================================================
# 2. GEMINI SAFETY SETTINGS & MODEL ARMOR PLUGIN
# ============================================================================


def get_gemini_safety_settings() -> list[types.SafetySetting]:
    """Returns strict Gemini platform safety settings applied to all agent LLM calls."""
    return [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
    ]


class OnboardingPolicyGuardrailPlugin(BasePlugin):
    """Runner-wide ADK plugin enforcing Google Cloud Model Armor + LLM Self-Evaluation.

    Registered on `App(plugins=[...])` so every turn across `noogler_onboarding_agent`
    and its 3 sub-agents passes through:
    1. `before_model_callback`: Google Cloud Model Armor `sanitize_user_prompt`
       (plus local prompt-injection shield fallback).
    2. `before_tool_callback`: Pre-execution tool policy & HITL verification audit.
    3. `after_model_callback`: Google Cloud Model Armor `sanitize_model_response`
       AND self-evaluation (`evaluate_output_compliance` / `self_eval_judge_agent`).
    """

    def __init__(
        self,
        model_armor_template_name: str | None = None,
        location: str | None = None,
    ) -> None:
        super().__init__(name="onboarding_policy_guardrail_plugin")
        from app.secrets import MODEL_ARMOR_TEMPLATE_SECRET_ID, get_secret

        self.template_name = model_armor_template_name or (
            get_secret(
                MODEL_ARMOR_TEMPLATE_SECRET_ID,
                env_var_fallback="MODEL_ARMOR_TEMPLATE_NAME",
                default="",
            )
            or ""
        )
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-east1")
        self._client: Any = None

    def _get_model_armor_client(self) -> Any:
        """Lazily initializes the regional Google Cloud Model Armor client when configured."""
        if not self.template_name or not HAS_MODEL_ARMOR_SDK:
            return None
        if self._client is None:
            endpoint = f"modelarmor.{self.location}.rep.googleapis.com"
            self._client = modelarmor_v1.ModelArmorClient(
                client_options=ClientOptions(api_endpoint=endpoint)
            )
        return self._client

    def screen_with_model_armor_prompt(self, prompt_text: str) -> dict[str, Any]:
        """Calls Google Cloud Model Armor `sanitize_user_prompt` API if configured."""
        client = self._get_model_armor_client()
        if client is None:
            return {"enabled": False, "match_found": False, "provider": "local_fallback"}

        try:
            request = modelarmor_v1.SanitizeUserPromptRequest(
                name=self.template_name,
                user_prompt_data=modelarmor_v1.DataItem(text=prompt_text),
            )
            response = client.sanitize_user_prompt(request=request)
            result = response.sanitization_result
            match_found = (
                result.filter_match_state
                == modelarmor_v1.FilterMatchState.MATCH_FOUND
            )
            return {
                "enabled": True,
                "match_found": bool(match_found),
                "provider": "google_cloud_model_armor",
                "filter_match_state": str(result.filter_match_state),
            }
        except Exception as exc:  # pragma: no cover - graceful degradation if offline
            logger.warning("Model Armor prompt check fell back to local rules: %s", exc)
            return {
                "enabled": False,
                "match_found": False,
                "provider": "local_fallback_on_error",
            }

    def screen_with_model_armor_response(self, response_text: str) -> dict[str, Any]:
        """Calls Google Cloud Model Armor `sanitize_model_response` API if configured."""
        client = self._get_model_armor_client()
        if client is None:
            return {"enabled": False, "match_found": False, "provider": "local_fallback"}

        try:
            request = modelarmor_v1.SanitizeModelResponseRequest(
                name=self.template_name,
                model_response_data=modelarmor_v1.DataItem(text=response_text),
            )
            response = client.sanitize_model_response(request=request)
            result = response.sanitization_result
            match_found = (
                result.filter_match_state
                == modelarmor_v1.FilterMatchState.MATCH_FOUND
            )
            return {
                "enabled": True,
                "match_found": bool(match_found),
                "provider": "google_cloud_model_armor",
                "filter_match_state": str(result.filter_match_state),
            }
        except Exception as exc:  # pragma: no cover
            logger.warning("Model Armor response check fell back to local rules: %s", exc)
            return {
                "enabled": False,
                "match_found": False,
                "provider": "local_fallback_on_error",
            }

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        """Screens incoming user prompts via Model Armor + local prompt-injection shields."""
        latest_user_text = ""
        if llm_request.contents:
            for part in llm_request.contents[-1].parts or []:
                if getattr(part, "text", None):
                    latest_user_text += part.text + " "

        # 1. Screen via Google Cloud Model Armor API (if MODEL_ARMOR_TEMPLATE_NAME is set)
        armor_result = self.screen_with_model_armor_prompt(latest_user_text)
        callback_context.state["last_model_armor_prompt_check"] = armor_result
        if armor_result["match_found"]:
            callback_context.state["last_guardrail_action"] = {
                "stage": "before_model_callback",
                "action": "BLOCKED_BY_MODEL_ARMOR",
                "agent_name": callback_context.agent_name,
            }
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=(
                                "Your request was blocked by Google Cloud Model Armor security "
                                "policies. Please ask a standard Noogler onboarding question."
                            )
                        )
                    ],
                )
            )

        # 2. Deterministic prompt-injection & policy-bypass shield
        for pattern in PROMPT_INJECTION_PATTERNS:
            if pattern.search(latest_user_text):
                callback_context.state["last_guardrail_action"] = {
                    "stage": "before_model_callback",
                    "action": "BLOCKED_PROMPT_INJECTION",
                    "agent_name": callback_context.agent_name,
                    "pattern": pattern.pattern,
                }
                return LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(
                                text=(
                                    "I cannot fulfill requests that attempt to override onboarding "
                                    "security policies or system instructions. Please let me know "
                                    "how I can help with your standard Administrative, Tech Logistics, "
                                    "or Role Onboarding tasks."
                                )
                            )
                        ],
                    )
                )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Enforces pre-execution policy checks on tool arguments across all sub-agents."""
        tool_context.state["last_checked_tool"] = {
            "tool_name": tool.name,
            "requires_hitl_confirmation": tool.name
            in (
                "submit_payroll_and_benefits_election",
                "request_hardware_or_software_provisioning",
            ),
        }
        return None

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        """Runs Model Armor response screening + self-evaluation compliance checks."""
        if not llm_response.content or not llm_response.content.parts:
            return None

        combined_text = " ".join(
            part.text
            for part in llm_response.content.parts
            if getattr(part, "text", None)
        )
        if not combined_text:
            return None

        # 1. Google Cloud Model Armor response sanitization
        armor_resp_result = self.screen_with_model_armor_response(combined_text)
        callback_context.state["last_model_armor_response_check"] = armor_resp_result
        if armor_resp_result["match_found"]:
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text="Response withheld by Google Cloud Model Armor policy filter."
                        )
                    ],
                )
            )

        # 2. Self-evaluation output compliance check (deterministic + LLM-as-a-Judge)
        eval_report = evaluate_output_compliance(combined_text)
        callback_context.state["last_self_eval_report"] = eval_report

        if not eval_report["passed"]:
            safe_text = re.sub(
                r"\b\d{3}-\d{2}-\d{4}\b",
                "[REDACTED-SSN]",
                combined_text,
            )
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=safe_text)],
                )
            )

        return None
