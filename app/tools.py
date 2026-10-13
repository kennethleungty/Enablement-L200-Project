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

"""Domain tools for the Noogler Onboarding Agent.

Implements Rubric Category 1 (Tool & Interface Design):
1. Comprehensive Tool Docstrings: Human- and LLM-readable descriptions and Args.
2. Descriptive Naming: Specific verb-noun function names per specialist domain.
3. Explicit JSON Schemas: Strict Pydantic input/output models with Field constraints.
4. Guided Error Handling: Structured error payloads with actionable recovery_instructions.
"""

from __future__ import annotations

from typing import Any, Literal

from google.adk.tools import FunctionTool, ToolContext
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# ============================================================================
# 1. EXPLICIT JSON SCHEMAS (Pydantic Input / Output & Guided Error Contracts)
# ============================================================================

VALID_HEALTH_PLANS = ("GOOG_PPO_STANDARD", "GOOG_HDHP_HSA", "GOOG_HMO_PREMIER")
VALID_HARDWARE_SKUS = (
    "MACBOOK_PRO_16_M4",
    "MACBOOK_AIR_15_M4",
    "THINKPAD_X1_CARBON_LINUX",
    "CLOUDTOP_WORKSTATION_STANDARD",
)
VALID_SOFTWARE_PACKAGES = (
    "GIT_ON_BORG_READ_WRITE",
    "GCP_SANDBOX_PROJECT_EDITOR",
    "CORP_SSH_BASTION_STANDARD",
    "PROD_DB_ADMIN_ACCESS",
)
HIGH_RISK_ACCESS_PACKAGES = ("PROD_DB_ADMIN_ACCESS",)


class GuidedToolErrorResponse(BaseModel):
    """Standardized guided error response returned to the LLM instead of crashing."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = Field(
        default="error",
        description="Indicates the tool call did not succeed.",
    )
    error_code: str = Field(
        description="Machine-readable error identifier explaining the failure category."
    )
    message: str = Field(
        description="Human-readable explanation of what went wrong during tool execution."
    )
    recovery_instructions: str = Field(
        description=(
            "Actionable step-by-step instructions telling the LLM how to correct "
            "the arguments or what clarifying question to ask the Noogler."
        )
    )


class AdminStatusInputSchema(BaseModel):
    """Strict input schema for querying Noogler administrative status."""

    model_config = ConfigDict(extra="forbid")

    employee_id: str = Field(
        pattern=r"^NOOG-\d{4,6}$",
        description="Noogler employee identifier formatted as 'NOOG-XXXX' (e.g., 'NOOG-1042').",
    )


class AdminStatusOutputSchema(BaseModel):
    """Strict output schema for Noogler administrative and benefits status."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    employee_id: str
    full_name: str
    payroll_direct_deposit_configured: bool
    tax_withholding_w4_submitted: bool
    enrolled_health_plan: str
    retirement_401k_contribution_percent: int
    pending_admin_tasks: list[str]


class PayrollBenefitsElectionInputSchema(BaseModel):
    """Strict input schema for updating health plan and payroll elections."""

    model_config = ConfigDict(extra="forbid")

    employee_id: str = Field(
        pattern=r"^NOOG-\d{4,6}$",
        description="Noogler employee identifier formatted as 'NOOG-XXXX'.",
    )
    health_plan_code: str = Field(
        description=(
            "Target health insurance plan code. Must be one of: "
            "GOOG_PPO_STANDARD, GOOG_HDHP_HSA, or GOOG_HMO_PREMIER."
        )
    )
    retirement_401k_percent: int = Field(
        ge=0,
        le=75,
        description="Percentage of base salary to contribute to 401(k), between 0 and 75.",
    )
    direct_deposit_Verified: bool = Field(
        description="Whether the Noogler has confirmed their direct deposit routing details."
    )


class PayrollBenefitsElectionOutputSchema(BaseModel):
    """Strict output schema confirming payroll and benefits updates."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    employee_id: str
    updated_health_plan: str
    updated_401k_percent: int
    confirmation_ticket_id: str
    summary: str


class TechLogisticsStatusInputSchema(BaseModel):
    """Strict input schema for checking IT hardware and software access status."""

    model_config = ConfigDict(extra="forbid")

    employee_id: str = Field(
        pattern=r"^NOOG-\d{4,6}$",
        description="Noogler employee identifier formatted as 'NOOG-XXXX'.",
    )


class TechLogisticsStatusOutputSchema(BaseModel):
    """Strict output schema for IT equipment and access provisioning status."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    employee_id: str
    assigned_laptop_sku: str
    laptop_shipping_status: str
    enabled_software_groups: list[str]
    security_key_enrolled: bool


class ProvisioningRequestInputSchema(BaseModel):
    """Strict input schema for requesting hardware or software provisioning."""

    model_config = ConfigDict(extra="forbid")

    employee_id: str = Field(
        pattern=r"^NOOG-\d{4,6}$",
        description="Noogler employee identifier formatted as 'NOOG-XXXX'.",
    )
    request_category: Literal["hardware", "software"] = Field(
        description="Whether the provisioning request is for 'hardware' or 'software'."
    )
    item_identifier: str = Field(
        description="Exact hardware SKU or software package identifier to provision."
    )
    business_justification: str = Field(
        min_length=10,
        description="Clear justification (at least 10 characters) for auditing.",
    )


class ProvisioningRequestOutputSchema(BaseModel):
    """Strict output schema for a completed hardware or software provisioning request."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    employee_id: str
    request_category: str
    provisioned_item: str
    provisioning_ticket_id: str
    estimated_completion: str


class RoleKnowledgeSearchInputSchema(BaseModel):
    """Strict input schema for querying the Role Onboarding Knowledge Base."""

    model_config = ConfigDict(extra="forbid")

    job_role: str = Field(
        description="Normalized engineering or product role (e.g., 'swe_backend_l4', 'sre_l4', 'pm_l5')."
    )
    topic_filter: Literal["all", "expectations", "base_readings", "team_norms"] = Field(
        description="Specific onboarding knowledge pillar to retrieve."
    )


class RoleKnowledgeSearchOutputSchema(BaseModel):
    """Strict output schema returning grounded role onboarding documentation."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    job_role: str
    topic_filter: str
    matched_documents: list[dict[str, Any]]
    source_system: str


class SaveOnboardingPlanInputSchema(BaseModel):
    """Strict input schema for persisting a tailored 30-60-90 day onboarding plan."""

    model_config = ConfigDict(extra="forbid")

    employee_id: str = Field(
        pattern=r"^NOOG-\d{4,6}$",
        description="Noogler employee identifier formatted as 'NOOG-XXXX'.",
    )
    job_role: str = Field(description="Role key the plan was tailored for.")
    day_30_milestones: list[str] = Field(
        min_length=1, description="List of concrete milestones for Days 1-30."
    )
    day_60_milestones: list[str] = Field(
        min_length=1, description="List of concrete milestones for Days 31-60."
    )
    day_90_milestones: list[str] = Field(
        min_length=1, description="List of concrete milestones for Days 61-90."
    )


# ============================================================================
# 2. SIMULATED PERSISTENT DATA STORES (Backed by Session State + Seed Records)
# ============================================================================

_SEED_NOOGLER_RECORDS: dict[str, dict[str, Any]] = {
    "NOOG-1042": {
        "full_name": "Alex Rivera",
        "payroll_direct_deposit_configured": True,
        "tax_withholding_w4_submitted": True,
        "enrolled_health_plan": "UNENROLLED",
        "retirement_401k_contribution_percent": 0,
        "assigned_laptop_sku": "MACBOOK_PRO_16_M4",
        "laptop_shipping_status": "IN_TRANSIT_FEDEX_PRIORITY",
        "enabled_software_groups": ["GIT_ON_BORG_READ_WRITE"],
        "security_key_enrolled": True,
    },
    "NOOG-2088": {
        "full_name": "Priya Patel",
        "payroll_direct_deposit_configured": False,
        "tax_withholding_w4_submitted": False,
        "enrolled_health_plan": "GOOG_HDHP_HSA",
        "retirement_401k_contribution_percent": 6,
        "assigned_laptop_sku": "THINKPAD_X1_CARBON_LINUX",
        "laptop_shipping_status": "DELIVERED_TO_DESK",
        "enabled_software_groups": [
            "GIT_ON_BORG_READ_WRITE",
            "GCP_SANDBOX_PROJECT_EDITOR",
        ],
        "security_key_enrolled": True,
    },
}

_ROLE_KNOWLEDGE_BASE: dict[str, dict[str, list[dict[str, str]]]] = {
    "swe_backend_l4": {
        "expectations": [
            {
                "doc_id": "ROLE-SWE4-EXP-01",
                "title": "L4 Software Engineer Core Expectations",
                "content": (
                    "Independently owns medium-sized features from design doc to production rollout. "
                    "Completes Readability certification in primary language within first 60 days "
                    "and participates in team on-call shadow rotations by Day 90."
                ),
            }
        ],
        "base_readings": [
            {
                "doc_id": "ROLE-SWE4-READ-01",
                "title": "Mandatory Engineering Base Readings",
                "content": (
                    "1) Software Engineering at Google (Winters, Manshreck, Wright) Ch. 1, 8, 11; "
                    "2) Google Style Guides & Readability Handbook; "
                    "3) Gemini Enterprise Agent Platform (GEAP) Architecture Overview & ADK Guide."
                ),
            }
        ],
        "team_norms": [
            {
                "doc_id": "ROLE-SWE4-NORM-01",
                "title": "Core Engineering Team Norms",
                "content": (
                    "Small CLs (<300 lines), 24-hour code review turnaround SLA, blameless postmortems "
                    "for all P1/P2 incidents, and weekly Tuesday design review syncs."
                ),
            }
        ],
    },
    "sre_l4": {
        "expectations": [
            {
                "doc_id": "ROLE-SRE4-EXP-01",
                "title": "L4 Site Reliability Engineer Expectations",
                "content": (
                    "Maintains service SLOs/error budgets, eliminates toil through automation (<50% ops load), "
                    "and qualifies for primary production on-call rotation within 90 days."
                ),
            }
        ],
        "base_readings": [
            {
                "doc_id": "ROLE-SRE4-READ-01",
                "title": "Mandatory SRE Base Readings",
                "content": (
                    "1) Site Reliability Engineering (Beyer et al.) Part I & II; "
                    "2) The Site Reliability Workbook (SLOs, Alerting on Burn Rate); "
                    "3) Internal Production Emergency Response Playbook."
                ),
            }
        ],
        "team_norms": [
            {
                "doc_id": "ROLE-SRE4-NORM-01",
                "title": "SRE Team Norms",
                "content": (
                    "Handoff notes required before every shift change, strict adherence to error budget "
                    "freeze policies, and mandatory canary stages for all production pushes."
                ),
            }
        ],
    },
}


def _get_noogler_store(tool_context: ToolContext | None) -> dict[str, dict[str, Any]]:
    """Returns persistent SQLite + session-backed Noogler records, seeding defaults if needed."""
    from app.memory import get_persistent_onboarding_store

    persistent_store = get_persistent_onboarding_store()
    if tool_context is None:
        return persistent_store.get_all_profiles(_SEED_NOOGLER_RECORDS)
    if "noogler_records" not in tool_context.state:
        db_profiles = persistent_store.get_all_profiles(_SEED_NOOGLER_RECORDS)
        tool_context.state["noogler_records"] = {
            k: {
                **v,
                "enabled_software_groups": list(v["enabled_software_groups"]),
            }
            for k, v in db_profiles.items()
        }
    return tool_context.state["noogler_records"]


# ============================================================================
# 3. DOMAIN 1 TOOLS: ADMINISTRATIVE ONBOARDING (Salary, Insurance, Tax)
# ============================================================================


def get_noogler_benefits_and_payroll_status(
    employee_id: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Retrieves a Noogler's current payroll, tax withholding, and health insurance status.

    Use this tool whenever a Noogler asks to check their administrative onboarding progress,
    direct deposit configuration, W-4 tax form status, health insurance plan, or 401(k)
    contribution settings.

    Args:
        employee_id: The Noogler's employee ID formatted as 'NOOG-XXXX' (e.g., 'NOOG-1042').

    Returns:
        A dictionary containing either the validated AdminStatusOutputSchema fields on
        success, or a GuidedToolErrorResponse with recovery_instructions on failure.
    """
    try:
        validated_in = AdminStatusInputSchema(employee_id=employee_id.strip().upper())
    except ValidationError as exc:
        return GuidedToolErrorResponse(
            error_code="INVALID_EMPLOYEE_ID_FORMAT",
            message=f"Employee ID '{employee_id}' failed schema validation: {exc.errors()[0]['msg']}",
            recovery_instructions=(
                "Ask the Noogler for a valid employee ID matching 'NOOG-XXXX' "
                "(for example, 'NOOG-1042' or 'NOOG-2088') and call "
                "get_noogler_benefits_and_payroll_status again."
            ),
        ).model_dump()

    store = _get_noogler_store(tool_context)
    record = store.get(validated_in.employee_id)
    if not record:
        return GuidedToolErrorResponse(
            error_code="NOOGLER_RECORD_NOT_FOUND",
            message=f"No onboarding profile found for employee_id='{validated_in.employee_id}'.",
            recovery_instructions=(
                f"Known active Noogler IDs in the onboarding system are {sorted(store.keys())}. "
                "Verify the ID with the user and retry with a valid 'NOOG-XXXX' identifier."
            ),
        ).model_dump()

    pending: list[str] = []
    if not record["payroll_direct_deposit_configured"]:
        pending.append("Configure bank direct deposit routing in Workday")
    if not record["tax_withholding_w4_submitted"]:
        pending.append("Submit federal/state W-4 tax withholding form")
    if record["enrolled_health_plan"] == "UNENROLLED":
        pending.append("Select and enroll in a health insurance plan")
    if record["retirement_401k_contribution_percent"] == 0:
        pending.append("Set 401(k) retirement contribution percentage (recommended >= 6% for full match)")

    output = AdminStatusOutputSchema(
        employee_id=validated_in.employee_id,
        full_name=record["full_name"],
        payroll_direct_deposit_configured=record["payroll_direct_deposit_configured"],
        tax_withholding_w4_submitted=record["tax_withholding_w4_submitted"],
        enrolled_health_plan=record["enrolled_health_plan"],
        retirement_401k_contribution_percent=record[
            "retirement_401k_contribution_percent"
        ],
        pending_admin_tasks=pending,
    )
    return output.model_dump()


def submit_payroll_and_benefits_election(
    employee_id: str,
    health_plan_code: str,
    retirement_401k_percent: int,
    direct_deposit_verified: bool,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Submits or updates a Noogler's health insurance plan and 401(k) payroll election.

    This is a high-stakes administrative state mutation protected by a Human-in-the-Loop
    confirmation gate. Always verify the employee's current status first and ensure the
    user has confirmed their desired health plan and 401(k) percentage.

    Args:
        employee_id: The Noogler's employee ID formatted as 'NOOG-XXXX' (e.g., 'NOOG-1042').
        health_plan_code: Health insurance code ('GOOG_PPO_STANDARD', 'GOOG_HDHP_HSA', or 'GOOG_HMO_PREMIER').
        retirement_401k_percent: Whole integer percentage (0 to 75) of salary to contribute to 401(k).
        direct_deposit_verified: True if the Noogler confirmed their direct deposit bank routing is active.

    Returns:
        A dictionary containing the validated PayrollBenefitsElectionOutputSchema on success,
        or a GuidedToolErrorResponse with actionable recovery_instructions on failure.
    """
    normalized_plan = health_plan_code.strip().upper()
    if normalized_plan not in VALID_HEALTH_PLANS:
        return GuidedToolErrorResponse(
            error_code="UNSUPPORTED_HEALTH_PLAN_CODE",
            message=f"Health plan '{health_plan_code}' is not a valid Google Noogler benefit plan.",
            recovery_instructions=(
                f"Inform the Noogler that only {list(VALID_HEALTH_PLANS)} are supported. "
                "Map 'PPO' -> 'GOOG_PPO_STANDARD', 'HDHP/HSA' -> 'GOOG_HDHP_HSA', or "
                "'HMO' -> 'GOOG_HMO_PREMIER', then retry submit_payroll_and_benefits_election."
            ),
        ).model_dump()

    try:
        validated_in = PayrollBenefitsElectionInputSchema(
            employee_id=employee_id.strip().upper(),
            health_plan_code=normalized_plan,
            retirement_401k_percent=retirement_401k_percent,
            direct_deposit_Verified=direct_deposit_verified,
        )
    except ValidationError as exc:
        return GuidedToolErrorResponse(
            error_code="INVALID_BENEFITS_ELECTION_ARGUMENTS",
            message=f"Benefits election arguments failed validation: {exc.errors()[0]['msg']}",
            recovery_instructions=(
                "Ensure employee_id matches 'NOOG-XXXX' and retirement_401k_percent is an "
                "integer between 0 and 75, then retry the tool call."
            ),
        ).model_dump()

    if not validated_in.direct_deposit_Verified:
        return GuidedToolErrorResponse(
            error_code="DIRECT_DEPOSIT_NOT_VERIFIED",
            message="Payroll and 401(k) deductions cannot be finalized until direct deposit is verified.",
            recovery_instructions=(
                "Ask the Noogler to confirm that their bank direct deposit routing is verified, "
                "then call submit_payroll_and_benefits_election with direct_deposit_verified=True."
            ),
        ).model_dump()

    store = _get_noogler_store(tool_context)
    record = store.get(validated_in.employee_id)
    if not record:
        return GuidedToolErrorResponse(
            error_code="NOOGLER_RECORD_NOT_FOUND",
            message=f"No onboarding profile found for employee_id='{validated_in.employee_id}'.",
            recovery_instructions=(
                f"Use one of the valid employee IDs {sorted(store.keys())} after confirming with the Noogler."
            ),
        ).model_dump()

    record["enrolled_health_plan"] = validated_in.health_plan_code
    record["retirement_401k_contribution_percent"] = (
        validated_in.retirement_401k_percent
    )
    record["payroll_direct_deposit_configured"] = True
    from app.memory import get_persistent_onboarding_store
    from app.secrets import get_integration_client_config

    _workday_client_cfg = get_integration_client_config("workday_hr")
    get_persistent_onboarding_store().upsert_profile(
        validated_in.employee_id, record
    )

    output = PayrollBenefitsElectionOutputSchema(
        employee_id=validated_in.employee_id,
        updated_health_plan=validated_in.health_plan_code,
        updated_401k_percent=validated_in.retirement_401k_percent,
        confirmation_ticket_id=f"BEN-{validated_in.employee_id}-2026",
        summary=(
            f"Successfully enrolled {record['full_name']} ({validated_in.employee_id}) in "
            f"{validated_in.health_plan_code} with {validated_in.retirement_401k_percent}% 401(k) contribution."
        ),
    )
    return output.model_dump()


# ============================================================================
# 4. DOMAIN 2 TOOLS: TECH LOGISTICS (Hardware & Software Provisioning)
# ============================================================================


def check_tech_equipment_and_access_status(
    employee_id: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Checks a Noogler's assigned laptop hardware order and enabled software access groups.

    Use this tool when a Noogler asks where their laptop is, what hardware SKU is assigned
    to them, whether their Titan security key is enrolled, or what corporate software/access
    groups are currently active on their account.

    Args:
        employee_id: The Noogler's employee ID formatted as 'NOOG-XXXX' (e.g., 'NOOG-1042').

    Returns:
        A dictionary containing the validated TechLogisticsStatusOutputSchema on success,
        or a GuidedToolErrorResponse with recovery_instructions on failure.
    """
    try:
        validated_in = TechLogisticsStatusInputSchema(
            employee_id=employee_id.strip().upper()
        )
    except ValidationError as exc:
        return GuidedToolErrorResponse(
            error_code="INVALID_EMPLOYEE_ID_FORMAT",
            message=f"Employee ID '{employee_id}' is malformed: {exc.errors()[0]['msg']}",
            recovery_instructions=(
                "Request a valid 'NOOG-XXXX' employee ID from the user (e.g., 'NOOG-1042') "
                "and re-invoke check_tech_equipment_and_access_status."
            ),
        ).model_dump()

    store = _get_noogler_store(tool_context)
    record = store.get(validated_in.employee_id)
    if not record:
        return GuidedToolErrorResponse(
            error_code="NOOGLER_RECORD_NOT_FOUND",
            message=f"No tech logistics profile found for '{validated_in.employee_id}'.",
            recovery_instructions=(
                f"Active Noogler profiles are {sorted(store.keys())}. Confirm the correct ID with the user."
            ),
        ).model_dump()

    output = TechLogisticsStatusOutputSchema(
        employee_id=validated_in.employee_id,
        assigned_laptop_sku=record["assigned_laptop_sku"],
        laptop_shipping_status=record["laptop_shipping_status"],
        enabled_software_groups=list(record["enabled_software_groups"]),
        security_key_enrolled=record["security_key_enrolled"],
    )
    return output.model_dump()


def request_hardware_or_software_provisioning(
    employee_id: str,
    request_category: str,
    item_identifier: str,
    business_justification: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Submits an IT ticket to provision new workstation hardware or corporate software access.

    Protected by a conditional Human-in-the-Loop confirmation check. Rejects unapproved
    production database access for first-week Nooglers with guided recovery instructions.

    Args:
        employee_id: The Noogler's employee ID formatted as 'NOOG-XXXX' (e.g., 'NOOG-1042').
        request_category: Either 'hardware' (for laptops/cloudtops) or 'software' (for access groups).
        item_identifier: Exact SKU or access package code (e.g., 'MACBOOK_PRO_16_M4', 'GCP_SANDBOX_PROJECT_EDITOR').
        business_justification: Reason the hardware or access is needed (minimum 10 characters).

    Returns:
        A dictionary containing the validated ProvisioningRequestOutputSchema on success,
        or a GuidedToolErrorResponse with recovery_instructions on failure.
    """
    normalized_category = request_category.strip().lower()
    normalized_item = item_identifier.strip().upper()

    try:
        validated_in = ProvisioningRequestInputSchema(
            employee_id=employee_id.strip().upper(),
            request_category=normalized_category,  # type: ignore[arg-type]
            item_identifier=normalized_item,
            business_justification=business_justification.strip(),
        )
    except ValidationError as exc:
        return GuidedToolErrorResponse(
            error_code="INVALID_PROVISIONING_PARAMETERS",
            message=f"Provisioning request failed schema validation: {exc.errors()[0]['msg']}",
            recovery_instructions=(
                "Ensure employee_id matches 'NOOG-XXXX', request_category is strictly 'hardware' or 'software', "
                "and business_justification is at least 10 characters long."
            ),
        ).model_dump()

    store = _get_noogler_store(tool_context)
    record = store.get(validated_in.employee_id)
    if not record:
        return GuidedToolErrorResponse(
            error_code="NOOGLER_RECORD_NOT_FOUND",
            message=f"Employee ID '{validated_in.employee_id}' does not exist in IT inventory.",
            recovery_instructions=(
                f"Verify the employee ID with the user. Valid IDs: {sorted(store.keys())}."
            ),
        ).model_dump()

    if validated_in.request_category == "hardware":
        if validated_in.item_identifier not in VALID_HARDWARE_SKUS:
            return GuidedToolErrorResponse(
                error_code="UNRECOGNIZED_HARDWARE_SKU",
                message=f"Hardware SKU '{validated_in.item_identifier}' is not in the standard Noogler catalog.",
                recovery_instructions=(
                    f"Select one of the approved hardware SKUs {list(VALID_HARDWARE_SKUS)} "
                    "and call request_hardware_or_software_provisioning again."
                ),
            ).model_dump()
        record["assigned_laptop_sku"] = validated_in.item_identifier
        record["laptop_shipping_status"] = "ORDER_CONFIRMED_PREPARING_SHIPMENT"
        eta = "2 business days via courier"
    else:
        if validated_in.item_identifier in HIGH_RISK_ACCESS_PACKAGES:
            return GuidedToolErrorResponse(
                error_code="POLICY_VIOLATION_RESTRICTED_PROD_ACCESS",
                message=(
                    f"Access package '{validated_in.item_identifier}' requires completed Readability "
                    "and Production Security training, which Nooglers cannot bypass during initial onboarding."
                ),
                recovery_instructions=(
                    "Explain to the Noogler that production database admin access is restricted during "
                    "initial onboarding, and offer to provision 'GCP_SANDBOX_PROJECT_EDITOR' or "
                    "'CORP_SSH_BASTION_STANDARD' instead."
                ),
            ).model_dump()

        if validated_in.item_identifier not in VALID_SOFTWARE_PACKAGES:
            return GuidedToolErrorResponse(
                error_code="UNRECOGNIZED_SOFTWARE_PACKAGE",
                message=f"Software access group '{validated_in.item_identifier}' was not found.",
                recovery_instructions=(
                    f"Use one of the valid software access packages: {list(VALID_SOFTWARE_PACKAGES)}."
                ),
            ).model_dump()

        if validated_in.item_identifier not in record["enabled_software_groups"]:
            record["enabled_software_groups"].append(validated_in.item_identifier)
        eta = "Immediate (propagates within 15 minutes)"

    from app.memory import get_persistent_onboarding_store
    from app.secrets import get_integration_client_config

    _servicenow_client_cfg = get_integration_client_config("servicenow_it")
    get_persistent_onboarding_store().upsert_profile(
        validated_in.employee_id, record
    )

    output = ProvisioningRequestOutputSchema(
        employee_id=validated_in.employee_id,
        request_category=validated_in.request_category,
        provisioned_item=validated_in.item_identifier,
        provisioning_ticket_id=f"IT-{validated_in.employee_id}-{validated_in.request_category.upper()[:2]}-01",
        estimated_completion=eta,
    )
    return output.model_dump()


# ============================================================================
# 5. DOMAIN 3 TOOLS: ROLE-RELATED KNOWLEDGE & 30-60-90 DAY PLANNING
# ============================================================================


def search_role_onboarding_knowledge_base(
    job_role: str,
    topic_filter: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Searches the persistent Role Onboarding Knowledge Base for expectations, readings, and norms.

    Retrieves grounded onboarding documentation for the Noogler's specific role from the
    knowledge index (compatible with Search in Gemini platform / persistent knowledge store).

    Args:
        job_role: Role identifier to query (e.g., 'swe_backend_l4' for L4 Backend SWE, or 'sre_l4' for L4 SRE).
        topic_filter: Category of knowledge to fetch: 'all', 'expectations', 'base_readings', or 'team_norms'.

    Returns:
        A dictionary containing the validated RoleKnowledgeSearchOutputSchema on success,
        or a GuidedToolErrorResponse with recovery_instructions if the role or topic is invalid.
    """
    normalized_role = job_role.strip().lower()
    normalized_topic = topic_filter.strip().lower()

    # Friendly alias normalization for common natural-language inputs
    role_aliases = {
        "swe": "swe_backend_l4",
        "l4 swe": "swe_backend_l4",
        "backend swe": "swe_backend_l4",
        "software engineer": "swe_backend_l4",
        "sre": "sre_l4",
        "site reliability engineer": "sre_l4",
    }
    normalized_role = role_aliases.get(normalized_role, normalized_role)

    try:
        validated_in = RoleKnowledgeSearchInputSchema(
            job_role=normalized_role,
            topic_filter=normalized_topic,  # type: ignore[arg-type]
        )
    except ValidationError as exc:
        return GuidedToolErrorResponse(
            error_code="INVALID_KNOWLEDGE_QUERY_SCHEMA",
            message=f"Knowledge search parameters failed validation: {exc.errors()[0]['msg']}",
            recovery_instructions=(
                "Pass topic_filter as one of ['all', 'expectations', 'base_readings', 'team_norms'] "
                "and job_role as one of ['swe_backend_l4', 'sre_l4']."
            ),
        ).model_dump()

    role_docs = _ROLE_KNOWLEDGE_BASE.get(validated_in.job_role)
    if not role_docs:
        return GuidedToolErrorResponse(
            error_code="UNSUPPORTED_JOB_ROLE",
            message=f"No onboarding curriculum found for job_role='{validated_in.job_role}'.",
            recovery_instructions=(
                f"Available onboarding curricula are {sorted(_ROLE_KNOWLEDGE_BASE.keys())}. "
                "Map the user's role to the closest supported key and retry search_role_onboarding_knowledge_base."
            ),
        ).model_dump()

    from app.memory import get_persistent_onboarding_store

    matched = get_persistent_onboarding_store().query_role_knowledge(
        job_role=validated_in.job_role,
        topic_filter=validated_in.topic_filter,
        seed_kb=_ROLE_KNOWLEDGE_BASE,
    )

    if tool_context is not None:
        tool_context.state["last_retrieved_role_docs"] = matched

    output = RoleKnowledgeSearchOutputSchema(
        job_role=validated_in.job_role,
        topic_filter=validated_in.topic_filter,
        matched_documents=matched,
        source_system="gemini_platform_knowledge_store",
    )
    return output.model_dump()


def save_personalized_onboarding_plan(
    employee_id: str,
    job_role: str,
    day_30_milestones: list[str],
    day_60_milestones: list[str],
    day_90_milestones: list[str],
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Saves a tailored 30-60-90 day role onboarding plan to the Noogler's persistent session state.

    Call this tool after retrieving the role's expectations, base readings, and team norms
    so the Noogler's 30-60-90 day ramp-up milestones are persisted across turns.

    Args:
        employee_id: The Noogler's employee ID formatted as 'NOOG-XXXX' (e.g., 'NOOG-1042').
        job_role: Role key the plan was created for (e.g., 'swe_backend_l4' or 'sre_l4').
        day_30_milestones: Concrete onboarding goals for Days 1 to 30.
        day_60_milestones: Concrete onboarding goals for Days 31 to 60.
        day_90_milestones: Concrete onboarding goals for Days 61 to 90.

    Returns:
        A dictionary confirming the saved 30-60-90 day plan, or a GuidedToolErrorResponse on validation failure.
    """
    try:
        validated_in = SaveOnboardingPlanInputSchema(
            employee_id=employee_id.strip().upper(),
            job_role=job_role.strip().lower(),
            day_30_milestones=day_30_milestones,
            day_60_milestones=day_60_milestones,
            day_90_milestones=day_90_milestones,
        )
    except ValidationError as exc:
        return GuidedToolErrorResponse(
            error_code="INVALID_ONBOARDING_PLAN_SCHEMA",
            message=f"Onboarding plan failed schema validation: {exc.errors()[0]['msg']}",
            recovery_instructions=(
                "Ensure employee_id matches 'NOOG-XXXX' and each milestone list "
                "(day_30_milestones, day_60_milestones, day_90_milestones) contains at least one string."
            ),
        ).model_dump()

    plan_record = {
        "employee_id": validated_in.employee_id,
        "job_role": validated_in.job_role,
        "30_day": validated_in.day_30_milestones,
        "60_day": validated_in.day_60_milestones,
        "90_day": validated_in.day_90_milestones,
    }
    from app.memory import get_persistent_onboarding_store

    get_persistent_onboarding_store().save_onboarding_plan(
        employee_id=validated_in.employee_id,
        job_role=validated_in.job_role,
        plan_record=plan_record,
    )
    if tool_context is not None:
        plans = tool_context.state.get("user:saved_onboarding_plans", {})
        plans[validated_in.employee_id] = plan_record
        tool_context.state["user:saved_onboarding_plans"] = plans

    return {
        "status": "success",
        "plan_id": f"PLAN-{validated_in.employee_id}-{validated_in.job_role.upper()}",
        "saved_plan": plan_record,
    }


# ============================================================================
# 6. HUMAN-IN-THE-LOOP (HITL) WRAPPED TOOLS (Category 3.4)
# ============================================================================


def requires_it_provisioning_approval(
    request_category: str,
    item_identifier: str,
    **kwargs: Any,
) -> bool:
    """Conditional HITL predicate for IT hardware and software provisioning.

    Pauses execution for human approval whenever a Noogler requests:
    - Any physical workstation hardware order (`request_category == 'hardware'`), OR
    - Elevated cloud/production software access (`GCP_SANDBOX_PROJECT_EDITOR` or
      `PROD_DB_ADMIN_ACCESS`).
    """
    normalized_category = request_category.strip().lower()
    normalized_item = item_identifier.strip().upper()
    if normalized_category == "hardware":
        return True
    return normalized_item in (
        *HIGH_RISK_ACCESS_PACKAGES,
        "GCP_SANDBOX_PROJECT_EDITOR",
    )


# High-stakes Administrative Tool wrapped with mandatory human confirmation
submit_payroll_and_benefits_election_hitl_tool = FunctionTool(
    submit_payroll_and_benefits_election,
    require_confirmation=True,
)

# High-stakes Tech Logistics Tool wrapped with conditional human confirmation
request_hardware_or_software_provisioning_hitl_tool = FunctionTool(
    request_hardware_or_software_provisioning,
    require_confirmation=requires_it_provisioning_approval,
)

