# ruff: noqa
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

"""Multi-Agent Coordinator and Specialist Sub-Agents for Noogler Onboarding.

Step 3a Implementation:
- Category 3.1 (Multi-Agent Patterns): 1 Root Coordinator + 3 Specialist Sub-Agents
  (`admin_onboarding_agent`, `tech_logistics_agent`, `role_knowledge_agent`).
- Category 3.2 (Strategic Model Routing): Flash (`gemini-3.8-flash`) for Coordinator,
  Administrative, and Tech Logistics agents; Pro (`gemini-3.8-pro`) for Role Knowledge
  & 30-60-90 Day Planning agent.
- Category 2.1 (Robust System Instructions): Clear constitutional instructions defining
  persona, domain boundaries, and strict operational constraints for every agent.
"""

from google.adk.agents import Agent
from google.adk.apps import App, ResumabilityConfig
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin

from app.guardrails import (
    OnboardingPolicyGuardrailPlugin,
)
from app.routing import (
    FLASH_MODEL,
    PRO_MODEL,
    build_gemini_model,
)
from app.tools import (
    check_tech_equipment_and_access_status,
    get_noogler_benefits_and_payroll_status,
    request_hardware_or_software_provisioning_hitl_tool,
    save_personalized_onboarding_plan,
    search_role_onboarding_knowledge_base,
    submit_payroll_and_benefits_election_hitl_tool,
)

MODEL = FLASH_MODEL

# ============================================================================
# SYSTEM CONSTITUTION & SPECIALIST INSTRUCTIONS
# ============================================================================

COORDINATOR_CONSTITUTION = """
You are the **Google Noogler Onboarding Coordinator** (`noogler_onboarding_agent`), the central routing assistant for new Google employees ("Nooglers").

### Constitutional Persona & Core Mandate
1. **Warm, Grounded & Policy-Compliant**: Welcome Nooglers warmly while strictly enforcing Google HR, IT Security, and Engineering onboarding policies.
2. **Never Fabricate Policy or Status**: Never guess payroll status, hardware shipping tracking, or engineering expectations. Always delegate to the appropriate specialist sub-agent.
3. **Protect Sensitive Data (PII)**: Never ask Nooglers for raw Social Security Numbers (SSNs), full bank account numbers, or passwords. Only use their employee ID (`NOOG-XXXX`).

### Multi-Agent Delegation Rules
Route the Noogler's request to the matching specialist sub-agent:
- **`admin_onboarding_agent`**: For salary/payroll direct deposit, W-4 tax withholding, health insurance plans (`GOOG_PPO_STANDARD`, `GOOG_HDHP_HSA`, `GOOG_HMO_PREMIER`), and 401(k) retirement elections.
- **`tech_logistics_agent`**: For workstation hardware (`MACBOOK_PRO_16_M4`, `THINKPAD_X1_CARBON_LINUX`, etc.), shipping status, Titan security keys, and software/corporate access provisioning (`GIT_ON_BORG_READ_WRITE`, `GCP_SANDBOX_PROJECT_EDITOR`, etc.).
- **`role_knowledge_agent`**: For role-specific engineering/SRE expectations, mandatory base readings, team norms, and personalized 30-60-90 day onboarding plans.
"""

ADMIN_AGENT_INSTRUCTION = """
You are the **Administrative Onboarding Specialist** (`admin_onboarding_agent`).
You manage payroll direct deposit checks, W-4 tax status, health insurance enrollment, and 401(k) elections.

### Operational Rules
1. Always call `get_noogler_benefits_and_payroll_status` first if the Noogler's current administrative status is unknown.
2. When calling `submit_payroll_and_benefits_election`, use only valid plan codes (`GOOG_PPO_STANDARD`, `GOOG_HDHP_HSA`, `GOOG_HMO_PREMIER`) and integer 401(k) percentages (0–75).
3. If a tool returns a structured error (`status == "error"`), follow its `recovery_instructions` field to guide the Noogler.
"""

TECH_LOGISTICS_AGENT_INSTRUCTION = """
You are the **Tech Logistics & Access Provisioning Specialist** (`tech_logistics_agent`).
You help Nooglers track workstation hardware orders and provision corporate software access.

### Operational Rules
1. Use `check_tech_equipment_and_access_status` to check assigned laptops, shipping status, and active software access groups.
2. Use `request_hardware_or_software_provisioning` to order approved hardware SKUs or software packages.
3. Never bypass security policies: first-week Nooglers cannot receive `PROD_DB_ADMIN_ACCESS`. If requested, explain the restriction and suggest `GCP_SANDBOX_PROJECT_EDITOR` or `CORP_SSH_BASTION_STANDARD` per the tool's `recovery_instructions`.
"""

ROLE_KNOWLEDGE_AGENT_INSTRUCTION = """
You are the **Role Knowledge & 30-60-90 Day Planning Specialist** (`role_knowledge_agent`), powered by Gemini Pro for deep synthesis.
You guide Nooglers on role-specific expectations, mandatory base readings, engineering team norms, and structured 30-60-90 day ramp-up plans.

### Operational Rules
1. Always ground your answers by calling `search_role_onboarding_knowledge_base` for the Noogler's role (e.g., `swe_backend_l4` or `sre_l4`).
2. When asked for a 30-60-90 day onboarding plan, synthesize concrete milestones grounded in the retrieved docs and persist the plan using `save_personalized_onboarding_plan`.
3. If a tool returns `status == "error"`, follow its `recovery_instructions`.
"""


# ============================================================================
# FACTORY FUNCTIONS FOR SPECIALIST SUB-AGENTS
# ============================================================================


def create_admin_onboarding_agent() -> Agent:
    """Creates Sub-Agent 1: Administrative Onboarding (Flash model, 2 tools)."""
    return Agent(
        name="admin_onboarding_agent",
        model=build_gemini_model(FLASH_MODEL),
        description=(
            "Specialist for administrative onboarding: salary/payroll direct deposit, "
            "W-4 tax withholding, health insurance plans, and 401(k) contribution elections."
        ),
        instruction=ADMIN_AGENT_INSTRUCTION,
        tools=[
            get_noogler_benefits_and_payroll_status,
            submit_payroll_and_benefits_election_hitl_tool,
        ],
    )


def create_tech_logistics_agent() -> Agent:
    """Creates Sub-Agent 2: Tech Logistics & IT Provisioning (Flash model, 2 tools)."""
    return Agent(
        name="tech_logistics_agent",
        model=build_gemini_model(FLASH_MODEL),
        description=(
            "Specialist for tech logistics: checking laptop shipping status, ordering "
            "workstation hardware SKUs, and provisioning software/corporate access groups."
        ),
        instruction=TECH_LOGISTICS_AGENT_INSTRUCTION,
        tools=[
            check_tech_equipment_and_access_status,
            request_hardware_or_software_provisioning_hitl_tool,
        ],
    )


def create_role_knowledge_agent() -> Agent:
    """Creates Sub-Agent 3: Role Knowledge & 30-60-90 Planning (Pro model, 2 tools)."""
    return Agent(
        name="role_knowledge_agent",
        model=build_gemini_model(PRO_MODEL),
        description=(
            "Specialist (Gemini Pro) for role-related knowledge: engineering/SRE expectations, "
            "mandatory base readings, team norms, and tailored 30-60-90 day onboarding plans."
        ),
        instruction=ROLE_KNOWLEDGE_AGENT_INSTRUCTION,
        tools=[
            search_role_onboarding_knowledge_base,
            save_personalized_onboarding_plan,
        ],
    )


# Expose specialist instances for inspection and testing
admin_onboarding_agent = create_admin_onboarding_agent()
tech_logistics_agent = create_tech_logistics_agent()
role_knowledge_agent = create_role_knowledge_agent()

# ============================================================================
# ROOT COORDINATOR AGENT (4 Agents Total: 1 Coordinator + 3 Sub-Agents)
# ============================================================================

root_agent = Agent(
    # Keep in sync with agents-cli-manifest.yaml: agents-cli derives this name
    # from the project `name:` recorded there, and telemetry reports it as
    # gen_ai.agent.name.
    name="noogler_onboarding_agent",
    model=build_gemini_model(FLASH_MODEL),
    description=(
        "Root coordinator agent for Google Noogler onboarding across Administrative, "
        "Tech Logistics, and Role-Related Knowledge domains."
    ),
    instruction=COORDINATOR_CONSTITUTION,
    sub_agents=[
        admin_onboarding_agent,
        tech_logistics_agent,
        role_knowledge_agent,
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
    plugins=[
        OnboardingPolicyGuardrailPlugin(),
        ReflectAndRetryToolPlugin(max_retries=2),
    ],
)
