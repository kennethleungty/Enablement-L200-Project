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

"""Runtime Secret Resolution via Google Cloud Secret Manager.

Architecture Overview:
- Ensures zero hardcoded API keys, credentials, or webhook tokens exist in the codebase.
- Provides `SecretManagerResolver` and `get_secret()`, which fetch secrets on demand
  from **Google Cloud Secret Manager** (`google.cloud.secretmanager.SecretManagerServiceClient`)
  using resource path `projects/{project_id}/secrets/{secret_id}/versions/{version}`,
  with an in-process cache and an environment-variable fallback for offline local
  development and automated tests.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Canonical secret identifiers managed in Google Cloud Secret Manager
WORKDAY_HR_API_SECRET_ID = "noogler-workday-hr-api-token"
SERVICENOW_IT_API_SECRET_ID = "noogler-servicenow-it-api-token"
MODEL_ARMOR_TEMPLATE_SECRET_ID = "noogler-model-armor-template-id"


class SecretManagerResolver:
    """Resolves sensitive credentials and configuration values from Google Cloud
    Secret Manager (`google-cloud-secret-manager`) without hardcoding secrets.
    """

    def __init__(
        self,
        project_id: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from google.cloud import secretmanager  # type: ignore[import-untyped]

        self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    def build_secret_version_path(
        self,
        secret_id: str,
        version: str = "latest",
    ) -> str:
        """Constructs the canonical Google Cloud Secret Manager version resource name."""
        project = self.project_id or "unconfigured-gcp-project"
        return f"projects/{project}/secrets/{secret_id}/versions/{version}"

    def resolve_secret(
        self,
        secret_id: str,
        *,
        env_var_fallback: str | None = None,
        version: str = "latest",
        default: str | None = None,
    ) -> str | None:
        """Fetches a secret payload from Google Cloud Secret Manager.

        Resolution order:
        1. Google Cloud Secret Manager (`access_secret_version`) when `project_id`
           is configured and Secret Manager is reachable (or a mock client is injected).
        2. Environment variable (`env_var_fallback` or normalized `secret_id.upper()`).
        3. Explicit `default` value (never a hardcoded credential).
        """
        if self.project_id or self._client is not None:
            try:
                client = self._get_client()
                name = self.build_secret_version_path(secret_id, version=version)
                response = client.access_secret_version(request={"name": name})
                payload_bytes = response.payload.data
                return (
                    payload_bytes.decode("utf-8").strip()
                    if isinstance(payload_bytes, bytes)
                    else str(payload_bytes).strip()
                )
            except Exception as exc:
                logger.debug(
                    "Secret Manager lookup for '%s' fell back to environment: %s",
                    secret_id,
                    exc,
                )

        env_key = env_var_fallback or secret_id.upper().replace("-", "_")
        if env_val := os.environ.get(env_key):
            return env_val.strip()

        return default


@functools.cache
def get_secret_resolver() -> SecretManagerResolver:
    """Returns a cached process-wide `SecretManagerResolver` instance."""
    return SecretManagerResolver()


def get_secret(
    secret_id: str,
    *,
    env_var_fallback: str | None = None,
    version: str = "latest",
    default: str | None = None,
) -> str | None:
    """Convenience helper to fetch a secret via Google Cloud Secret Manager."""
    return get_secret_resolver().resolve_secret(
        secret_id,
        env_var_fallback=env_var_fallback,
        version=version,
        default=default,
    )


def get_integration_client_config(service_name: str) -> dict[str, Any]:
    """Returns non-hardcoded integration metadata for HR/IT/Security clients,
    resolving credentials dynamically via Google Cloud Secret Manager.
    """
    secret_map = {
        "workday_hr": (WORKDAY_HR_API_SECRET_ID, "WORKDAY_HR_API_TOKEN"),
        "servicenow_it": (SERVICENOW_IT_API_SECRET_ID, "SERVICENOW_IT_API_TOKEN"),
        "model_armor": (MODEL_ARMOR_TEMPLATE_SECRET_ID, "MODEL_ARMOR_TEMPLATE_ID"),
    }
    secret_id, env_var = secret_map.get(
        service_name, (f"noogler-{service_name}-secret", service_name.upper())
    )
    resolved_value = get_secret(secret_id, env_var_fallback=env_var)
    return {
        "service_name": service_name,
        "secret_id": secret_id,
        "secret_manager_resource": get_secret_resolver().build_secret_version_path(
            secret_id
        ),
        "is_configured": bool(resolved_value),
    }
