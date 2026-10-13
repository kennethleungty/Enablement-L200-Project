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

# Declarative Google Cloud Secret Manager & Security Infrastructure
# Provisions the Secret Manager secret containers referenced by `app/secrets.py`
# and grants least-privilege accessor roles to the agent runtime service identity.

locals {
  onboarding_secret_ids = [
    "noogler-workday-hr-api-token",
    "noogler-servicenow-it-api-token",
    "noogler-model-armor-template-id",
  ]
}

resource "google_secret_manager_secret" "onboarding_secrets" {
  for_each  = toset(local.onboarding_secret_ids)
  project   = var.project_id
  secret_id = each.value

  replication {
    auto {}
  }

  labels = {
    app       = var.project_name
    managed_by = "terraform"
  }

  depends_on = [google_project_service.services]
}

resource "google_secret_manager_secret_iam_member" "agent_secret_accessor" {
  for_each  = google_secret_manager_secret.onboarding_secrets
  project   = var.project_id
  secret_id = each.value.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = google_project_service_identity.vertex_sa.member
}
