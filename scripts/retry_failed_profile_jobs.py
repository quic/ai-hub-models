#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""
Script to retry failed profile jobs that failed due to known flaky reasons.
This script is intended to be run after the exec_scorecard job in the scorecard.yml workflow
and right before the collection workflows.

The script will:
1. Load the profile jobs YAML file from the artifacts directory
2. Find jobs with failure reasons that match known flaky reasons
3. Clone those jobs to retry them
4. Update the YAML file with the new job IDs
5. Save the updated YAML back to the same file location

The updated YAMLs will be saved to the same location as the original YAMLs,
which is typically in the artifacts directory specified by QAIHM_TEST_ARTIFACTS_DIR.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import qai_hub as hub

from qai_hub_models.scorecard.results.yaml import ProfileScorecardJobYaml
from qai_hub_models.utils.hub_clients import get_scorecard_client_or_raise
from qai_hub_models.utils.testing import get_profile_job_ids_file

# List of known flaky failure reasons.
FLAKY_FAILURES = [
    "Failed (Job timed out after 8h)",
    "Failed (Waiting for device timed out after 6h)",
    "Failed (Failed to profile the model: unexpected device error)",
]


def is_flaky_failure(failure_reason: Optional[str]) -> bool:
    """
    Check if a failure reason is in the list of known flaky reasons.

    Args:
        failure_reason: The failure reason string from the job

    Returns:
        True if the failure reason is considered flaky, False otherwise
    """
    if not failure_reason:
        return False

    formatted_reason = f"Failed ({failure_reason})"

    # Check if any known flaky failure reason is in the formatted reason (case insensitive)
    return any(reason.lower() in formatted_reason.lower() for reason in FLAKY_FAILURES)


def get_job_failure_reason(hub_client, job_id: str) -> Optional[str]:
    """
    Get the failure reason for a job if it failed.

    Args:
        hub_client: The Hub client to use for API calls
        job_id: The ID of the job to check

    Returns:
        The failure reason if the job failed, None otherwise
    """
    job = hub_client.get_job(job_id)
    job_status = job.get_status()

    # Only return the failure reason if the job failed
    return job_status.message if job_status.failure else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry failed profile jobs that failed due to known flaky reasons"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=os.environ.get("QAIHM_TEST_ARTIFACTS_DIR", "build/scorecard-yamls"),
        help="Directory containing scorecard artifacts. The updated YAMLs will be saved back to this directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without actually cloning jobs",
    )
    parser.add_argument(
        "--collect-failure-reasons",
        action="store_true",
        help="Collect and print all unique failure reasons without retrying jobs",
    )
    return parser.parse_args()


def collect_failure_reasons(hub_client, profile_yaml) -> dict[str, int]:
    """
    Collect unique failure reasons from the jobs in the profile YAML that match the known flaky failure reasons.

    Args:
        hub_client: The Hub client to use for API calls
        profile_yaml: The ProfileScorecardJobYaml object

    Returns:
        A dictionary mapping failure reasons to their occurrence count
    """
    failure_reasons = {}

    print("Collecting known flaky failure reasons...")

    # Iterate through all entries in the job_id_mapping
    for job_key, job_value in profile_yaml.job_id_mapping.items():
        # Get the job ID (all values are strings in our YAML format)
        job_id = job_value

        # Get the failure reason if any
        failure_reason = get_job_failure_reason(hub_client, job_id)

        # Only collect failure reasons that match the known flaky failure reasons
        if failure_reason and is_flaky_failure(failure_reason):
            formatted_reason = f"Failed ({failure_reason})"
            print(f"Found flaky failure for job {job_id} (key: {job_key})")
            print(f"  Failure reason: {formatted_reason}")

            # Add to the dictionary or increment the count
            if formatted_reason in failure_reasons:
                failure_reasons[formatted_reason] += 1
            else:
                failure_reasons[formatted_reason] = 1

    return failure_reasons


def main():
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    dry_run = args.dry_run
    collect_reasons = args.collect_failure_reasons

    # Get the profile jobs YAML file path
    profile_jobs_file = get_profile_job_ids_file(artifacts_dir)
    if not profile_jobs_file.exists():
        print(f"Profile jobs file not found at {profile_jobs_file}")
        return

    print(f"Loading profile jobs from: {profile_jobs_file}")

    # Load the profile jobs YAML
    profile_yaml = ProfileScorecardJobYaml.from_file(profile_jobs_file)

    # Initialize the Hub client
    # We initialize the client even in dry run mode to fetch job statuses,
    # but we won't use it to clone jobs unless dry_run is False
    hub_client = get_scorecard_client_or_raise()
    print("Successfully initialized Hub client")

    # Collecting failure reasons, do that then exit. (debug to understand and update list of flaky failures)
    if collect_reasons:
        start_time = time.time()
        failure_reasons = collect_failure_reasons(hub_client, profile_yaml)
        elapsed_time = time.time() - start_time

        print(
            f"\n=== Unique Flaky Failure Reasons (collected in {elapsed_time:.2f} seconds) ==="
        )
        for reason, count in sorted(
            failure_reasons.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"Count: {count}, Reason: {reason}")

        return

    # Track if we made any changes
    changes_made = False

    # Find all jobs with flaky failures
    failed_job_keys = {}
    for job_key, job_value in profile_yaml.job_id_mapping.items():
        job_id = job_value

        # Get the failure reason if any
        failure_reason = get_job_failure_reason(hub_client, job_id)

        # Check if it's a flaky failure
        if failure_reason and is_flaky_failure(failure_reason):
            formatted_reason = f"Failed ({failure_reason})"
            print(f"Found flaky failure for job {job_id} (key: {job_key})")
            print(f"  Failure reason: {formatted_reason}")
            failed_job_keys[job_key] = job_id

    # If no flaky failures found, exit early
    if not failed_job_keys:
        print("No flaky failures found, no changes made")
        return

    # In dry run mode, just show what would be done
    if dry_run:
        print(f"[DRY RUN] Found {len(failed_job_keys)} jobs with flaky failures")
        print(f"[DRY RUN] Changes would have been saved to: {profile_jobs_file}")

        # Show a summary of changes that would be made
        print("\n=== Changes that would be made to the YAML file ===")
        for job_key, job_id in failed_job_keys.items():
            print(f"  {job_key}: {job_id} -> [would be replaced with new job ID]")
        return

    # Log what we're about to do
    print(
        f"\nSubmitting {len(failed_job_keys)} new profile jobs for the failed ones listed above"
    )

    # Process each flaky job
    for job_key, job_id in failed_job_keys.items():
        # Clone the job
        print(f"Cloning job {job_id}...")
        prev_job = hub_client.get_job(job_id)
        assert isinstance(prev_job, hub.ProfileJob)
        new_job = hub_client.submit_profile_job(
            prev_job.model, prev_job.device, prev_job.options
        )
        print(f"  Submitted new job: {new_job.job_id}")

        # Update the YAML
        profile_yaml.job_id_mapping[job_key] = new_job.job_id
        changes_made = True

    # Save the updated YAML if changes were made
    if changes_made:
        print(f"Saving updated profile jobs YAML to: {profile_jobs_file}")
        profile_yaml.to_file(profile_jobs_file)
        print("Profile jobs YAML updated with new job IDs")


if __name__ == "__main__":
    main()
