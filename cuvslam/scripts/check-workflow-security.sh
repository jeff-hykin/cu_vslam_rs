#!/bin/bash
set -euo pipefail

# Runner labels that indicate self-hosted runners with sensitive access.
# Update this list when new runner tags are provisioned.
SELF_HOSTED_TAGS="gpu|jetson-orin|jetson-thor"

FORK_GUARD="github.event.pull_request.head.repo.full_name == github.repository"
REFERENCE_URL="https://securitylab.github.com/resources/github-actions-preventing-pwn-requests/"

errors=0

for workflow in .github/workflows/*.yml .github/workflows/*.yaml; do
  [ -f "$workflow" ] || continue

  if ! grep -qE '(^|[[:space:][,])pull_request(_target)?([[:space:]:,]|\]|$)' "$workflow"; then
    continue
  fi

  # --- Check 1: pull_request_target ---
  if grep -q 'pull_request_target' "$workflow"; then
    echo "ERROR: $workflow uses 'pull_request_target' trigger."
    echo ""
    echo "  This trigger runs workflow code from the base branch but grants write"
    echo "  permissions and access to secrets. If the workflow checks out PR head"
    echo "  code, an attacker can execute arbitrary code with elevated privileges"
    echo "  on self-hosted runners."
    echo ""
    echo "  Use 'pull_request' instead, which runs with read-only permissions and"
    echo "  no access to repository secrets."
    echo ""
    echo "  Reference: $REFERENCE_URL"
    echo ""
    errors=$((errors + 1))
  fi

  # --- Check 2: self-hosted runner jobs must have fork protection ---
  # Extract job names that use self-hosted runner tags.
  current_job=""
  while IFS= read -r line; do
    # Detect job definitions (top-level keys under jobs:)
    if echo "$line" | grep -qE '^  [a-zA-Z0-9_-]+:$'; then
      current_job=$(echo "$line" | sed 's/^ *//;s/:$//')
    fi

    # Detect runs-on with self-hosted tags (inline or multi-line YAML list)
    uses_self_hosted=false
    if echo "$line" | grep -qE "runs-on:.*($SELF_HOSTED_TAGS)"; then
      uses_self_hosted=true
    elif echo "$line" | grep -qE '^\s+runs-on:\s*$'; then
      # Multi-line list: scan following indented list items
      found_in_list=false
      while IFS= read -r next_line; do
        if echo "$next_line" | grep -qE '^\s+-\s+'; then
          if echo "$next_line" | grep -qE "($SELF_HOSTED_TAGS)"; then
            found_in_list=true
          fi
        else
          break
        fi
      done
      if [ "$found_in_list" = "true" ]; then
        uses_self_hosted=true
      fi
    fi
    if [ "$uses_self_hosted" = "true" ]; then
      if [ -z "$current_job" ]; then
        continue
      fi

      # Look for the fork guard in the job block (between this job and the next)
      job_start=$(grep -n "^  ${current_job}:" "$workflow" | head -1 | cut -d: -f1)
      if [ -z "$job_start" ]; then
        continue
      fi

      # Find the next job definition or end of file
      job_end=$(tail -n +"$((job_start + 1))" "$workflow" | grep -n '^  [a-zA-Z0-9_-]*:$' | head -1 | cut -d: -f1)
      if [ -n "$job_end" ]; then
        job_end=$((job_start + job_end))
      else
        job_end=$(wc -l < "$workflow")
      fi

      job_block=$(sed -n "${job_start},${job_end}p" "$workflow")

      if ! echo "$job_block" | grep -qF "$FORK_GUARD"; then
        echo "ERROR: $workflow job '$current_job' runs on self-hosted runner without fork protection."
        echo ""
        echo "  Self-hosted runners have access to GPU hardware, mounted storage, and"
        echo "  persistent state. Without fork protection, a PR from a forked repo"
        echo "  can execute untrusted code on these runners."
        echo ""
        echo "  Add this condition to the job:"
        echo "    if: $FORK_GUARD"
        echo ""
        echo "  This restricts the job to PRs created from branches within the repo,"
        echo "  blocking execution from forks."
        echo ""
        errors=$((errors + 1))
      fi
    fi
  done < "$workflow"
done

if [ "$errors" -gt 0 ]; then
  echo "Found $errors CI security issue(s). See errors above."
  exit 1
fi
