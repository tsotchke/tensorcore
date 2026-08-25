---
name: Offer Apple hardware evidence
about: Volunteer a one-shot physical M4 or M5 TensorCore runtime artifact
title: "hardware evidence: Apple M[4/5] at <commit>"
labels: ""
assignees: ""
---

<!--
Read docs/hardware_evidence_contribution.md before submitting.
Never paste a password, PAT, runner token, serial number, or raw system_profiler output.
The resource and owner ids below are public provenance labels, not credentials.
-->

## Hardware

- Chip class: <!-- M4 or M5 -->
- macOS version:
- Xcode version:
- SDK version (`xcrun --show-sdk-version`):
- Public resource id:
- Public authority owner:

## Source identity

- Requested full commit SHA:
- Evidence `meta.git_head`:
- Evidence SHA-256:

## Handoff

- Artifact attachment or maintainer-approved URL:
- Final `APPLE_FAMILY_EVIDENCE_HANDOFF` line:

## Confirmation

- [ ] I ran `scripts/run_apple_family_evidence_handoff.py` from the exact requested commit.
- [ ] `git status --short --untracked-files=no` was empty before the run.
- [ ] The handoff completed successfully and the returned JSON has not been edited.
- [ ] I did not include credentials, registration tokens, serial numbers, or raw `system_profiler` output.
- [ ] I understand the artifact must pass the repository checker before it counts as release evidence.
