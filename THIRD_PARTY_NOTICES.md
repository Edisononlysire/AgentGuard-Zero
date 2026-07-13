# Third-Party Notices

This repository vendors the runtime subset of two modified training components
because AgentGuard-Zero relies on project-specific rollout and worker behavior
that is not represented by an unmodified package release. Unused upstream
examples, tests, code-execution tools, search tools, and SQL tools are omitted.

| Component | Upstream | License | Preserved file |
|---|---|---|---|
| VerL | https://github.com/volcengine/verl | Apache-2.0 | `third_party/VERL_LICENSE` |
| Verl-Tool | https://github.com/TIGER-AI-Lab/verl-tool | MIT | `third_party/VERL_TOOL_LICENSE` |

The VerL notice text is preserved at `third_party/VERL_NOTICE.txt`. All files
under `third_party/` remain governed by their upstream license and copyright
notices. AgentGuard-Zero-specific code outside `third_party/` is governed by
the repository-level Apache-2.0 license.
