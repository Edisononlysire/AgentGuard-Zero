# Third-Party Notices

The focused T12 candidate-ranker release uses externally installed PyTorch,
Transformers, PEFT, and related packages under their respective upstream licenses.
It does not vendor the retired VerL or Verl-Tool training runtimes.

The earlier training components and their original notices remain available in
Git history at the `pre-cleanup-20260907` tag. License material is retained here
for provenance; it is not an instruction to install the old training stack.

| Historical component | Upstream | License material |
|---|---|---|
| VerL | https://github.com/volcengine/verl | `third_party/VERL_LICENSE`, `third_party/VERL_NOTICE.txt` |
| Verl-Tool | https://github.com/TIGER-AI-Lab/verl-tool | `third_party/VERL_TOOL_LICENSE` |

AgentGuard-Zero project code remains under the repository-level Apache-2.0
license. Retained source-level attribution and copyright notices remain intact.
