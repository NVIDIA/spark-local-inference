# Security Policy

## Reporting a vulnerability

Report suspected security vulnerabilities in Flare privately to the
[NVIDIA Product Security Incident Response Team (PSIRT)](https://www.nvidia.com/en-us/security/report-vulnerability/).
Use the vulnerability disclosure program linked on that page, or email
[psirt@nvidia.com](mailto:psirt@nvidia.com). For email reports, encrypt the
message using the NVIDIA public PGP key linked from the reporting page.

Do not disclose security vulnerabilities in public GitHub issues, discussions,
or pull requests.

Include the affected Flare version or commit, relevant configuration, a
description of the vulnerability and its potential impact, and reproduction
steps or proof-of-concept code.

See [NVIDIA's PSIRT policies](https://www.nvidia.com/en-us/security/psirt-policies/)
for information about vulnerability handling and disclosure.

## Model core integrity

Flare core signatures authenticate metadata and stored executable bytes. Executable
resources use one zstd level-3 frame per resource; compression is not encryption and
provides no confidentiality. The accepted publisher namespace must exactly match the
signed namespace. Treat signed cores as executable input, not sandboxed code.
