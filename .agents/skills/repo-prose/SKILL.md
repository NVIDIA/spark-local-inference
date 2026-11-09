---
name: repo-prose
description: Write or review repository prose and technical reports that make sense without conversation or development history. Apply to documentation, code prose, commits, PRs, issue comments, plans, and agent reports.
---

# Write Without Hidden Context

All repository prose must make sense to a technically capable reader who has the
repository but none of the conversation or development history. Apply this skill
to documentation, comments, docstrings, names, errors, commit and PR text, issue
and review comments, plans, progress messages, handoffs, and completion reports.

- Describe the system as it exists. State its purpose, behavior, invariants,
  interfaces, evidence, and limitations directly. Do not narrate the journey.
- Do not use lifecycle labels as identities: `Phase 2`, `pilot`, `next`,
  `current`, `new`, `old`, `latest`, and similar terms are not technical names.
- Do not use false definite references. Phrases such as `the 1M-token capture`,
  `the experiment`, or `this approach` require a locally introduced, unambiguous
  referent. Counts, dates, and versions are attributes, not identities.
- On first reference, give the object's semantic role and, when relevant, its
  durable identifier: artifact name, path, schema, revision, manifest, or hash.
- Explain concepts before identifiers. Do not make internal codenames,
  experiment labels, profile numbers, implementation shorthand, or names such
  as `XOR-Cheb-T12` the vocabulary of the design. Mention literal identifiers
  only after describing what they mean and only when the reader needs them.
- Canonical documentation is a present-state specification, not a changelog.
  Replace stale claims instead of layering history on top. Put chronology,
  rejected attempts, and retrospectives only in explicitly historical documents.
- Distinguish implementation from qualification. Where support or readiness
  matters, state what is implemented and the conditions under which it is
  qualified; identify research-only or unsupported behavior explicitly. An
  implementation or passing test suite does not establish support for untested
  workloads.
- State evidence as conditions, measurement, result, and conclusion. Identify
  the tested artifact and relevant configuration, separate measurements from
  hypotheses, and disclose missing hardware, inputs, and unrun checks.
- Comments explain invariants, intent, and non-obvious constraints, never change
  history. TODOs name the missing condition and the criterion for removal.
- Commits and PRs state the problem, resulting behavior, technical reason,
  compatibility impact, and validation relevant to the change. They do not
  recount attempts or pivots. Scale detail to the change; a small correction
  does not need a full report.
- State the substantive fact, requirement, assumption, or limitation directly.
  Do not add commentary about why a statement is present or how the author
  wants the reader to interpret it.

## Review before publishing

Read the final text as a reader without the development conversation. Check that
the affected component, behavior, and evidence can be understood from the text
and its explicit references. Rewrite unexplained references and claims that
exceed the evidence before committing, posting, or sending the result. Review
revised PR titles and descriptions as well as the text accompanying code changes.

When reporting a prose defect, quote the passage, identify the missing context
or unsupported claim, and describe the information needed to repair it. Supply
a rewrite when the facts are known; do not invent details to make it sound
complete. Judge meaning, not the presence of individual words on a banned list.

Illustrative rewrites:

| Requires hidden context | States the behavior or evidence |
|---|---|
| "The new path fixes the capture issue." | "Mixed speculative export preserves separate hidden-state and sampling buffers, allowing Qwen graph construction to complete." |
| "Phase 2 removes the duplicate buffers." | "Dense Qwen graph variants share scratch storage because their executions are serialized." |
| "All tests passed; ready to ship." | "The GPU regression suite passed on SM120. Tensor-parallel execution has not been tested." |

Use these forms only when the implementation and evidence support the statements.

Final test: if understanding any sentence requires "you had to be there,"
rewrite it.

## Delegation and reports

- Include this skill's resolved absolute path in each delegated assignment and
  require the delegate to read and follow it for progress messages, findings,
  completion reports, and handoffs. Pass the requirement to further delegates.
- Supply the task context and relevant artifacts explicitly. A skill reference
  does not replace the context needed to do the work.
- Lead reports with the result and its practical consequence. Identify the
  affected component or artifact, provide supporting evidence, and distinguish
  applied changes from proposals and executed checks from planned checks.
- Use ordinary readable sentences. Keep spaces around words and numbers; do
  not compress reports into concatenated tokens or unexplained abbreviations.
- Resolve ambiguous references and unsupported conclusions in returned reports
  before incorporating them into documentation, decisions, or published text.
