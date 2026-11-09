# Working on Flare

Flare is a local inference server for NVIDIA SM103/SM120/SM121 GPUs. Ignition, the
Python package in `ignition/`, prepares models with PyTorch, Transformers, and
FlashInfer, including b12x. Its `flare_heliopause` extension is built by the
`heliopause/` crate. The Rust CLI in `flare-cli/` uses the application library in
`flare/`, which links the Helios runtime in `helios/` to serve requests through
prepared CUDA graphs. Flare and Heliopause depend on Helios; Helios has no Python
or preparation-tool dependency.

## Required setup and commit review

On entering a clone or worktree, and after pulling changes, re-read this file,
[the design self-review](docs/design-review.md), and
[the repository prose skill](.agents/skills/repo-prose/SKILL.md). Before committing,
enable the repository's `.githooks/commit-msg` hook by following the
[hook installation instructions](docs/design-review.md#hook-installation).
Preserve any existing active hooks when installing.

Before every commit, follow the staged-diff review and acknowledgment instructions
in `docs/design-review.md`. A rejected acknowledgment requires reading that
document and reviewing the staged change before retrying; do not treat the
rejection as a formatting repair.

## Required prose review

Before writing or reviewing repository prose, read and apply
[the repository prose skill](.agents/skills/repo-prose/SKILL.md). It applies to
documentation, code comments, docstrings, names, errors, commit messages, PR titles
and descriptions, issue and review comments, plans, and technical reports. Code
tasks are included whenever they produce prose, including progress messages and
completion reports.

Before committing or publishing, review the final text against the skill. Rewrite
passages that require conversation history, introduce unexplained identifiers, or
claim results beyond the stated evidence. Review edited PR descriptions and
comments before publishing them, even when no repository files change.

Every delegated task must include the skill's resolved absolute path and require
the delegate to read and follow it. Pass this requirement to further delegates,
and review returned prose before incorporating or publishing it. A review
acknowledgment does not substitute for reading the skill and reviewing the text.

## Required PR descriptions

A PR description must explain the argument for the change to a reviewer who knows
the repository but has none of the author's working context. **Every description
must answer these four questions, in this order: why, what exactly changed, what
benefit or result is claimed, and what evidence supports it.** A description that
makes the reviewer reconstruct these answers from implementation details, test
inventories, or linked PRs is incomplete. Rewrite it before publishing, even if
every individual statement is accurate.

1. **Why is this change being proposed?** Identify the affected workflow and the
   concrete defect, cost, or missing capability. Explain why it matters and why
   the chosen approach addresses it. Give the relevant dataflow, ownership, or
   execution constraint that makes the approach valid. Technical vocabulary and
   a sequence of operations do not substitute for a causal explanation. Assume
   an experienced engineer who needs local context, not introductory instruction.
2. **What exactly did this PR change?** Name the affected components and explain
   the concrete implementation and behavior changes. Make the before/after
   explicit: what is replaced, removed, or added, and how the resulting path
   works. Do not describe both behaviors in the same tense and leave the reader
   to infer the transition. Identify material numerical, compatibility, and
   resource-lifetime changes. A file list or a label such as "optimized prefill"
   does not explain what was done.
3. **What is the claimed benefit or result?** State the expected or observed
   effect precisely, distinguishing an expectation from a measured result. Give
   the headline comparison and metric direction for a measured improvement,
   with its scope and material limitations attached. Distinguish component
   measurements from whole-model or service results, and isolated changes from
   combined-stack results. Never attribute a combined result to one PR. The
   claimed benefit must be clear before presenting the supporting test and
   benchmark details.
4. **What evidence supports those claims?** Connect each material claim to the
   relevant checks or measurements: the behavior tested, baseline or reference,
   configuration, result, and limits of the conclusion. State what remains
   untested or inconclusive. Explain differences between runs when passed and
   skipped cases would otherwise appear contradictory. Keep material failures,
   skipped checks, and measurement limitations visible. Test counts and labels
   such as "qualified" cannot stand in for the tested contract and result.
   Reproduction commands must identify the required inputs and configuration
   directly or through a precise reference; placeholder arguments alone do not
   explain a measurement. Never invent evidence to complete the description.

Apply these requirements to the supporting detail:

- **Give details a clear hierarchy.** Establish the central design and principal
  correctness or compatibility risks before supporting implementation and
  validation details. Include details because they help assess the change.
  Avoid exhaustive inventories of changed files, preserved properties, buffers,
  and tests. Explain the relevance of non-obvious constraints. Keep reproduction
  instructions and merge logistics after the technical explanation.
- **Resolve references where they are needed.** Explain a component's role before
  using its internal shorthand. Replace phrases such as "ordinary tuning" or
  "canonical policy" with the relevant settings or a precise reference. Links
  may carry supporting detail, but the description must contain the central
  argument. For stacked PRs, explain what each necessary dependency supplies,
  what this diff contributes, and the required merge order; a chain of PR numbers
  alone is insufficient.
- **Remove process assurances that add no technical evidence.** Statements that
  source "matches the tested implementation," work was reviewed, or a branch is
  ready do not explain behavior or establish correctness. Report the actual
  checks and their scope. Do not add audit artifacts or development chronology.

Before publishing or updating a PR, read its description independently of the
diff and development conversation. **Do not publish it if the rationale, exact
change, claimed benefit, supporting evidence, or material limitations must be
guessed.** Repair missing explanations rather than adding headings or more raw
detail. Scale length to the change: a small fix may need only a short paragraph
and relevant validation. The content and order are mandatory; particular headings
and a fixed-length template are not.

## Contribution scope and rejection criteria

Flare prepares and serves supported model recipes on NVIDIA GPUs through
FlashInfer/b12x and prepared CUDA graphs. Single-host execution and cross-node
tensor parallelism are both in scope. Orbit manages the hosts, preparation,
cores, and rank processes needed for those serving workflows.

Every contribution must address an explicit maintainer request, a demonstrated
defect, a measured performance or memory problem, or inaccurate documentation.
Identify the affected workflow and relevant implementation before editing.
Hypothetical future consumers, agent-generated suggestions, and lint findings
alone do not establish a need for a change.

- **Reject unapproved scope expansion.** CPU or non-NVIDIA execution,
  interchangeable execution backends, training or fine-tuning, multi-model
  hosting and routing, unrelated service APIs, and general-purpose cluster
  orchestration require an explicit maintainer decision before implementation.
  Supported cross-node preparation and serving do not authorize these additions.
- **Reject architectural violations.** Production Python callbacks, eager
  serving, runtime compilation or recapture, steady-state GPU allocation,
  model-specific native policy or state-update algorithms, and executable
  extension mechanisms in prepared metadata violate the boundaries below.
  Flags, enums, callbacks, optional paths, and temporary implementations cannot
  be used to bypass them.
- **Reject weakened guarantees.** Preserve complete cache and speculative state,
  inactive-row behavior, asynchronous resource lifetimes, checkpoint precision,
  model semantics, and the declared model core contract. Do not remove
  validation, suppress errors, silently substitute execution paths, or relax references,
  workloads, or numerical bounds to obtain passing checks or favorable metrics.
  Numerical bounds must follow the required arithmetic and measured repeatability.
- **Reject unsupported abstractions and churn.** Do not add frameworks,
  extension points, speculative compatibility, fallback layers, or configuration
  switches without a concrete requirement. Unsolicited formatting, renaming,
  comment expansion, annotation sweeps, and file rearrangements are not
  contributions by themselves. Refactoring must directly support the requested
  behavior or remove a demonstrated source of defects or cost.
- **Require focused, meaningful validation.** Tests must exercise the affected
  behavior or contract; source-text checks and mocks that bypass the behavior
  cannot establish runtime correctness. Do not present scaffolding or partially
  wired features as supported. Identify missing hardware, inputs, and unrun
  checks explicitly. Keep each PR focused on one coherent outcome and describe
  its problem, resulting behavior, validation, and material limitations.

An explicit maintainer request covering the work is sufficient authorization
for changes within these constraints and their necessary supporting steps.
A feature request, passing tests, or a performance gain does not waive an
architectural invariant. Resolve conflicts through an explicit maintainer
decision on the design before implementing a change to that invariant; a commit
acknowledgment or routine PR approval is not that decision. Apply these criteria
to documentation and test changes that would weaken the rules as well as code.

When reviewing a contribution that violates these criteria, identify the
concrete violation and relevant rule, and recommend rejection. Do not rescue
it by adding an opt-in, weakening the rule, expanding its scope, or promising
future cleanup.

## Choose guidance by task

Read the guidance for every task affected by the work before implementing or
running it. These requirements apply by task, including work outside the named
source directories. For preparation or model validation, read the relevant model
and hardware sections of the preparation guide.

| Task | Required guidance |
|---|---|
| Build, install dependencies, or launch Flare | [Build and run model checkpoints](#build-and-run-model-checkpoints), [environment setup](docs/model-preparation.md#environment-setup), and the relevant [CLI commands](docs/cli-design.md) |
| Prepare, export, or serve an existing model | [Model preparation and serving](docs/model-preparation.md); [direct execution](docs/direct-execution.md) when reproducing faults with `--direct` |
| Add or change a model recipe or execution component | [Model development](docs/model-development.md), [preparation development](docs/preparation-development.md), and the relevant [core design](docs/core-design.md#2-model-preparation) sections |
| Change tuning, compilation, memory planning, capture, or TP preparation | [Preparation development](docs/preparation-development.md) and [preparation resources](docs/model-preparation.md#preparation-resources) |
| Change Helios runtime, scheduling, caches, transport, or frontend | [Prepared model contract](helios/model-contract.md), [execution manifest](helios/manifest.md), and the relevant [Helios runtime](helios/README.md) and [core design](docs/core-design.md#3-helios-runtime-and-graph-interface) sections |
| Change Protobuf metadata or core loading/export | [Runtime core design](docs/runtime-core-design.md), [prepared model contract](helios/model-contract.md), and [core compatibility](#model-core-compatibility) |
| Diagnose or optimize performance | [GPU performance work](docs/performance-work.md), plus model or preparation development guidance for affected code |
| Change CLI, API, or configuration behavior | [CLI design](docs/cli-design.md) and the relevant [API](docs/api.md) sections |

For sampling or frontend metadata changes, also read
[sampling and frontend metadata](docs/model-development.md#sampling-and-frontend-metadata).
Model onboarding and changes to tokenizers, chat templates, or native prompt
encoders must satisfy the [renderer contract and validation requirements](docs/model-development.md#prompt-renderers-and-checkpoint-boundaries).

## Build and run model checkpoints

Run commands from the repository root:

```sh
uv sync --locked --extra prepare --extra test
cargo build --release --manifest-path flare-cli/Cargo.toml
```

After changing `heliopause/` or Helios code used by the extension, rebuild the
installed extension:

```sh
uv pip install --python .venv/bin/python --no-deps ./heliopause
```

The [preparation guide](docs/model-preparation.md) owns dependency updates,
local FlashInfer installation, GPU selection, compiler and tuning resources,
model-specific launch instructions, and hardware setup. The
[CLI design](docs/cli-design.md) defines flags, configuration, signing, and
core selection.

Use the preparation guide's [DeepSeek V4.1](docs/model-preparation.md#deepseek-v41-flash)
or [DeepSeek V4](docs/model-preparation.md#deepseek-v4-flash) commands for those
models. For repeated exports with compatible weights, follow
[re-exporting with cached weights](docs/model-preparation.md#re-exporting-with-cached-weights).

For serving-only Rust changes, rebuild Flare and load a compatible core.
To exercise changes to prepared kernels, capacities, or the core format,
export a core containing those changes. Cores must satisfy the declared
[core contract](#model-core-compatibility). Changing prepared kernels or
capacities requires re-export; an engine upgrade does not.
Launch an exported core with `flare-cli/target/release/flare run CORE` when
testing a local build. Stop Flare with Ctrl-C or SIGTERM so native replay drains
before CUDA resources are released.

`ignition debug-serve` is only for diagnosing model and preparation faults.
Do not use it for actual serving or apples-to-apples performance comparisons
against other runtimes, including its native, `--direct`, and `--eager` modes.
Use `ignition export` followed by `flare run CORE` for those workflows.
Debug execution retains preparation resources and bounded preparation caches;
its process, memory, and loading paths differ from native core serving.

For debugging without export, native `ignition debug-serve` requires Heliopause's
`qualification` Cargo feature. Diagnostic `ignition debug-serve --direct` does not.
Follow the
[development launch instructions](docs/cli-design.md#development-without-core-export).

## Architecture and ownership

Helios's shared runtime policy and separation from model computation are
non-negotiable architectural invariants. Correct output, performance gains,
delivery pressure, and intended cleanup do not justify violating them. A
temporary model-specific policy path is still a violation. Resolve the
architectural boundary before committing the change.

- Keep Helios as simple as possible. Model-specific execution behavior,
  composition, and optimization belong in Ignition and its captured programs.
  Limit native specialization to concrete cache storage requirements and how
  that storage is exposed to captured graphs. Express differences through cache
  schemes and prepared layout metadata, with contracts reusable by every model
  that has those requirements. Model state-update computation belongs in
  Ignition; native code manages its declared storage without implementing
  per-model algorithms.
- Admission, page reservation and growth, eviction, preemption, readmission,
  and offload are shared runtime policy and mechanisms. Do not select them by
  model or recipe identity, duplicate them in model contracts, or hide model
  allowlists behind capability flags. A cache scheme may require different page
  geometry, state storage, or restore constraints; keep those differences in
  its storage contract and let shared policy consume that contract.
- `PreparedContract` is validated Protobuf data, never a runtime policy extension
  point. Declare concrete graph value sources, storage views, dependencies,
  frontend syntax, and capacities; shared code validates and binds them. Never
  add per-model boolean flags, including `supports_*`, `can_*`, or default-false
  opt-ins, to enable or disable runtime behavior. Enum values, callbacks, schema
  names, and metadata fields must not encode the same model allowlist. Every
  member must have a meaning independent of model identity, and its consumers
  must implement shared behavior
  from actual data, geometry, and dependencies. Review both declarations and all
  consumers. Tool/reasoning parsers and image preparation/layout implementations
  use independent typed enum identifiers; those identifiers cannot select
  scheduling, cache policy, text-graph input packing, or completion behavior.
- A runtime defect reproduced on one model must be traced through every
  supported cache scheme before choosing the implementation boundary. Apply
  the fix to every compatible contract. Any restriction must follow from a
  concrete storage or execution constraint expressed in the contract. For
  example, incremental page reservation must depend on the cache's growth and
  continuation requirements; testing it with GLM does not justify a GLM-only
  scheduler path. Validate affected cache schemes and request lifecycles, and
  report untested models separately from implementation restrictions.
- Introducing a captured-graph operation family requires prior explicit approval
  from project owner Luke Alonso <lalonso@gmail.com>. Shape and capacity variants
  of an existing family do not constitute a new type when its operation contract
  stays unchanged. A new operation contract requires approval even if it reuses
  an existing graph kind or enum value. A design acknowledgment does not grant
  that approval.
- Recipes in `ignition/recipes/` own model-specific composition, checkpoint mapping,
  state layouts, and graph preparation. They may retain canonical forwards or
  compose prepared components as the model requires.
- Put reusable execution mechanisms in `ignition/components/`. Recipes should be
  able to use components independently without adopting a rigid model framework.
  Shared/routed expert overlap, projections, collectives, and state transfers
  are examples of mechanisms that should not be copied between recipes.
- Keep native input packing, frontend processing, transport, scheduling, cache
  machinery, and request handling shared. Express model-dependent behavior through
  prepared metadata and frontend assets; keep native specialization focused on
  cache storage contracts and graph cache bindings.

## Operators and hardware

- Flare is fully coupled to FlashInfer/b12x by design. It has no backend or
  device abstractions. Use FlashInfer/b12x and NVIDIA CUDA interfaces directly;
  do not introduce interchangeable kernel backends or device portability layers.
- Prefer proven b12x operators and FlashInfer sampling over duplicate kernels.
  Inspect their supported preparation, tuning, and workspace interfaces before
  introducing custom implementations.

## Python prepares; Helios serves

- Python loads weights, resolves and autotunes kernels, plans storage, warms
  execution paths, and captures supported graph variants before serving starts.
  `ignition export` writes a runtime core and closes the prepared model.
  `flare run CORE` loads that core in a separate Rust serving process.
- Helios owns sockets, request processing, tokenization, chat templates,
  scheduling, cache operations, graph replay, and output streaming. Serving
  must not call back into Python.
- CUDA graphs are the model execution boundary. Do not introduce an eager
  serving fallback, runtime compilation, graph recapture, or GPU allocation in
  steady-state execution. Reject requests outside prepared capacities explicitly.
- Keep graph executables, captured buffers, resource owners, and the CUDA
  context alive for their full native use. Model state and metadata must be
  explicit storage that Helios can manage without Python execution.
- `ignition debug-serve --direct [--eager]` is the one Python execution path, a
  diagnostic and qualification tool requested by the project owner for
  reproducing model faults without Helios ([direct execution](docs/direct-execution.md)).
  It is never production serving or a fallback, and only its explicit flags
  select it.
- Treat KV, recurrent state, and other model histories as explicit cache state.
  Prefix reuse, branching, and speculative rollback must preserve all relevant
  state, not just token positions and KV pages.

## Model core compatibility

- Every engine version must load models supported by previous engine versions
  without re-export, re-signing, or changes to captured graphs. Preserve loading
  and serving compatibility when changing core metadata, storage contracts,
  and execution code. Retain readers and tests needed by existing cores.
- Execution ABI numbers are descriptive metadata, not loading gates. Validate
  actual graph operations, buffer layouts, capacities, and dependencies. Protobuf
  additions must use defaults that preserve existing core behavior; an absent
  or zero completion token capacity means eight.
- Version semantic changes without abandoning previously supported models.
  Reserve removed Protobuf fields and enum values. Preserve signature, resource,
  hardware, and structural validation. Test previous core layouts through
  loading and serving alongside newly exported cores.

## Configuration

- Flare, Helios, recipes, and components never read an environment variable to
  choose behavior. Feature toggles, experiment switches, tuning overrides,
  debug modes, and alternate code paths selected through the environment are
  prohibited, including temporary ones.
- A setting has one of three homes: the recipe declaration, the configuration
  file, or a command-line flag, as [the CLI design](docs/cli-design.md)
  specifies. A setting that only experiments need is an advanced flag carried
  through the plan into the recipe's `PreparationConfig`, so the plan, its
  printed report, and its recorded measurements describe what was prepared.
- `CUDA_VISIBLE_DEVICES` is the only variable Flare interprets: it renumbers the
  GPUs that `--gpus` indexes. Before starting Helios, Flare narrows that mask to
  the selected GPUs and passes contiguous local ordinals to the runtime. It
  sets `CUDA_FORCE_PRELOAD_LIBRARIES=0` and `CUDA_DISABLE_JIT=1` as fixed driver
  policy: serving loads compiled cubins without driver compilation.
  `ignition.cli.environment` sets the variables that
  PyTorch, b12x, Triton, and the Hugging Face hub read, from Flare's
  configuration, before importing them. Detecting a terminal's capabilities is
  not configuration.
- Tests may read variables that opt into GPU tests and locate their
  checkpoints, such as `FLARE_RUN_CUDA_TESTS`. A test variable must not change
  what the code under test does.

## CUDA kernel and model optimization authorship

Claude Opus 5.5 has repeatedly produced defective low-level kernel changes in
this project. Do not use it to design or implement low-level CUDA kernels or
changes to existing model fusion and GPU execution optimizations. This includes
CUDA C++, PTX, CuTe DSL, and Triton kernels, and changes to projection packing,
fused arithmetic, stream overlap, or scratch aliasing in optimized model paths.
Do not incorporate its generated changes to these mechanisms through another
contributor or agent.

General PyTorch modeling, recipe integration, checkpoint loading, frontend
processing, preparation orchestration, CLI/runtime work, tests, and documentation
are permitted when they do not design or change the restricted mechanisms above.

## Validation and performance

Choose verification by the property at risk, following
[verification guidance](docs/verification.md). Use ordinary tests for observable
behavior and regressions, Kani for properties of sequential Rust over an input
domain, and TLA+ for protocol states and asynchronous ordering. Add only the
checks the affected behavior needs. A passing model does not establish Rust
conformance or CUDA correctness; validate those boundaries separately.

- Compare other runtimes against an exported core running through `flare run`.
  Timings from `ignition debug-serve` are diagnostic measurements and cannot
  establish comparative serving performance.
- When a potential correctness, memory, or performance deficiency appears in
  one model, inspect the shared components and equivalent paths in every
  supported recipe before scoping the fix. Track follow-up work across all
  affected models; distinguish confirmed findings from untested paths.
- Validate the affected execution path and capacities. Model and preparation
  development guidance defines shape, graph-transition, state, and lifetime
  checks; GPU performance guidance defines measurement requirements.
- Keep ordinary tests and raw benchmark measurements. Do not create development
  audit reports, receipts, provenance bundles, or hash inventories.

## Task tracking

Use [docs/TODO.txt](docs/TODO.txt) for miscellaneous follow-up items. Keep entries
brief and link to detailed design or performance notes when needed. Update or
remove entries as work is completed. Keep this file focused on durable project
guidance rather than task lists or progress notes.
