# Orbit proposals

These features are not implemented. They are design notes for discovery,
fabric validation, preparation through agents, and reconciliation. Available
commands and their behavior are documented in the [Orbit reference](orbit.md).
Track implementation and qualification in [TODO.txt](TODO.txt).

## Discovery

The proposed discovery service would advertise `_flare-agent._tcp.local.`
through the `mdns-sd` crate, with a `--no-mdns` opt-out. Its TXT record would carry:

| Key | Value |
| --- | --- |
| `v` | Agent protocol version |
| `fp` | Node fingerprint |
| `paired` | `0` for an unpaired node, otherwise `1` |
| `gpu` | Device name from the CUDA driver API, read without creating a context, for example `NVIDIA GB10` |
| `driver` | NVIDIA driver version |
| `free` | Free bytes under the agent cache |
| `rails` | RoCE endpoints as `roce_fabric::encode` writes them: `name address/prefix` per device with an IPv4 GID, `name -` otherwise |

`flare orbit discover` would list advertised agents for three seconds and print
name, address, fingerprint, pairing state, GPU, and rails. `flare orbit pair`
would accept advertised names as well as hostnames and addresses. It would record the
address the advertisement arrived from, which is on the network the controller
shares with the node, and confirm it by connecting. RoCE rail addresses would
not select the agent address. Networks that block multicast would retain
explicit pairing by hostname or IP address.

## Fabric check

The proposed `flare orbit check --cluster sparks [--roce-ring]` would request
fabric endpoints through STATUS and plan routes with `roce_fabric::plan_routes`
or `plan_ring_routes`, the functions serving uses. It would report the planned
rails and endpoints that share no subnet.

It would ask every node to exercise the planned rails. The node would run the
RoCE proxy from `helios/native/roce_proxy.c` against a collective region in
registered host memory and write the control words that GPU kernels write
during serving from the host, so the proxy could post one one-shot transfer per
rail to each planned peer. This is the method of FlashInfer's
`validate_roce_ring_native.py`, which drives the same proxy without a CUDA
context. The check would report each rail's completion and latency. It would
require an authorized controller and a node with no running group.

With `--roce-ring`, the check would propose a node order with a physical link
between every pair of neighbors; `--write` would store it in the cluster.
Without `--roce-ring`, it would preserve the listed order.

## Preparation

The proposed `flare orbit prepare RECIPE --cluster sparks [export options]` would run
`ignition export` on the cluster's first node and `ignition join` on the others
through a PREPARE request. A proposed `[preparation] checkout` setting in each
node's `~/.config/flare/config.toml` would name the checkout whose `.venv` runs
Ignition. Nodes would report source identities before starting, and the
controller would reject mismatches.

Ignition would retain sequenced HMAC frames. The controller would generate
32 random bytes per preparation and send them in each PREPARE frame. Each
agent would write them to a private temporary file and pass its path through
`--checkpoint-token-file`, a proposed flag for `ignition export` and
`ignition join` that would override `[preparation] checkpoint_token_file`. The
token would last only for that preparation. Output would stream to the
controller with `[rank R name]` prefixes, and the exported core would stay
on rank zero's host, from where
`flare orbit run maxwell:/data/flare-runtime/glm53-spark-tp4.safetensors`
serves it.

## Reconciliation

A proposed `flare orbit apply [--dry-run]` would compare saved groups in the
orbit file with the agents' running groups. Its plan would identify groups to
stop because their definitions were removed or changed, groups to start, and
their placement. Stops would precede starts so placement can use released GPUs.
`--dry-run` would report the plan without applying it.

The existing `run`, `stop`, and `push` commands would remain the execution
primitives. Grants, core verification, placement, metrics, and one rank zero
per group would retain their roles. This proposal adds no consensus service;
controllers need a shared desired configuration to reconcile the same intent.

## Qualification

- Discovery must report the same node identity as an authenticated connection.
- Fabric validation must agree with the rails serving selects and report a
  deliberately missing link before preparation.
- Agent-managed preparation must produce a signed TP4 core that serves
  equivalently to preparation started directly on the hosts.
- Reconciliation must preserve unchanged running groups, show changes without
  applying them in dry-run mode, and release GPUs before replacement launches.
