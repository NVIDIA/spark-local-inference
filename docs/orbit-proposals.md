# Orbit proposals

Designed, not implemented. This document specifies four Orbit capabilities:
agent discovery, a fabric check, preparation through agents, and reconciliation
of saved groups. The code provides none of the commands, requests, flags, or
settings named here except where a sentence says so. The
[Orbit reference](orbit.md) documents the implemented commands.

## Discovery

Each agent advertises `_flare-agent._tcp.local.` through the `mdns-sd` crate;
`--no-mdns` disables the advertisement. Its TXT record carries:

| Key | Value |
| --- | --- |
| `v` | Agent protocol version |
| `fp` | Node fingerprint |
| `paired` | `0` for an unpaired node, otherwise `1` |
| `gpu` | Device name from the CUDA driver API, read without creating a context, for example `NVIDIA GB10` |
| `driver` | NVIDIA driver version |
| `free` | Free bytes under the agent cache |
| `rails` | RoCE endpoints as `roce_fabric::encode` writes them: `name address/prefix` per device with an IPv4 GID, `name -` otherwise |

`flare orbit discover` lists advertised agents for three seconds and prints
name, address, fingerprint, pairing state, GPU, and rails. `flare orbit pair`
accepts advertised names as well as hostnames and addresses. It records the
address the advertisement arrived from, which is on the network the controller
shares with the node, and confirms it by connecting. RoCE rail addresses do
not select the agent address. On a network that blocks multicast, pairing uses
an explicit hostname or IP address.

## Fabric check

`flare orbit check --cluster sparks [--roce-ring]` requests fabric endpoints
through STATUS and plans routes with `roce_fabric::plan_routes` or
`plan_shape_routes`, the functions serving uses. It reports the planned rails
and the endpoints that share no subnet. The implemented FABRIC request returns
each agent's active RoCE IPv4 endpoints for planning core transfers; it
exercises no rail.

The check asks every node to exercise the planned rails. The node runs the
RoCE proxy from `helios/native/roce_proxy.c` against a collective region in
registered host memory and writes, from the host, the control words that GPU
kernels write during serving, so the proxy posts one one-shot transfer per
rail to each planned peer. `tests/experimental/b12x/comm/validate_roce_ring_native.py`
in the FlashInfer repository drives FlashInfer's copy of that proxy the same
way, without a CUDA context; this repository holds no copy of the validator.
The check reports each rail's completion and latency. It requires an authorized
controller and a node with no running group.

With `--roce-ring`, the check proposes a node order with a physical link
between every pair of neighbors; `--write` stores it in the cluster. Without
`--roce-ring`, it preserves the listed order.

## Preparation

`flare orbit prepare RECIPE --cluster sparks [export options]` runs
`ignition export` on the cluster's first node and `ignition join` on the others
through a PREPARE request. A `[preparation] checkout` setting in each node's
`~/.config/flare/config.toml` names the checkout whose `.venv` runs Ignition.
Nodes report source identities before starting, and the controller rejects
mismatches.

Ignition authenticates checkpoint and export frames with sequenced HMACs, as
implemented. The controller generates 32 random bytes per preparation and sends
them in each PREPARE frame. Each agent writes them to a private temporary file
and passes its path through `--checkpoint-token-file`, a flag for
`ignition export` and `ignition join` that overrides
`[preparation] checkpoint_token_file`. The token lasts only for that
preparation. Output streams to the controller with `[rank R name]` prefixes,
and the exported core stays on rank zero's host, from where
`flare orbit run NODE:/absolute/path` serves it.

## Reconciliation

`flare orbit apply [--dry-run]` compares saved groups in the orbit file with
the agents' running groups. Its plan identifies groups to stop because their
definitions were removed or changed, groups to start, and their placement.
Stops precede starts so placement can use released GPUs. `--dry-run` reports
the plan without applying it.

The implemented `run`, `stop`, and `push` commands are the execution
primitives. Grants, core verification, placement, metrics, and one rank zero
per group keep the roles the [Orbit reference](orbit.md) defines. The design
adds no consensus service; controllers need a shared desired configuration to
reconcile the same intent.

## Qualification

An implementation counts as qualified when these conditions hold:

- Discovery reports the same node identity as an authenticated connection.
- The fabric check agrees with the rails serving selects and reports a
  deliberately missing link before preparation.
- Agent-managed preparation produces a signed TP4 core that serves
  equivalently to preparation started directly on the hosts.
- Reconciliation preserves unchanged running groups, shows changes without
  applying them in dry-run mode, and releases GPUs before replacement launches.
