/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! The RoCE fabric a rank's transport uses, found on its host when it serves.
//!
//! A core records what its kernels need of the transport (world, region layout, proxy ABI),
//! but not the fabric it was prepared on: which RDMA devices a host has, their GIDs, and how
//! hosts are cabled differ between clusters and even between launches. Each rank discovers its
//! active RoCEv2 devices, ranks exchange those endpoint lists at the rendezvous, and every rank
//! plans the same routes from them: which local and remote device carries each rail to each peer.
//!
//! Routing follows b12x's `_routes.py`. On a switched fabric rail `h` pairs device `h` at both
//! ends. Hosts cabled point to point (three Sparks in a ring) reach a neighbour only through
//! devices on a shared subnet, so rails then pair devices per peer by subnet. A fabric can combine
//! both: Sparks whose first port joins a switch and whose second port is cabled to one partner
//! reach that partner over subnets only the two of them hold and every other peer over the
//! switch's subnets. A subnet's breadth is the number of ranks holding an address on it: two for a
//! direct cable, the whole world for a switch. Index pairing applies only when every subnet joining
//! two or more ranks joins all of them; otherwise per-peer pairing prefers the narrowest subnets,
//! so a cabled partner is reached over its cable. IPv4 link-local subnets (169.254.0.0/16) count as
//! world-wide, because hosts using link-local addressing share that subnet whether or not their
//! ports are cabled together. Rails are ordered by breadth and then link address so both ends agree.
//!
//! Each endpoint also records its PCIe link (PCI domain:bus:device): a Spark's ConnectX-7 exposes
//! each port as two functions, one behind each of its two PCIe x4 links, and the functions behind
//! one link share its bandwidth. Index pairing requires distinct links among every rank's rail
//! devices, and per-peer pairing prefers rails on distinct links at both ends before it compares
//! breadth, so two ranks that hold every subnet in common (a cabled pair that also shares a switch)
//! still place one rail behind each link.

use std::net::Ipv4Addr;
use std::path::Path;

use crate::{ExecutionError, Failure, Result};

/// Each peer is reached over at most this many devices.
pub const MAX_RAILS: usize = 2;
/// A rank opens at most this many devices.
pub const MAX_DEVICES: usize = 4;
/// b12x's default GID index: RoCEv2 over IPv4 on a DGX Spark.
const PREFERRED_GID_INDEX: u32 = 3;
const GID_TABLE: u32 = 16;

fn failure(message: impl Into<String>) -> crate::Error {
    ExecutionError::Poisoned(Failure::new(message.into())).into()
}

/// One RDMA device: its name, the address and prefix length when its GID is IPv4-mapped, and the
/// PCIe link it shares with sibling functions (PCI domain:bus:device) when known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub name: String,
    pub ipv4: Option<(Ipv4Addr, u8)>,
    pub link: Option<String>,
}

/// The PCI domain:bus:device of an RDMA function's `device` entry under `root/name`.
fn pcie_link(root: &Path, name: &str) -> Option<String> {
    let target = std::fs::read_link(root.join(name).join("device")).ok()?;
    let address = target.file_name()?.to_str()?;
    let (base, function) = address.rsplit_once('.')?;
    (base.matches(':').count() == 2
        && !function.is_empty()
        && function.bytes().all(|b| b.is_ascii_digit()))
    .then(|| base.to_owned())
}

impl Endpoint {
    fn network(&self) -> Option<(u32, u8)> {
        self.ipv4.map(|(address, prefix)| {
            let mask = if prefix == 0 {
                0
            } else {
                u32::MAX << (32 - prefix as u32)
            };
            (u32::from(address) & mask, prefix)
        })
    }
}

/// This host's RoCE devices in name order, and the GID index and traffic class they use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fabric {
    pub gid_index: u32,
    pub traffic_class: u32,
    pub endpoints: Vec<Endpoint>,
}

/// How the ranks of a multi-host core link over RoCE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// Every rank links to every other rank.
    Mesh,
    /// Experimental ring in rank order: each rank links to its two neighbors and the native proxy
    /// forwards writes to every other rank around the shorter arc.
    Ring,
    /// Experimental chain in rank order: a ring with no link between rank 0 and the last rank, so
    /// the ends have one neighbor and the proxy forwards along the chain.
    Chain,
}

impl Shape {
    /// The shape two command-line flags select; the chain flag wins because it implies relaying.
    pub fn from_flags(ring: bool, chain: bool) -> Self {
        if chain {
            Shape::Chain
        } else if ring {
            Shape::Ring
        } else {
            Shape::Mesh
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Shape::Mesh => "mesh",
            Shape::Ring => "ring",
            Shape::Chain => "chain",
        }
    }

    /// Whether the native proxy relays writes between ranks that hold no direct link.
    pub fn relays(self) -> bool {
        self != Shape::Mesh
    }

    /// Whether `rank` holds a direct link to `peer` among `world` ranks; a rank never links to itself.
    pub fn linked(self, rank: usize, peer: usize, world: usize) -> bool {
        match self {
            Shape::Mesh => peer != rank,
            Shape::Ring => {
                peer != rank && (peer == (rank + 1) % world || peer == (rank + world - 1) % world)
            }
            Shape::Chain => peer + 1 == rank || peer == rank + 1,
        }
    }
}

/// Which links may carry rails.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Links {
    /// Every verified link. A peer that shares a subnet held by fewer ranks than the world (a
    /// direct cable) is reached over it rather than over a subnet every rank holds (a switch).
    #[default]
    All,
    /// Only subnets every rank holds, as if the fabric were a switch alone: direct cables carry no
    /// rails. With two ranks every subnet joins both, so a cable is indistinguishable from a switch.
    Shared,
}

impl Links {
    /// The links a command-line flag selects.
    pub fn from_switch_only(switch_only: bool) -> Self {
        if switch_only {
            Links::Shared
        } else {
            Links::All
        }
    }
}

/// The devices this rank opens and, per peer, the (local, remote) device indexes of each rail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    pub hcas: Vec<String>,
    pub gid_index: u32,
    pub traffic_class: u32,
    pub rails: usize,
    pub routes: Vec<Vec<(u32, u32)>>,
    /// Per peer, whether every rail rides a direct cable: a subnet held by fewer ranks than the
    /// world, which only a cluster of three or more ranks can identify.
    pub direct: Vec<bool>,
    pub shape: Shape,
    pub links: Links,
}

impl Topology {
    /// Each linked peer with the local devices carrying its rails, in rail order, marked `direct`
    /// when every rail rides a direct cable, for example `rank 1 direct roceP2p1s0f1,rocep1s0f1`.
    pub fn peer_routes(&self) -> Vec<String> {
        self.routes
            .iter()
            .enumerate()
            .filter(|(_, links)| !links.is_empty())
            .map(|(peer, links)| {
                let devices: Vec<&str> = links
                    .iter()
                    .map(|&(local, _)| self.hcas[local as usize].as_str())
                    .collect();
                let direct = if self.direct.get(peer) == Some(&true) {
                    " direct"
                } else {
                    ""
                };
                format!("rank {peer}{direct} {}", devices.join(","))
            })
            .collect()
    }
}

fn read(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|text| text.trim().to_owned())
}

/// The IPv4 address an IPv4-mapped GID (::ffff:a.b.c.d) holds.
fn mapped_ipv4(gid: &str) -> Option<Ipv4Addr> {
    let hex: String = gid.chars().filter(|c| *c != ':').collect();
    if hex.len() != 32
        || !hex[..20].bytes().all(|b| b == b'0')
        || !hex[20..24].eq_ignore_ascii_case("ffff")
    {
        return None;
    }
    u32::from_str_radix(&hex[24..], 16).ok().map(Ipv4Addr::from)
}

/// Discover the RoCE devices under `root` (normally /sys/class/infiniband): every device whose
/// port 1 is active and holds a RoCEv2 GID at the chosen index. The index is 3 when every active
/// device has a RoCEv2 IPv4 GID there, otherwise the lowest index at which one does. `prefix`
/// gives a network device's prefix length for an address it holds.
pub fn discover_in(root: &Path, prefix: impl Fn(&str, Ipv4Addr) -> Option<u8>) -> Result<Fabric> {
    let mut names: Vec<String> = std::fs::read_dir(root)
        .map_err(|error| {
            failure(format!(
                "cannot list RDMA devices in {}: {error}",
                root.display()
            ))
        })?
        .filter_map(|entry| {
            entry
                .ok()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
        })
        .filter(|name| {
            read(&root.join(name).join("ports/1/state"))
                .is_some_and(|state| state.contains("ACTIVE"))
        })
        .collect();
    names.sort();
    let roce_v2_ipv4 = |name: &str, index: u32| {
        let port = root.join(name).join("ports/1");
        read(&port.join(format!("gid_attrs/types/{index}")))
            .is_some_and(|kind| kind.eq_ignore_ascii_case("RoCE v2"))
            && read(&port.join(format!("gids/{index}")))
                .as_deref()
                .and_then(mapped_ipv4)
                .is_some()
    };
    let gid_index = if !names.is_empty()
        && names
            .iter()
            .all(|name| roce_v2_ipv4(name, PREFERRED_GID_INDEX))
    {
        PREFERRED_GID_INDEX
    } else {
        (0..GID_TABLE)
            .find(|&index| names.iter().any(|name| roce_v2_ipv4(name, index)))
            .ok_or_else(|| failure("no active RDMA device has a RoCEv2 IPv4 GID"))?
    };
    let endpoints: Vec<Endpoint> = names
        .iter()
        .filter(|name| roce_v2_ipv4(name, gid_index))
        .take(MAX_DEVICES)
        .map(|name| {
            let port = root.join(name).join("ports/1");
            let address = read(&port.join(format!("gids/{gid_index}")))
                .as_deref()
                .and_then(mapped_ipv4);
            let device = read(&port.join(format!("gid_attrs/ndevs/{gid_index}")));
            let ipv4 = address
                .zip(device)
                .and_then(|(address, device)| prefix(&device, address).map(|p| (address, p)));
            Endpoint {
                name: name.clone(),
                ipv4,
                link: pcie_link(root, name),
            }
        })
        .collect();
    Ok(Fabric {
        gid_index,
        traffic_class: 0,
        endpoints,
    })
}

/// The prefix length of the IPv4 address `address` on network device `device`, from getifaddrs.
#[cfg(target_os = "linux")]
fn interface_prefix(device: &str, address: Ipv4Addr) -> Option<u8> {
    let mut list: *mut libc::ifaddrs = std::ptr::null_mut();
    // SAFETY: getifaddrs fills a list this function frees before returning; it reads entries only
    // while the list is live and checks every pointer it dereferences.
    unsafe {
        if libc::getifaddrs(&mut list) != 0 {
            return None;
        }
        let mut found = None;
        let mut entry = list;
        while !entry.is_null() {
            let item = &*entry;
            entry = item.ifa_next;
            if item.ifa_addr.is_null()
                || item.ifa_netmask.is_null()
                || (*item.ifa_addr).sa_family as i32 != libc::AF_INET
            {
                continue;
            }
            if std::ffi::CStr::from_ptr(item.ifa_name).to_str() != Ok(device) {
                continue;
            }
            let held = &*(item.ifa_addr as *const libc::sockaddr_in);
            if Ipv4Addr::from(u32::from_be(held.sin_addr.s_addr)) != address {
                continue;
            }
            let mask = &*(item.ifa_netmask as *const libc::sockaddr_in);
            found = Some(u32::from_be(mask.sin_addr.s_addr).count_ones() as u8);
            break;
        }
        libc::freeifaddrs(list);
        found
    }
}

/// Discover this host's RoCE devices.
#[cfg(target_os = "linux")]
pub fn discover() -> Result<Fabric> {
    discover_in(Path::new("/sys/class/infiniband"), interface_prefix)
}

/// Encode endpoints for the rendezvous exchange: one `name address/prefix` line per device, or
/// `name -` for a device without an IPv4 GID, followed by ` domain:bus:device` when its PCIe link
/// is known.
pub fn encode(endpoints: &[Endpoint]) -> Vec<u8> {
    endpoints
        .iter()
        .map(|e| {
            let address = match e.ipv4 {
                Some((address, prefix)) => format!("{address}/{prefix}"),
                None => "-".to_owned(),
            };
            match &e.link {
                Some(link) => format!("{} {address} {link}\n", e.name),
                None => format!("{} {address}\n", e.name),
            }
        })
        .collect::<String>()
        .into_bytes()
}

pub fn decode(bytes: &[u8]) -> Result<Vec<Endpoint>> {
    let text = std::str::from_utf8(bytes).map_err(|_| failure("RoCE endpoints are not text"))?;
    text.lines()
        .map(|line| {
            let mut fields = line.split(' ');
            let (Some(name), Some(address), link, None) =
                (fields.next(), fields.next(), fields.next(), fields.next())
            else {
                return Err(failure("malformed RoCE endpoint"));
            };
            if link.is_some_and(|link| link.matches(':').count() != 2) {
                return Err(failure("malformed RoCE endpoint PCIe link"));
            }
            let ipv4 = match address {
                "-" => None,
                _ => {
                    let (ip, prefix) = address
                        .split_once('/')
                        .ok_or_else(|| failure("malformed RoCE endpoint"))?;
                    let ip = ip
                        .parse()
                        .map_err(|_| failure("malformed RoCE endpoint address"))?;
                    let prefix: u8 = prefix
                        .parse()
                        .map_err(|_| failure("malformed RoCE endpoint prefix"))?;
                    if prefix > 32 {
                        return Err(failure("malformed RoCE endpoint prefix"));
                    }
                    Some((ip, prefix))
                }
            };
            Ok(Endpoint {
                name: name.to_owned(),
                ipv4,
                link: link.map(str::to_owned),
            })
        })
        .collect()
}

fn same_link(a: &Endpoint, b: &Endpoint) -> bool {
    matches!((a.network(), b.network()), (Some(x), Some(y)) if x == y)
}

/// Per IPv4 network, the number of ranks holding an address on it. A link-local network
/// (169.254.0.0/16) counts as held by every rank, since sharing it does not imply a cable.
fn breadths(endpoints: &[Vec<Endpoint>]) -> std::collections::BTreeMap<(u32, u8), usize> {
    let world = endpoints.len();
    let mut counts = std::collections::BTreeMap::new();
    for devices in endpoints {
        let networks: std::collections::BTreeSet<_> =
            devices.iter().filter_map(Endpoint::network).collect();
        for network in networks {
            *counts.entry(network).or_insert(0) += 1;
        }
    }
    for ((address, prefix), count) in counts.iter_mut() {
        if *prefix >= 16 && address & 0xffff_0000 == 0xa9fe_0000 {
            *count = world;
        }
    }
    counts
}

/// How many of `devices` share a known PCIe link with an earlier one.
fn shared_links<'a>(devices: impl IntoIterator<Item = &'a Endpoint>) -> usize {
    let known: Vec<&str> = devices
        .into_iter()
        .filter_map(|d| d.link.as_deref())
        .collect();
    let distinct: std::collections::BTreeSet<&str> = known.iter().copied().collect();
    known.len() - distinct.len()
}

/// True when rail `h` can pair device `h` with device `h` between every pair of ranks, no rank's
/// rail devices share a PCIe link, and, when `links` admits direct cables, no subnet links some
/// ranks but not all of them (a direct cable that per-peer pairing prefers).
fn index_pairing_valid(
    endpoints: &[Vec<Endpoint>],
    rails: usize,
    breadths: &std::collections::BTreeMap<(u32, u8), usize>,
    links: Links,
) -> bool {
    let world = endpoints.len();
    if links == Links::All
        && breadths
            .values()
            .any(|&breadth| 2 <= breadth && breadth < world)
    {
        return false;
    }
    if endpoints.iter().any(|devices| devices.len() < rails) {
        return false;
    }
    if endpoints
        .iter()
        .any(|devices| shared_links(&devices[..rails]) != 0)
    {
        return false;
    }
    for a in 0..endpoints.len() {
        for b in a + 1..endpoints.len() {
            for h in 0..rails {
                let (x, y) = (&endpoints[a][h], &endpoints[b][h]);
                // A device without an IPv4 GID cannot be checked: trust the device order.
                if x.ipv4.is_some() && y.ipv4.is_some() && !same_link(x, y) {
                    return false;
                }
            }
        }
    }
    true
}

/// (unverified, subnet breadth, network, lower address, higher address) of a candidate link.
type Key = (u32, u64, u64, u64, u64);

/// The `rails` candidates with distinct local and remote devices that use the most verified
/// links, then the fewest rails sharing a PCIe link at either end (`local` and `remote` name the
/// devices), then the narrowest subnets (smallest total breadth), ties broken by the smallest
/// sorted keys; None when no such set exists.
fn best_matching(
    candidates: &[(Key, u32, u32)],
    rails: usize,
    local: &[Endpoint],
    remote: &[Endpoint],
) -> Option<Vec<(u32, u32)>> {
    let mut best: Option<((u32, usize, u64, Vec<Key>), Vec<(u32, u32)>)> = None;
    let mut visit = |combo: &[usize]| {
        let chosen: Vec<_> = combo.iter().map(|&i| candidates[i]).collect();
        let locals: std::collections::BTreeSet<_> = chosen.iter().map(|c| c.1).collect();
        let remotes: std::collections::BTreeSet<_> = chosen.iter().map(|c| c.2).collect();
        if locals.len() < rails || remotes.len() < rails {
            return;
        }
        let shared = shared_links(chosen.iter().filter_map(|c| local.get(c.1 as usize)))
            + shared_links(chosen.iter().filter_map(|c| remote.get(c.2 as usize)));
        let score = (
            chosen.iter().map(|c| c.0.0).sum(),
            shared,
            chosen.iter().map(|c| c.0.1).sum(),
            chosen.iter().map(|c| c.0).collect(),
        );
        if best.as_ref().is_none_or(|(current, _)| score < *current) {
            best = Some((score, chosen.iter().map(|c| (c.1, c.2)).collect()));
        }
    };
    // Combinations in lexicographic order, as itertools.combinations yields them.
    let n = candidates.len();
    if rails == 0 || rails > n {
        return None;
    }
    let mut index: Vec<usize> = (0..rails).collect();
    loop {
        visit(&index);
        let Some(i) = (0..rails).rev().find(|&i| index[i] != i + n - rails) else {
            break;
        };
        index[i] += 1;
        for j in i + 1..rails {
            index[j] = index[j - 1] + 1;
        }
    }
    best.map(|(_, links)| links)
}

/// Route planning over every rank's endpoints: subnet breadths and whether index pairing applies
/// are properties of the whole fabric, so every rank and every peer see the same ones.
struct Planner<'a> {
    endpoints: &'a [Vec<Endpoint>],
    breadths: std::collections::BTreeMap<(u32, u8), usize>,
    rails: usize,
    links: Links,
    index_pairing: bool,
}

impl<'a> Planner<'a> {
    fn new(endpoints: &'a [Vec<Endpoint>], rails: usize, links: Links) -> Result<Self> {
        if !(1..=MAX_RAILS).contains(&rails) {
            return Err(failure("invalid RoCE route request"));
        }
        let breadths = breadths(endpoints);
        let index_pairing = index_pairing_valid(endpoints, rails, &breadths, links);
        Ok(Self {
            endpoints,
            breadths,
            rails,
            links,
            index_pairing,
        })
    }

    /// The (local device, remote device) indexes carrying each rail from `rank` to `peer`.
    fn peer(&self, rank: usize, peer: usize) -> Result<Vec<(u32, u32)>> {
        let rails = self.rails;
        if rank >= self.endpoints.len() || peer >= self.endpoints.len() {
            return Err(failure("invalid RoCE route request"));
        }
        if peer == rank {
            return Ok(Vec::new());
        }
        if self.index_pairing {
            return Ok((0..rails as u32).map(|h| (h, h)).collect());
        }
        let (local, remote) = (&self.endpoints[rank], &self.endpoints[peer]);
        let world = self.endpoints.len();
        // Verified same-subnet links keyed by subnet breadth (narrowest first, so a direct cable
        // precedes a switch), network and the link's two addresses, and same-index pairs without an
        // IPv4 GID at either end, trusted in device order. Keys are symmetric. Links::Shared admits
        // only subnets every rank holds.
        let mut candidates: Vec<(Key, u32, u32)> = Vec::new();
        for (l, x) in local.iter().enumerate() {
            for (r, y) in remote.iter().enumerate() {
                if !same_link(x, y) {
                    continue;
                }
                let network = x.network().unwrap();
                let breadth = self.breadths[&network];
                if self.links == Links::Shared && breadth < world {
                    continue;
                }
                let (a, b) = (
                    u32::from(x.ipv4.unwrap().0) as u64,
                    u32::from(y.ipv4.unwrap().0) as u64,
                );
                candidates.push((
                    (0, breadth as u64, network.0 as u64, a.min(b), a.max(b)),
                    l as u32,
                    r as u32,
                ));
            }
        }
        for h in 0..local.len().min(remote.len()) {
            if local[h].ipv4.is_none() || remote[h].ipv4.is_none() {
                candidates.push(((1, 0, h as u64, 0, 0), h as u32, h as u32));
            }
        }
        candidates.sort();
        best_matching(&candidates, rails, local, remote).ok_or_else(|| {
            let reach = (1..rails).rev().find(|&k| best_matching(&candidates, k, local, remote).is_some()).unwrap_or(0);
            failure(format!(
                "rank {rank}: {reach} of {rails} RoCE rails have a link to rank {peer}; local {local:?}, remote {remote:?}"
            ))
        })
    }
}

/// Per peer, the (local device, remote device) indexes carrying each rail; empty for `rank`.
pub fn plan_routes(
    endpoints: &[Vec<Endpoint>],
    rank: usize,
    rails: usize,
    links: Links,
) -> Result<Vec<Vec<(u32, u32)>>> {
    if rank >= endpoints.len() {
        return Err(failure("invalid RoCE route request"));
    }
    let planner = Planner::new(endpoints, rails, links)?;
    (0..endpoints.len())
        .map(|peer| planner.peer(rank, peer))
        .collect()
}

/// The rails from `rank` to `peer` alone, planned over every rank's endpoints exactly as
/// `plan_routes` plans them, so that a caller connecting one pair at a time (fabric TCP transfers)
/// uses the links the RDMA transport uses.
pub fn plan_link(
    endpoints: &[Vec<Endpoint>],
    rank: usize,
    peer: usize,
    rails: usize,
    links: Links,
) -> Result<Vec<(u32, u32)>> {
    Planner::new(endpoints, rails, links)?.peer(rank, peer)
}

/// Per peer of `rank`, whether every rail rides a subnet narrower than the world: a direct cable
/// rather than a subnet every rank holds. A two-rank fabric cannot tell the two apart.
fn direct_peers(endpoints: &[Vec<Endpoint>], rank: usize, routes: &[Vec<(u32, u32)>]) -> Vec<bool> {
    let world = endpoints.len();
    let breadths = breadths(endpoints);
    routes
        .iter()
        .map(|links| {
            world > 2
                && !links.is_empty()
                && links.iter().all(|&(local, _)| {
                    endpoints[rank][local as usize]
                        .network()
                        .is_some_and(|network| breadths[&network] < world)
                })
        })
        .collect()
}

/// The routes of `shape` for `rank`: a mesh routes to every rank; a ring or chain routes only to
/// the ranks `shape` links, and the native proxy forwards writes to the others. The same plan works
/// over a switch while keeping unlinked QPs unconnected.
pub fn plan_shape_routes(
    endpoints: &[Vec<Endpoint>],
    rank: usize,
    rails: usize,
    shape: Shape,
    links: Links,
) -> Result<Vec<Vec<(u32, u32)>>> {
    if shape == Shape::Mesh {
        return plan_routes(endpoints, rank, rails, links);
    }
    if !(2..=16).contains(&endpoints.len()) || rank >= endpoints.len() {
        return Err(failure("experimental RoCE ring requires 2..16 ranks"));
    }
    let world = endpoints.len();
    let peers: Vec<_> = (0..world).filter(|p| *p == rank || shape.linked(rank, *p, world)).collect();
    let subset: Vec<_> = peers.iter().map(|p| endpoints[*p].clone()).collect();
    let local = peers.iter().position(|p| *p == rank).unwrap();
    let planned = plan_routes(&subset, local, rails, links)?;
    Ok((0..world)
        .map(|p| {
            peers
                .iter()
                .position(|q| *q == p)
                .map_or_else(Vec::new, |i| planned[i].clone())
        })
        .collect())
}

impl Fabric {
    /// The topology this rank opens for `shape` over `links`, from every rank's endpoints in rank
    /// order.
    pub fn topology(
        &self,
        endpoints: &[Vec<Endpoint>],
        rank: usize,
        shape: Shape,
        links: Links,
    ) -> Result<Topology> {
        if endpoints.get(rank) != Some(&self.endpoints) || endpoints.iter().any(Vec::is_empty) {
            return Err(failure("every rank needs at least one RoCE device"));
        }
        let rails = endpoints.iter().map(Vec::len).min().unwrap().min(MAX_RAILS);
        let routes = plan_shape_routes(endpoints, rank, rails, shape, links)?;
        Ok(Topology {
            hcas: self.endpoints.iter().map(|e| e.name.clone()).collect(),
            gid_index: self.gid_index,
            traffic_class: self.traffic_class,
            rails,
            direct: direct_peers(endpoints, rank, &routes),
            routes,
            shape,
            links,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(name: &str, address: &str) -> Endpoint {
        let (ip, prefix) = address.split_once('/').unwrap();
        Endpoint {
            name: name.into(),
            ipv4: Some((ip.parse().unwrap(), prefix.parse().unwrap())),
            link: None,
        }
    }

    fn gid(address: Ipv4Addr) -> String {
        let [a, b, c, d] = address.octets();
        format!("0000:0000:0000:0000:0000:ffff:{a:02x}{b:02x}:{c:02x}{d:02x}")
    }

    fn device(root: &Path, name: &str, state: &str, gids: &[(u32, &str, Ipv4Addr)]) {
        let port = root.join(name).join("ports/1");
        for directory in ["gids", "gid_attrs/types", "gid_attrs/ndevs"] {
            std::fs::create_dir_all(port.join(directory)).unwrap();
        }
        std::fs::write(port.join("state"), format!("{state}\n")).unwrap();
        for (index, kind, address) in gids {
            std::fs::write(port.join(format!("gids/{index}")), gid(*address)).unwrap();
            std::fs::write(
                port.join(format!("gid_attrs/types/{index}")),
                format!("{kind}\n"),
            )
            .unwrap();
            std::fs::write(
                port.join(format!("gid_attrs/ndevs/{index}")),
                format!("net-{name}\n"),
            )
            .unwrap();
        }
    }

    fn scratch(label: &str) -> std::path::PathBuf {
        let root =
            std::env::temp_dir().join(format!("helios-fabric-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn active_roce_v2_devices_are_found_in_name_order_with_their_prefixes() {
        let root = scratch("found");
        let (a, b) = (Ipv4Addr::new(10, 200, 1, 12), Ipv4Addr::new(10, 200, 0, 12));
        device(
            &root,
            "rocep1s0f0",
            "4: ACTIVE",
            &[(1, "IB/RoCE v1", a), (3, "RoCE v2", a)],
        );
        device(&root, "roceP2p1s0f0", "4: ACTIVE", &[(3, "RoCE v2", b)]);
        // sysfs links each RDMA device to its PCI function; the link drops the function number.
        let function = root.join("pci/0002:01:00.0");
        std::fs::create_dir_all(&function).unwrap();
        std::os::unix::fs::symlink(&function, root.join("roceP2p1s0f0/device")).unwrap();
        device(
            &root,
            "rocep1s0f1",
            "1: DOWN",
            &[(3, "RoCE v2", Ipv4Addr::new(10, 9, 9, 9))],
        );
        let fabric = discover_in(&root, |device, _| {
            (device != "net-rocep1s0f0").then_some(24)
        })
        .unwrap();
        assert_eq!(fabric.gid_index, 3);
        assert_eq!(
            fabric.endpoints,
            vec![
                Endpoint {
                    name: "roceP2p1s0f0".into(),
                    ipv4: Some((b, 24)),
                    link: Some("0002:01:00".into()),
                },
                // A device whose netmask cannot be read has no checkable address.
                Endpoint {
                    name: "rocep1s0f0".into(),
                    ipv4: None,
                    link: None,
                },
            ]
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn another_roce_v2_index_serves_when_index_three_is_not_one() {
        let root = scratch("index");
        let a = Ipv4Addr::new(192, 168, 7, 2);
        device(
            &root,
            "mlx5_0",
            "4: ACTIVE",
            &[(1, "RoCE v2", a), (3, "IB/RoCE v1", a)],
        );
        let fabric = discover_in(&root, |_, _| Some(16)).unwrap();
        assert_eq!((fabric.gid_index, fabric.endpoints.len()), (1, 1));
        std::fs::remove_dir_all(&root).unwrap();
        std::fs::create_dir_all(&root).unwrap();
        assert!(discover_in(&root, |_, _| Some(16)).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn endpoints_survive_the_rendezvous_encoding() {
        let endpoints = vec![
            spark("rocep1s0f0", "10.200.1.11/24"),
            endpoint("rocep1s0f1", "10.200.5.11/24"),
            Endpoint {
                name: "x".into(),
                ipv4: None,
                link: Some("0002:01:00".into()),
            },
        ];
        assert_eq!(decode(&encode(&endpoints)).unwrap(), endpoints);
        assert_eq!(
            decode(b"rocep1s0f0 10.200.1.11/24\n").unwrap(),
            vec![endpoint("rocep1s0f0", "10.200.1.11/24")]
        );
        assert!(decode(b"rocep1s0f0 10.0.0.1/40\n").is_err());
        assert!(decode(b"rocep1s0f0 10.0.0.1/24 0000:01\n").is_err());
        assert!(decode(b"rocep1s0f0 10.0.0.1/24 0000:01:00 extra\n").is_err());
    }

    #[test]
    fn a_switched_fabric_pairs_devices_by_index() {
        let rank = |host: u8| {
            vec![
                endpoint("f0", &format!("10.200.1.{host}/24")),
                endpoint("f1", &format!("10.200.0.{host}/24")),
            ]
        };
        let endpoints = vec![rank(12), rank(13), rank(11), rank(14)];
        let routes = plan_routes(&endpoints, 1, 2, Links::All).unwrap();
        assert_eq!(
            routes,
            vec![
                vec![(0, 0), (1, 1)],
                vec![],
                vec![(0, 0), (1, 1)],
                vec![(0, 0), (1, 1)]
            ]
        );
    }

    /// Four Sparks whose first port joins a switch (10.200.0.0/24 and 10.200.1.0/24) and whose
    /// second port is cabled to one partner: ranks 1 and 2 over 10.200.2.0/24 and 10.200.3.0/24,
    /// ranks 0 and 3 over 10.200.4.0/24 and 10.200.5.0/24. Devices are in discovery (name) order.
    /// A DGX Spark ConnectX-7 function: each PCIe x4 link carries one function of each port.
    fn spark(name: &str, address: &str) -> Endpoint {
        let link = if name.starts_with("roceP2") {
            "0002:01:00"
        } else {
            "0000:01:00"
        };
        Endpoint {
            link: Some(link.into()),
            ..endpoint(name, address)
        }
    }

    fn switch_with_partner_cables() -> Vec<Vec<Endpoint>> {
        (0..4u32)
            .map(|rank| {
                let host = 11 + rank;
                let direct = if rank == 1 || rank == 2 { 2 } else { 4 };
                vec![
                    spark("roceP2p1s0f0", &format!("10.200.0.{host}/24")),
                    spark("roceP2p1s0f1", &format!("10.200.{direct}.{host}/24")),
                    spark("rocep1s0f0", &format!("10.200.1.{host}/24")),
                    spark("rocep1s0f1", &format!("10.200.{}.{host}/24", direct + 1)),
                ]
            })
            .collect()
    }

    const PARTNER: [usize; 4] = [3, 2, 1, 0];
    const SWITCH_DEVICES: [&str; 2] = ["roceP2p1s0f0", "rocep1s0f0"];
    const DIRECT_DEVICES: [&str; 2] = ["roceP2p1s0f1", "rocep1s0f1"];

    fn route_devices(
        endpoints: &[Vec<Endpoint>],
        rank: usize,
        links: &[(u32, u32)],
    ) -> Vec<String> {
        let mut names: Vec<_> = links
            .iter()
            .map(|&(local, _)| endpoints[rank][local as usize].name.clone())
            .collect();
        names.sort();
        names
    }

    fn sorted(names: [&str; 2]) -> Vec<String> {
        let mut names: Vec<String> = names.iter().map(|n| n.to_string()).collect();
        names.sort();
        names
    }

    #[test]
    fn a_partner_cable_carries_both_rails_to_the_partner_and_the_switch_the_rest() {
        // Listing the switch functions first would make index pairing valid; the cable still wins.
        for switch_first in [false, true] {
            let mut endpoints = switch_with_partner_cables();
            if switch_first {
                for devices in &mut endpoints {
                    devices.sort_by_key(|e| !SWITCH_DEVICES.contains(&e.name.as_str()));
                }
            }
            let plans: Vec<_> = (0..4)
                .map(|rank| plan_routes(&endpoints, rank, 2, Links::All).unwrap())
                .collect();
            for rank in 0..4 {
                assert!(plans[rank][rank].is_empty());
                for peer in (0..4).filter(|&p| p != rank) {
                    let links = &plans[rank][peer];
                    let expected = if peer == PARTNER[rank] {
                        DIRECT_DEVICES
                    } else {
                        SWITCH_DEVICES
                    };
                    assert_eq!(route_devices(&endpoints, rank, links), sorted(expected));
                    for &(l, r) in links {
                        assert!(same_link(
                            &endpoints[rank][l as usize],
                            &endpoints[peer][r as usize]
                        ));
                    }
                    let mirrored: Vec<_> = plans[peer][rank].iter().map(|&(l, r)| (r, l)).collect();
                    assert_eq!(*links, mirrored, "ranks {rank} and {peer} disagree");
                }
            }
        }
    }

    #[test]
    fn a_partner_cable_down_at_one_end_moves_that_pair_to_the_switch() {
        let mut endpoints = switch_with_partner_cables();
        endpoints[2].retain(|e| SWITCH_DEVICES.contains(&e.name.as_str()));
        let plans: Vec<_> = (0..4)
            .map(|rank| plan_routes(&endpoints, rank, 2, Links::All).unwrap())
            .collect();
        let devices =
            |rank: usize, peer: usize| route_devices(&endpoints, rank, &plans[rank][peer]);
        assert_eq!(devices(1, 2), sorted(SWITCH_DEVICES));
        let mirrored: Vec<_> = plans[2][1].iter().map(|&(l, r)| (r, l)).collect();
        assert_eq!(plans[1][2], mirrored);
        assert_eq!(devices(0, 3), sorted(DIRECT_DEVICES));
        assert_eq!(devices(3, 0), sorted(DIRECT_DEVICES));
    }

    #[test]
    fn link_local_addresses_on_two_uncabled_ports_do_not_attract_rails() {
        // Ranks 1 and 3 share 169.254.0.0/16 on ports cabled to other hosts.
        let switch = |host: u8| {
            vec![
                endpoint("roceP2p1s0f0", &format!("10.200.0.{host}/24")),
                endpoint("rocep1s0f0", &format!("10.200.1.{host}/24")),
            ]
        };
        let mut endpoints = vec![switch(14), switch(12), switch(13), switch(11)];
        endpoints[1].push(endpoint("rocep1s0f1", "169.254.210.217/16"));
        endpoints[3].push(endpoint("rocep1s0f1", "169.254.21.112/16"));
        for rank in 0..4 {
            let routes = plan_routes(&endpoints, rank, 2, Links::All).unwrap();
            for (peer, links) in routes.iter().enumerate() {
                let expected = if peer == rank {
                    vec![]
                } else {
                    vec![(0, 0), (1, 1)]
                };
                assert_eq!(*links, expected, "rank {rank} to {peer}");
            }
        }
    }

    #[test]
    fn a_cabled_pair_keeps_its_two_rails_on_separate_pcie_links() {
        // Two ranks cabled to each other and to a switch hold every subnet in common; index pairing
        // in name order would put both rails behind one PCIe x4 link.
        let all = switch_with_partner_cables();
        let pair = vec![all[0].clone(), all[3].clone()];
        for rank in 0..2 {
            let links = &plan_routes(&pair, rank, 2, Links::All).unwrap()[1 - rank];
            let pcie: std::collections::BTreeSet<_> = links
                .iter()
                .map(|&(l, _)| pair[rank][l as usize].link.clone())
                .collect();
            assert_eq!(pcie.len(), 2, "rank {rank}: {links:?}");
            assert_eq!(route_devices(&pair, rank, links), sorted(SWITCH_DEVICES));
        }
        // Without PCIe link information the planner keeps device order.
        let unknown: Vec<Vec<Endpoint>> = pair
            .iter()
            .map(|devices| {
                devices
                    .iter()
                    .map(|e| Endpoint {
                        link: None,
                        ..e.clone()
                    })
                    .collect()
            })
            .collect();
        assert_eq!(
            plan_routes(&unknown, 0, 2, Links::All).unwrap()[1],
            vec![(0, 0), (1, 1)]
        );
    }

    #[test]
    fn switch_only_links_route_every_peer_over_the_switch_despite_partner_cables() {
        let endpoints = switch_with_partner_cables();
        let plans: Vec<_> = (0..4)
            .map(|rank| plan_routes(&endpoints, rank, 2, Links::Shared).unwrap())
            .collect();
        for rank in 0..4 {
            for peer in (0..4).filter(|&p| p != rank) {
                let links = &plans[rank][peer];
                assert_eq!(
                    route_devices(&endpoints, rank, links),
                    sorted(SWITCH_DEVICES)
                );
                let mirrored: Vec<_> = plans[peer][rank].iter().map(|&(l, r)| (r, l)).collect();
                assert_eq!(*links, mirrored, "ranks {rank} and {peer} disagree");
            }
        }
        // Hosts cabled only to each other have no subnet every rank holds.
        let ring: Vec<Vec<Endpoint>> = (0..4u32)
            .map(|rank| {
                let previous = (rank + 3) % 4;
                vec![
                    endpoint("p0f0", &format!("10.{}.0.1/30", rank + 1)),
                    endpoint("p1f0", &format!("10.{}.0.2/30", previous + 1)),
                ]
            })
            .collect();
        let error = plan_shape_routes(&ring, 0, 1, Shape::Ring, Links::Shared)
            .unwrap_err()
            .to_string();
        assert!(error.contains("0 of 1 RoCE rails"), "{error}");
    }

    #[test]
    fn a_single_link_plan_matches_the_full_plan_for_every_pair() {
        let mut down = switch_with_partner_cables();
        down[2].retain(|e| SWITCH_DEVICES.contains(&e.name.as_str()));
        for endpoints in [switch_with_partner_cables(), down] {
            for links in [Links::All, Links::Shared] {
                for rank in 0..4 {
                    let full = plan_routes(&endpoints, rank, 2, links).unwrap();
                    for peer in 0..4 {
                        let link = plan_link(&endpoints, rank, peer, 2, links).unwrap();
                        assert_eq!(link, full[peer], "{links:?} rank {rank} to {peer}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_topology_marks_and_lists_each_peer_route() {
        let endpoints = switch_with_partner_cables();
        let fabric = Fabric {
            gid_index: 3,
            traffic_class: 0,
            endpoints: endpoints[1].clone(),
        };
        let topology = fabric
            .topology(&endpoints, 1, Shape::Mesh, Links::All)
            .unwrap();
        assert_eq!(topology.direct, vec![false, false, true, false]);
        assert_eq!(
            topology.peer_routes(),
            vec![
                "rank 0 roceP2p1s0f0,rocep1s0f0",
                "rank 2 direct roceP2p1s0f1,rocep1s0f1",
                "rank 3 roceP2p1s0f0,rocep1s0f0",
            ]
        );
        let switched = fabric
            .topology(&endpoints, 1, Shape::Mesh, Links::Shared)
            .unwrap();
        assert_eq!(switched.direct, vec![false; 4]);
        assert_eq!(switched.peer_routes()[1], "rank 2 roceP2p1s0f0,rocep1s0f0");
    }

    #[test]
    fn a_ring_of_sparks_pairs_devices_by_subnet_and_both_ends_agree() {
        // Each Spark's port 0 is cabled to the next Spark's port 1; each port is two functions on
        // their own point-to-point subnets.
        // The subnet between Sparks a and b (port 0 of a, port 1 of b) is 10.<link>.<function>.0/30.
        let link = |a: u32, b: u32| 10 * (a + 1) + b + 1;
        let spark = |me: u32| {
            let next = (me + 1) % 3;
            let previous = (me + 2) % 3;
            vec![
                endpoint("p0f0", &format!("10.{}.0.1/30", link(me, next))),
                endpoint("p0f1", &format!("10.{}.1.1/30", link(me, next))),
                endpoint("p1f0", &format!("10.{}.0.2/30", link(previous, me))),
                endpoint("p1f1", &format!("10.{}.1.2/30", link(previous, me))),
            ]
        };
        let endpoints: Vec<_> = (0..3).map(spark).collect();
        let plans: Vec<_> = (0..3)
            .map(|rank| plan_routes(&endpoints, rank, 2, Links::All).unwrap())
            .collect();
        for a in 0..3 {
            for b in 0..3 {
                if a == b {
                    continue;
                }
                let mirrored: Vec<_> = plans[b][a].iter().map(|&(l, r)| (r, l)).collect();
                assert_eq!(plans[a][b], mirrored, "ranks {a} and {b} disagree");
                for &(l, r) in &plans[a][b] {
                    assert!(same_link(
                        &endpoints[a][l as usize],
                        &endpoints[b][r as usize]
                    ));
                }
            }
        }
        // Rank 0 reaches rank 1 over its port 0 and rank 2 over its port 1.
        assert_eq!(plans[0][1], vec![(0, 2), (1, 3)]);
        assert_eq!(plans[0][2], vec![(2, 0), (3, 1)]);
        // With one function per port, only one rail reaches each neighbour.
        let single: Vec<Vec<Endpoint>> = endpoints
            .iter()
            .map(|e| vec![e[0].clone(), e[2].clone()])
            .collect();
        let error = plan_routes(&single, 0, 2, Links::All)
            .unwrap_err()
            .to_string();
        assert!(error.contains("1 of 2 RoCE rails"), "{error}");
    }

    #[test]
    fn variable_size_rings_route_only_neighbors_on_physical_and_switched_fabrics() {
        for world in [2, 3, 4, 5, 6, 8, 16] {
            for switched in [false, true] {
                let endpoints: Vec<Vec<Endpoint>> = (0..world)
                    .map(|rank| {
                        if switched {
                            return vec![
                                endpoint("f0", &format!("10.1.0.{}/24", rank + 1)),
                                endpoint("f1", &format!("10.2.0.{}/24", rank + 1)),
                            ];
                        }
                        let previous = (rank + world - 1) % world;
                        vec![
                            endpoint("p0f0", &format!("10.{}.0.1/30", rank + 1)),
                            endpoint("p0f1", &format!("10.{}.1.1/30", rank + 1)),
                            endpoint("p1f0", &format!("10.{}.0.2/30", previous + 1)),
                            endpoint("p1f1", &format!("10.{}.1.2/30", previous + 1)),
                        ]
                    })
                    .collect();
                if !switched && world > 3 {
                    assert!(plan_routes(&endpoints, 0, 2, Links::All).is_err());
                }
                let plans: Vec<_> = (0..world)
                    .map(|r| plan_shape_routes(&endpoints, r, 2, Shape::Ring, Links::All).unwrap())
                    .collect();
                for rank in 0..world {
                    assert!(plans[rank][rank].is_empty());
                    for peer in 0..world {
                        if peer == rank
                            || peer == (rank + 1) % world
                            || peer == (rank + world - 1) % world
                        {
                            continue;
                        }
                        assert!(plans[rank][peer].is_empty());
                    }
                    for peer in [(rank + 1) % world, (rank + world - 1) % world] {
                        assert_eq!(plans[rank][peer].len(), 2);
                        let reverse: Vec<_> =
                            plans[peer][rank].iter().map(|&(r, l)| (l, r)).collect();
                        assert_eq!(plans[rank][peer], reverse);
                    }
                }
            }
        }
    }

    #[test]
    fn ring_rejects_wrong_world_size_and_missing_neighbor_links() {
        assert!(plan_shape_routes(&[], 0, 2, Shape::Ring, Links::All).is_err());
        let endpoints = vec![vec![endpoint("f0", "10.0.0.1/30")]; 4];
        assert!(plan_shape_routes(&endpoints, 4, 2, Shape::Ring, Links::All).is_err());
        assert!(plan_shape_routes(&endpoints, 0, 2, Shape::Ring, Links::All).is_err());
    }

    #[test]
    fn a_line_connects_neighbors_only_and_needs_no_link_between_its_ends() {
        // Four hosts cabled 0-1, 1-2, 2-3 on distinct /30 subnets; the ends share no subnet.
        let endpoints = vec![
            vec![endpoint("a", "10.0.0.1/30")],
            vec![endpoint("b", "10.0.0.2/30"), endpoint("c", "10.0.1.1/30")],
            vec![endpoint("d", "10.0.1.2/30"), endpoint("e", "10.0.2.1/30")],
            vec![endpoint("f", "10.0.2.2/30")],
        ];
        let routes = plan_shape_routes(&endpoints, 0, 1, Shape::Chain, Links::All).unwrap();
        assert_eq!(routes, vec![vec![], vec![(0, 0)], vec![], vec![]]);
        let routes = plan_shape_routes(&endpoints, 3, 1, Shape::Chain, Links::All).unwrap();
        assert_eq!(routes, vec![vec![], vec![], vec![(0, 1)], vec![]]);
        let routes = plan_shape_routes(&endpoints, 1, 1, Shape::Chain, Links::All).unwrap();
        assert_eq!(routes, vec![vec![(0, 0)], vec![], vec![(1, 0)], vec![]]);
        // The same hosts as a ring need the missing 3-0 link.
        assert!(plan_shape_routes(&endpoints, 0, 1, Shape::Ring, Links::All).is_err());
    }

    #[test]
    fn a_topology_opens_every_device_and_caps_rails_at_the_smallest_rank() {
        let fabric = Fabric {
            gid_index: 3,
            traffic_class: 0,
            endpoints: vec![endpoint("a", "10.0.0.1/24"), endpoint("b", "10.0.1.1/24")],
        };
        let peer = vec![endpoint("c", "10.0.0.2/24")];
        let topology = fabric
            .topology(
                &[fabric.endpoints.clone(), peer],
                0,
                Shape::Mesh,
                Links::All,
            )
            .unwrap();
        assert_eq!(
            (
                topology.hcas.len(),
                topology.rails,
                topology.routes[1].clone()
            ),
            (2, 1, vec![(0, 0)])
        );
        assert!(
            fabric
                .topology(
                    &[fabric.endpoints.clone(), vec![]],
                    0,
                    Shape::Mesh,
                    Links::All
                )
                .is_err()
        );
    }
}
