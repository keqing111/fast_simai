"""Small non-ring ns-3 topologies derived from all-to-all hardware profiles."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import HardwareProfile


@dataclass(frozen=True)
class TopologyLink:
    src: int
    dst: int
    bandwidth_gbps: float
    latency_ns: float

    def __post_init__(self) -> None:
        if self.src < 0 or self.dst < 0 or self.src == self.dst:
            raise ValueError("topology link endpoints are invalid")
        if self.bandwidth_gbps <= 0 or self.latency_ns < 0:
            raise ValueError("topology rates must be positive and latency non-negative")


@dataclass(frozen=True)
class Ns3Topology:
    gpu_ids: tuple[int, ...]
    nvswitch_ids: tuple[int, ...]
    tor_id: int | None
    links: tuple[TopologyLink, ...]
    gpus_per_server: int
    gpu_type: str = "H100"

    @property
    def node_count(self) -> int:
        return len(self.gpu_ids) + len(self.nvswitch_ids) + (self.tor_id is not None)

    def serialize(self) -> str:
        switch_count = 1 if self.tor_id is not None else 0
        first = (
            f"{self.node_count} {self.gpus_per_server} {len(self.nvswitch_ids)} "
            f"{switch_count} {len(self.links)} {self.gpu_type}"
        )
        special_nodes = [*self.nvswitch_ids]
        if self.tor_id is not None:
            special_nodes.append(self.tor_id)
        lines = [first, " ".join(str(node) for node in special_nodes)]
        for link in self.links:
            bandwidth = f"{link.bandwidth_gbps:.12g}Gbps"
            latency = f"{link.latency_ns / 1_000_000:.12g}ms"
            lines.append(f"{link.src} {link.dst} {bandwidth} {latency} 0")
        return "\n".join(lines) + "\n"


def build_topology(
    hardware: HardwareProfile,
    *,
    expected_num_devices: int | None = None,
) -> Ns3Topology:
    if expected_num_devices is not None and expected_num_devices != hardware.num_devices:
        raise ValueError("device count does not match the hardware profile")
    gpu_ids = tuple(range(hardware.num_devices))
    nvswitch_ids = tuple(
        range(hardware.num_devices, hardware.num_devices + hardware.num_servers)
    )
    tor_id = hardware.num_devices + hardware.num_servers if hardware.num_servers > 1 else None
    if hardware.intra.beta_ns_per_byte > 0:
        intra_bandwidth = 8.0 / hardware.intra.beta_ns_per_byte
    else:
        intra_bandwidth = hardware.link_bandwidth_gbps * hardware.gpus_per_server
    uplink_bandwidth = (
        hardware.link_bandwidth_gbps
        * hardware.gpus_per_server
        / hardware.oversubscription
    )
    links: list[TopologyLink] = []
    for gpu_id in gpu_ids:
        server_id = gpu_id // hardware.gpus_per_server
        links.append(
            TopologyLink(
                gpu_id,
                nvswitch_ids[server_id],
                intra_bandwidth,
                hardware.intra.alpha_ns,
            )
        )
    if tor_id is not None:
        for nvswitch_id in nvswitch_ids:
            links.append(
                TopologyLink(
                    nvswitch_id,
                    tor_id,
                    uplink_bandwidth,
                    hardware.inter.alpha_ns,
                )
            )
    return Ns3Topology(
        gpu_ids,
        nvswitch_ids,
        tor_id,
        tuple(links),
        hardware.gpus_per_server,
    )


def write_topology(
    hardware: HardwareProfile,
    path: Path,
    *,
    expected_num_devices: int | None = None,
) -> dict[str, Any]:
    topology = build_topology(
        hardware,
        expected_num_devices=expected_num_devices,
    )
    target = Path(path)
    content = topology.serialize()
    target.write_text(content, encoding="utf-8")
    return {
        "topology_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "node_count": topology.node_count,
        "link_count": len(topology.links),
        "gpu_ids": list(topology.gpu_ids),
        "nvswitch_ids": list(topology.nvswitch_ids),
        "tor_id": topology.tor_id,
    }
