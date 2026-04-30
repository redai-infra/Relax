#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import socket
import time
from collections import Counter, defaultdict
from pathlib import Path


TCP_STATES = {
    "01": "ESTABLISHED",
    "02": "SYN_SENT",
    "03": "SYN_RECV",
    "04": "FIN_WAIT1",
    "05": "FIN_WAIT2",
    "06": "TIME_WAIT",
    "07": "CLOSE",
    "08": "CLOSE_WAIT",
    "09": "LAST_ACK",
    "0A": "LISTEN",
    "0B": "CLOSING",
}


def _read_port_range() -> tuple[int, int]:
    text = Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()
    return int(text[0]), int(text[1])


def _hex_to_ipv4(hex_addr: str) -> str:
    return ".".join(str(int(hex_addr[i : i + 2], 16)) for i in (6, 4, 2, 0))


def _inode_to_proc() -> dict[str, set[tuple[str, str]]]:
    mapping: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for pid in filter(str.isdigit, os.listdir("/proc")):
        fd_dir = Path("/proc") / pid / "fd"
        try:
            comm = (Path("/proc") / pid / "comm").read_text().strip()
            for fd in fd_dir.iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith("socket:["):
                    mapping[target[8:-1]].add((pid, comm))
        except OSError:
            continue
    return mapping


def _parse_tcp(path: str, port_lo: int, port_hi: int, host: str | None):
    host_hex = None
    if host:
        host_hex = "".join(f"{int(part):02X}" for part in host.split(".")[::-1])

    state_counts = Counter()
    ep_state_counts = Counter()
    ep_ports = set()
    timewait_remote = Counter()
    listen_inodes: Counter[str] = Counter()
    established_inodes: Counter[str] = Counter()

    for line in Path(path).read_text().splitlines()[1:]:
        parts = line.split()
        local, remote, state, inode = parts[1], parts[2], parts[3], parts[9]
        local_addr, local_port_hex = local.rsplit(":", 1)
        remote_addr, remote_port_hex = remote.rsplit(":", 1)
        local_port = int(local_port_hex, 16)
        remote_port = int(remote_port_hex, 16)
        state_name = TCP_STATES.get(state, state)

        if host_hex is not None and local_addr != host_hex:
            continue

        state_counts[state_name] += 1
        if port_lo <= local_port <= port_hi:
            ep_ports.add(local_port)
            ep_state_counts[state_name] += 1

        if state_name == "TIME_WAIT":
            remote_ip = _hex_to_ipv4(remote_addr) if len(remote_addr) == 8 else remote_addr
            timewait_remote[(remote_ip, remote_port)] += 1
        elif state_name == "LISTEN":
            listen_inodes[inode] += 1
        elif state_name == "ESTABLISHED":
            established_inodes[inode] += 1

    return state_counts, ep_state_counts, ep_ports, timewait_remote, listen_inodes, established_inodes


def _proc_counter(inode_counts: Counter[str], inode_map: dict[str, set[tuple[str, str]]]) -> Counter[str]:
    result = Counter()
    for inode, count in inode_counts.items():
        procs = inode_map.get(inode) or {("<none>", "<none>")}
        for pid, comm in procs:
            result[f"{comm} pid={pid}"] += count
    return result


def _listen_owner_by_port(inode_map: dict[str, set[tuple[str, str]]]) -> dict[int, str]:
    owners: dict[int, str] = {}
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        parts = line.split()
        local, state, inode = parts[1], parts[3], parts[9]
        if TCP_STATES.get(state, state) != "LISTEN":
            continue
        local_port = int(local.rsplit(":", 1)[1], 16)
        procs = sorted(inode_map.get(inode) or {("<none>", "<none>")})
        owners[local_port] = ",".join(f"{comm} pid={pid}" for pid, comm in procs)
    return owners


def print_snapshot(host: str | None = None, top_n: int = 10) -> None:
    port_lo, port_hi = _read_port_range()
    inode_map = _inode_to_proc()
    listen_owners = _listen_owner_by_port(inode_map)
    state_counts, ep_state_counts, ep_ports, timewait_remote, listen_inodes, established_inodes = _parse_tcp(
        "/proc/net/tcp", port_lo, port_hi, host
    )

    range_size = port_hi - port_lo + 1
    print(f"ts={time.strftime('%Y-%m-%d %H:%M:%S')} host={host or '*'} port_range={port_lo}-{port_hi}")
    print(
        f"ephemeral_used_unique={len(ep_ports)}/{range_size} free_estimate={range_size - len(ep_ports)} "
        f"state_counts={dict(state_counts)} ep_state_counts={dict(ep_state_counts)}"
    )

    print("top_time_wait_remote:")
    for (remote_ip, remote_port), count in timewait_remote.most_common(top_n):
        owner = listen_owners.get(remote_port, "<no-listener-now>")
        print(f"  {count:7d} {remote_ip}:{remote_port} owner={owner}")

    print("top_listen_owners:")
    for proc, count in _proc_counter(listen_inodes, inode_map).most_common(top_n):
        print(f"  {count:7d} {proc}")

    print("top_established_owners:")
    for proc, count in _proc_counter(established_inodes, inode_map).most_common(top_n):
        print(f"  {count:7d} {proc}")


def main() -> None:
    host = os.environ.get("PORT_WATCH_HOST")
    top_n = int(os.environ.get("PORT_WATCH_TOP_N", "10"))
    interval = float(os.environ.get("PORT_WATCH_INTERVAL", "0"))
    while True:
        print_snapshot(host=host, top_n=top_n)
        if interval <= 0:
            break
        print("-" * 80, flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    main()
