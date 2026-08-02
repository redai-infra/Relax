# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("off_contract", type=Path)
    parser.add_argument("on_contract", type=Path)
    args = parser.parse_args()

    off = json.loads(args.off_contract.read_text(encoding="utf-8"))
    on = json.loads(args.on_contract.read_text(encoding="utf-8"))
    errors = []
    if off.get("arm") != "off" or off.get("policy_enabled") is not False:
        errors.append("OFF contract does not identify a disabled policy arm")
    if on.get("arm") != "on" or on.get("policy_enabled") is not True:
        errors.append("ON contract does not identify an enabled policy arm")
    if off.get("comparison_fingerprint") != on.get("comparison_fingerprint"):
        errors.append("comparison fingerprints differ")
    if off.get("comparison_payload") != on.get("comparison_payload"):
        errors.append("comparison payloads differ")
    if off.get("instrumentation") != on.get("instrumentation"):
        errors.append("instrumentation contracts differ")

    hardware = {
        "off": off.get("gpu_inventory"),
        "on": on.get("gpu_inventory"),
    }
    if errors:
        raise SystemExit("; ".join(errors))
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "comparison_fingerprint": off["comparison_fingerprint"],
                "only_behavioral_difference": "RELAX_SYNC_INTENT_POLICY",
                "hardware_inventory": hardware,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
