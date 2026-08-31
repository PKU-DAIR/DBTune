#!/usr/bin/env python3
"""Derive the online-only knob file: keep knobs MySQL can change with SET GLOBAL (no restart).

    python scripts/make_online_knob_file.py experiment/gen_knobs/mysql_perf_8.0.json \
        experiment/gen_knobs/mysql_perf_8.0_online.json
"""
import json
import sys


# Dynamic by the manual, but not settable in practice: MySQL refuses to move
# innodb_commit_concurrency between 0 and non-zero at runtime (deprecated anyway).
DROP = ("innodb_commit_concurrency",)

# MySQL >= 8.0.30 replaces the static pair innodb_log_file_size x innodb_log_files_in_group
# with one DYNAMIC variable (SET GLOBAL resizes the redo log online). 100 MB default, tune up
# to 8 GB (the offline winners used 1.2-6.7 GB).
ADD = {"innodb_redo_log_capacity": {"default": 104857600, "dynamic": "Yes", "max": 8589934592,
                                    "min": 104857600, "scope": "Global", "type": "integer"}}


def is_dynamic(spec: dict) -> bool:
    return str(spec.get("dynamic", "")).lower() in ("yes", "true", "1")


def main(src: str, dst: str) -> None:
    knobs = json.load(open(src))
    online = {name: spec for name, spec in knobs.items() if is_dynamic(spec) and name not in DROP}
    online.update(ADD)
    with open(dst, "w") as f:
        json.dump(online, f, indent=4)
    dropped = sorted(set(knobs) - set(online))
    print("%d knobs -> %d online (+%d added: %s); dropped %d: %s"
          % (len(knobs), len(online), len(ADD), ", ".join(ADD), len(dropped), ", ".join(dropped)))


if __name__ == "__main__":
    main(*sys.argv[1:3])
