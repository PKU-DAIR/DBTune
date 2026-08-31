"""Validate a DBTune knob JSON against the system variables a local mysqld binary supports.

Usage:
    python validate_knob_file.py [--mysqld ../mysql_build/bin/mysqld]
                                 [--knobs ./experiment/gen_knobs/mysql_all_8.0.json]
                                 [--out ./experiment/gen_knobs/mysql_all_8044_validated.json]

`mysqld --no-defaults --verbose --help` prints every supported variable (one per line in
the "Variables (--variable-name=value)" table, dash-separated). Knobs missing from that
list would make the server reject its config at restart, so they are dropped from the
output file with a report.
"""
import argparse
import json
import os
import re
import subprocess
import sys


def mysqld_variables(mysqld_path):
    proc = subprocess.run(
        [mysqld_path, '--no-defaults', '--verbose', '--help'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    lines = proc.stdout.splitlines()
    # The variables table starts after a line like:
    # "Variables (--variable-name=value)" followed by a separator row of dashes.
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith('Variables ('))
    except StopIteration:
        sys.exit('Could not find variables table in mysqld --verbose --help output.\n'
                 'stderr was:\n' + proc.stderr[-2000:])
    names = set()
    for line in lines[start + 1:]:
        m = re.match(r'^([a-z][a-z0-9-]*)\s', line)
        if not m:
            if names and not line.strip():
                break
            continue
        names.add(m.group(1).replace('-', '_'))
    return names


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('--mysqld', default=os.path.join(here, '..', 'mysql_build', 'bin', 'mysqld'))
    p.add_argument('--knobs', default=os.path.join(here, 'experiment', 'gen_knobs', 'mysql_all_8.0.json'))
    p.add_argument('--out', default=os.path.join(here, 'experiment', 'gen_knobs', 'mysql_all_8044_validated.json'))
    args = p.parse_args()

    supported = mysqld_variables(args.mysqld)
    print(f'mysqld reports {len(supported)} variables')

    with open(args.knobs) as f:
        knobs = json.load(f)

    kept, dropped = {}, []
    for name, spec in knobs.items():
        if name in supported:
            kept[name] = spec
        else:
            dropped.append(name)

    print(f'{args.knobs}: {len(knobs)} knobs -> kept {len(kept)}, dropped {len(dropped)}')
    for name in dropped:
        print(f'  DROPPED (unsupported by this mysqld): {name}')

    with open(args.out, 'w') as f:
        json.dump(kept, f, indent=4)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
