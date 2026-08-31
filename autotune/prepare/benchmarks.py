"""Benchmark families: how a Workload's parameters become runner settings and a dataset.

This is the only place that knows benchmark option names. The Miyabi backend asks an
adapter for `runner_args()` ([database] ini keys) and `dataset()` (which snapshot to load).
"""
from __future__ import annotations

from dataclasses import dataclass

from .workload import Workload


@dataclass(frozen=True)
class Dataset:
    """A loaded database snapshot: `name` is the directory under mysql_build/, `builder`
    the loader family, `params` what the loader needs (e.g. rows per table)."""
    name: str
    builder: str
    params: dict


class Benchmark:
    family: str = "abstract"
    supported: bool = False

    def runner_args(self, w: Workload) -> dict:
        raise NotImplementedError

    def dataset(self, w: Workload) -> Dataset:
        raise NotImplementedError


class SysbenchOLTP(Benchmark):
    """sysbench oltp_read_write. Parameters: rows (per table), threads, zipf (0 = uniform),
    optional point_selects (statements per transaction; a code-level knob, kept for the
    existing S4 cell). Table count is fixed by the base ini (150)."""
    family = "sysbench"
    supported = True

    def runner_args(self, w: Workload) -> dict:
        args = {"thread_num": str(int(w.get("threads"))),
                "sysbench_table_size": str(int(w.get("rows")))}
        zipf = _get(w, "zipf", 0.7)
        if zipf > 0:
            args["sysbench_zipfian_exp"] = "%g" % zipf
        else:
            args["sysbench_rand_type"] = "uniform"
        point_selects = _get(w, "point_selects", 0)
        if point_selects:
            args["sysbench_extra_args"] = "--point-selects=%d" % point_selects
        return args

    def dataset(self, w: Workload) -> Dataset:
        rows = int(w.get("rows"))
        return Dataset("data_150x%dk" % (rows // 1000), "sysbench", {"table_size": rows})


class TPCC(Benchmark):
    """TPC-C. Parameters: warehouses, threads, write_ratio (NewOrder+Payment share), skew.
    The parameter mapping is defined here, but DBTune's TPC-C runner and dataset loader do
    not exist yet, so the backend refuses TPC-C tasks until they do."""
    family = "tpcc"
    supported = False

    def runner_args(self, w: Workload) -> dict:
        return {"workload": "tpcc", "thread_num": str(int(w.get("threads"))),
                "tpcc_warehouses": str(int(w.get("warehouses"))),
                "tpcc_write_ratio": "%g" % w.get("write_ratio"), "tpcc_skew": "%g" % w.get("skew")}

    def dataset(self, w: Workload) -> Dataset:
        wh = int(w.get("warehouses"))
        return Dataset("data_tpcc_w%d" % wh, "tpcc", {"warehouses": wh})


BENCHMARKS = {b.family: b for b in (SysbenchOLTP(), TPCC())}


def _get(w: Workload, name: str, default: float) -> float:
    return dict(w.params).get(name, default)
