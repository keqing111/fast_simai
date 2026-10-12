# SimAI/ns-3 All-to-All Baseline

This package prepares one traffic matrix, applies several open-loop schedulers to the same demand, runs each schedule through SimAI's ns-3 backend, and joins ns-3 FCT records back to schedule rows by `flow_id`.

The experiment assumptions, metric definitions, planned comparisons, and result placeholders are maintained in [REPORT.md](REPORT.md).

## Scope

Implemented schedulers:

- `direct-concurrent`: release every nonzero matrix cell at time zero.
- `spreadout`: cyclic one-to-one GPU stages.
- `roundrobin`: intra-server stages followed by directed inter-server stages.
- `fast`: the verified FAST pipeline2 port. For a one-server profile, where FAST has no inter-server permutation set, it reduces deterministically to local cyclic stages.

Implemented synthetic traffic:

- `balanced`
- FAST's uniform generator
- FAST's truncated-Zipf generator (`0 < alpha < 1`)

The current integration is deliberately open loop. FAST computes the complete schedule before ns-3 starts; ns-3 congestion does not feed back into FAST. A scheduled `start_ns` is an offset from the all-to-all collective release epoch, not a dependency barrier. The analyzer reports stage spill so that violations of FAST's predicted slots remain visible.

Not included yet: NCCL-like scheduling, ring/hybrid-cube routing policies, real traces, compute overlap, runtime device jitter, and online rescheduling.

## Build

From the SimAI repository root:

```bash
git submodule update --init SimCCL ns-3-alibabacloud
SIMAI_NCCL_VERSION=v2.30 ./scripts/build.sh -c ns3
```

The external-schedule adapter is supported only with the pinned SimCCL v2.30 implementation. The resulting executable is `bin/SimAI_simulator`.

Run Python commands from this directory with the source tree on `PYTHONPATH`:

```bash
cd astra-sim-alibabacloud/experiments/alltoall
export PYTHONPATH=src
```

## Prepare, run, and analyze

Prepare the bundled two-server balanced scenario:

```bash
python3 -m simai_a2a prepare \
  --config configs/smoke.json \
  --output results/balanced-smoke
```

This writes one matrix, one generated topology, a zero-compute MICRO workload, and four scheduler directories. All schedules reference the same matrix hash.

Run one scheduler:

```bash
python3 -m simai_a2a run \
  --scenario results/balanced-smoke \
  --algorithm fast \
  --topology results/balanced-smoke/topology.txt \
  --binary ../../../bin/SimAI_simulator \
  --timeout 300
```

Every invocation creates a new `runs/<algorithm>/attempt-NNNN` directory. It contains local copies of the workload, topology, and schedule; a rendered ns-3 config; stdout/stderr; FCT/PFC files; `run_manifest.json`; and, on strict completion, `summary.json`. Exit status is nonzero for timeout, simulator failure, or incomplete/mismatched FCT output.

Re-run analysis for an existing attempt:

```bash
python3 -m simai_a2a analyze \
  --scenario results/balanced-smoke \
  --algorithm fast \
  --attempt results/balanced-smoke/runs/fast/attempt-0001
```

## Artifact contracts

`matrix/matrix.npy` is signed `int64`, square, nonnegative, and has a zero diagonal. `matrix_manifest.json` records its SHA-256, logical bytes, generator parameters, skew metrics, and code revisions.

Each scheduler directory contains:

- `schedule.csv`: no header; exactly `src,dst,size_bytes,start_ns`.
- `schedule_metadata.csv`: `flow_id,phase,stage_id`; `flow_id` is the zero-based schedule row.
- `schedule_manifest.json`: input hashes, output hashes, flow/byte counts, prediction fields, and provenance.

External ns-3 FCT rows have nine whitespace-separated columns. The first eight retain the native ns-3 fields and the ninth is the decimal schedule `flow_id`. The analyzer requires exactly one matching row for every scheduled flow and exact agreement on `(src,dst,size_bytes)`.

`summary.json` contains actual and predicted makespan separately, logical and physical bytes, algorithm bandwidth, FCT mean/p50/p95/p99/max, PFC count, fidelity ratio, and FAST slot-boundary violations.

## Verification

Unit and integration suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Full 3-topology × 4-scheduler ns-3 smoke matrix:

```bash
SIMAI_NS3_BINARY="$(realpath ../../../bin/SimAI_simulator)" \
  PYTHONPATH=src python3 -m unittest tests.test_e2e_smoke -v
```

Standalone strict C++ schedule parser test, from the repository root:

```bash
g++ -std=c++11 -I SimCCL/src/mock/v2.30 \
  astra-sim-alibabacloud/experiments/alltoall/tests/cpp/external_schedule_test.cc \
  SimCCL/src/mock/v2.30/ExternalSchedule.cc \
  -o /tmp/simai-external-schedule-test
/tmp/simai-external-schedule-test
```
