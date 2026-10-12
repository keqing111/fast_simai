# SimAI/NS-3 All-to-All Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible all-to-all baseline that generates synthetic matrices, emits schedules for four schedulers, executes them through SimAI's ns-3 backend, and reports flow-, collective-, and FAST-stage metrics.

**Architecture:** A self-contained Python package under `astra-sim-alibabacloud/experiments/alltoall/` owns matrix generation, scheduler APIs, artifact schemas, experiment execution, and analysis. SimAI receives a four-column schedule through a narrowly scoped external-schedule adapter in the pinned SimCCL v2.30 path; every scheduled flow is preloaded and released at an offset relative to the collective start. The Python and C++ halves meet only through versioned files and are tested separately before the ns-3 vertical smoke test.

**Tech Stack:** Python 3.10+, NumPy, standard-library `unittest`, C++11, SimAI/Astra-Sim, SimCCL v2.30, ns-3.

**Spec:** `docs/superpowers/specs/2026-10-12-alltoall-baseline-design.md`

## Global Constraints

- Treat `/home/julius/fast/SimAI` as the canonical implementation; `/home/julius/simAI/SimAI` is read-only reference material.
- Do not create Git commits or push anything during this implementation; use test output plus `git diff --check` as task checkpoints.
- Initialize `SimCCL` and `ns-3-alibabacloud` at the exact SHAs already recorded by this repository; do not update them to branch tips.
- External scheduling is supported for the repository default `SIMAI_NCCL_VERSION=v2.30`; reject another version explicitly in the experiment runner.
- `schedule.csv` has no header and exactly four columns: `src,dst,size_bytes,start_ns`; `flow_id` is its zero-based record number.
- `start_ns` is an offset from the collective release epoch, not an absolute ns-3 time and not a dependency barrier.
- The first implementation contains only `direct-concurrent`, `spreadout`, `roundrobin`, and faithful `fast` pipeline2.
- The first implementation contains only balanced, FAST uniform, and FAST truncated-Zipf traffic; Zipf matrices are not normalized across alpha values.
- Do not implement NCCL-like, ring/hybrid-cube routing, real traces, compute overlap, device jitter, or runtime FAST feedback.
- Runtime code and scripts must not contain personal absolute paths; all paths come from CLI arguments, config files, or repository-relative discovery.
- Use MICRO workload with zero compute and use ns-3 as the only accepted end-to-end backend.

## Review Focus

- Malformed, overflowing, header-bearing, or out-of-range schedule rows must fail the run before simulation rather than be skipped; pinned by Task 6 parser tests.
- Repeated `(src,dst,size,start)` flows must remain separately attributable by record-number `flow_id`; pinned by Tasks 6 and 8.
- A delayed flow must keep its owning `Sys` alive until its send-completion callback and FCT write; pinned by Task 7 lifecycle test.
- FAST may emit more physical bytes than the input's logical bytes, but logical demand must remain the matrix sum and inter-server demand must be conserved; pinned by Task 5.
- Zipf alpha near one, zero-valued uniform cells, invalid matrices, and integer overflow during scaling must have defined behavior; pinned by Tasks 2 and 3.

---

### Task 1: Python package and typed artifact model

**Files:**
- Create: `astra-sim-alibabacloud/experiments/alltoall/pyproject.toml`
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/__init__.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/model.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/artifacts.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tests/test_model.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tests/test_artifacts.py`

**Interfaces:**
- Consumes: NumPy arrays and JSON-compatible dictionaries.
- Produces: `LinkModel`, `HardwareProfile`, `TrafficMatrix`, `ScheduledFlow`, `Schedule`, `AllToAllScheduler`; `write_matrix_artifact(...)`, `read_matrix_artifact(...)`, `write_schedule_artifact(...)`, and `read_schedule_artifact(...)`.

- [ ] **Step 1: Write failing model validation tests**

  In `test_model.py`, assert that a valid `4x4 int64` zero-diagonal matrix is accepted; non-square, floating, negative, and nonzero-diagonal matrices raise `ValueError`; `num_servers * gpus_per_server` must equal the matrix dimension; and a `ScheduledFlow` rejects `src == dst`, zero bytes, negative time, and endpoints outside the hardware profile.

- [ ] **Step 2: Run the model tests and observe the expected import failure**

  Run: `cd astra-sim-alibabacloud/experiments/alltoall && python3 -m unittest tests.test_model -v`

  Expected: FAIL because `simai_a2a.model` does not exist.

- [ ] **Step 3: Implement the dataclasses and scheduler protocol**

  `LinkModel.duration_ns(size_bytes: int) -> int` returns rounded integer ns from `alpha_ns + beta_ns_per_byte * size_bytes`. `TrafficMatrix.logical_bytes` is the Python-int sum of all cells. `Schedule` stores `algorithm`, ordered `flows`, `predicted_makespan_ns`, and JSON-compatible `metadata`; it validates that `flow_id == list index` after deterministic sorting by `(start_ns, phase, stage_id, src, dst, size_bytes)`.

- [ ] **Step 4: Write and run artifact round-trip tests**

  `test_artifacts.py` must assert `.npy` remains `int64`, hashes survive a write/read round trip, `schedule.csv` has exactly four integer columns and no header, `schedule_metadata.csv` joins every row by `flow_id`, and tampering with either file raises a hash/shape error. Run `python3 -m unittest tests.test_model tests.test_artifacts -v`; expected: PASS.

- [ ] **Step 5: Record the no-commit checkpoint**

  Run: `git diff --check && git status --short`; expected: no whitespace errors and only intentional all-to-all/doc changes plus initialized-submodule state from later tasks.

### Task 2: Matrix skew metrics

**Files:**
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/metrics.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tests/test_metrics.py`

**Interfaces:**
- Consumes: `TrafficMatrix` and `gpus_per_server: int`.
- Produces: `gini(values: np.ndarray) -> float` and `matrix_skew_metrics(matrix: TrafficMatrix, gpus_per_server: int) -> dict[str, float]`.

- [ ] **Step 1: Write failing metric tests**

  Assert `gini([0, 0, 0]) == 0`, `gini([1, 1, 1]) == 0`, and `gini([0, 0, 3]) == 2/3`. For a balanced matrix, assert cell/source/destination/server-pair Gini are zero and source/destination bottleneck ratios are one. Assert server-pair aggregation rejects a dimension not divisible by `gpus_per_server`.

- [ ] **Step 2: Run the metric tests and verify failure**

  Run: `python3 -m unittest tests.test_metrics -v`; expected: FAIL because `simai_a2a.metrics` does not exist.

- [ ] **Step 3: Implement metrics with explicit empty/all-zero semantics**

  Exclude the matrix diagonal from cell Gini, use full row/column sums for source/destination metrics, and use only off-diagonal directed server pairs for server-pair Gini. Return `0.0` for all-zero Gini and bottleneck ratio, and convert NumPy sums to Python integers before division.

- [ ] **Step 4: Run the metric and package tests**

  Run: `python3 -m unittest discover -s tests -v`; expected: all Task 1–2 tests PASS.

- [ ] **Step 5: Record the no-commit checkpoint**

  Run: `git diff --check` and save the passing test command/output in the task log used in the final handoff.

### Task 3: Reproducible synthetic traffic generators

**Files:**
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/traffic.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tests/test_traffic.py`

**Interfaces:**
- Consumes: generator name, `num_devices`, `seed`, and generator-specific parameters.
- Produces: `balanced_matrix(num_devices: int, pair_bytes: int) -> TrafficMatrix`, `fast_uniform_matrix(num_devices: int, bytes_per_unit: int, seed: int) -> TrafficMatrix`, `fast_zipf_matrix(num_devices: int, alpha: float, bytes_per_unit: int, seed: int) -> TrafficMatrix`, and `generate_matrix(name: str, **kwargs) -> TrafficMatrix`.

- [ ] **Step 1: Write failing generator tests**

  Assert a three-device balanced matrix with `pair_bytes=4` has six off-diagonal cells and `logical_bytes == 24`; same seeds produce byte-identical uniform/Zipf matrices while different seeds do not; uniform weights are in `[0, 9999]`; diagonals are zero; and `zipf_inverse_cdf_fast(p, 10000, 0.9)` returns `[1, 18, 236, 1830, 9999]` for `p=[0,.25,.5,.75,1]`.

- [ ] **Step 2: Add failing boundary tests**

  Assert `num_devices < 2`, `bytes_per_unit <= 0`, `alpha <= 0`, `alpha >= 1`, and any multiplication exceeding signed `int64` raise `ValueError` rather than wrap. Include alpha `0.999` as a valid case and retain sampled zero cells in uniform traffic.

- [ ] **Step 3: Run the traffic tests and verify failure**

  Run: `python3 -m unittest tests.test_traffic -v`; expected: FAIL because the generator functions do not exist.

- [ ] **Step 4: Implement the three generators**

  Use `numpy.random.Generator(PCG64(seed))`; implement the Newton iteration from FAST `simulation/test.cpp::zipf_inverse_cdf_fast` with `N=10000` and tolerance `0.01`; sample one independent probability per matrix cell, multiply the integer weight by `bytes_per_unit`, check overflow, then clear the diagonal. Store generator parameters and seed in `TrafficMatrix.metadata`.

- [ ] **Step 5: Run all Python tests and checkpoint**

  Run: `python3 -m unittest discover -s tests -v && git diff --check`; expected: all Task 1–3 tests PASS and no whitespace errors.

### Task 4: Direct, SpreadOut, and round-robin schedulers

**Files:**
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/schedulers/__init__.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/schedulers/direct.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/schedulers/spreadout.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/schedulers/roundrobin.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tests/test_baseline_schedulers.py`

**Interfaces:**
- Consumes: `TrafficMatrix`, `HardwareProfile`, and explicit seed through `AllToAllScheduler.schedule(...)`.
- Produces: `DirectConcurrentScheduler`, `SpreadOutScheduler`, `RoundRobinScheduler`, and `get_scheduler(name: str) -> AllToAllScheduler`.

- [ ] **Step 1: Write failing direct and registry tests**

  Assert `direct-concurrent` emits exactly one start-zero flow per nonzero off-diagonal cell, preserves each `(src,dst,size)`, sets phase `direct`/stage `0`, and reports physical bytes equal to logical bytes. Assert unknown scheduler names raise `ValueError` listing the four accepted names.

- [ ] **Step 2: Write failing SpreadOut tests**

  For four devices, assert stages follow cyclic offsets `1,2,3`, every original nonzero cell appears exactly once, all flows in a stage share a start, starts are cumulative maxima of the appropriate intra/inter `LinkModel.duration_ns`, and output is independent of seed.

- [ ] **Step 3: Write failing round-robin tests**

  For `2 servers x 2 GPUs`, assert intra stages precede inter-server stages, each stage is one-to-one at both senders and receivers, every matrix cell appears exactly once, bytes are conserved, and two invocations produce identical schedules.

- [ ] **Step 4: Implement all three schedulers and run their tests**

  Implement native cyclic traversal for SpreadOut without the old experiment's shuffled stage order. Implement round-robin as intra-server cyclic GPU offsets followed by directed server offsets and cyclic destination-GPU offsets. Run `python3 -m unittest tests.test_baseline_schedulers -v`; expected: PASS.

- [ ] **Step 5: Run regression tests and checkpoint**

  Run: `python3 -m unittest discover -s tests -v && git diff --check`; expected: all Task 1–4 tests PASS.

### Task 5: Faithful FAST pipeline2 scheduler

**Files:**
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/schedulers/fast_port.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/schedulers/fast.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tests/test_fast_scheduler.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tests/test_fast_crosscheck.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tests/fixtures/fast_cpp/`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tools/refresh_fast_crosscheck.py`

**Interfaces:**
- Consumes: Task 1 models and the verified implementation currently at `../poc_simai/fast_decomp_py/fast_decomp.py` during the one-time import.
- Produces: `FastScheduler(scaling: bool = False)` implementing `AllToAllScheduler`; internal `run_fast_full(...)` and `emit_schedule(...)` remain independently cross-checkable.

- [ ] **Step 1: Import the faithful port and native fixtures without personal paths**

  Copy the verified Python implementation into `fast_port.py`, remove absolute-path documentation, and copy the 13 `.mat`/`.dump` fixture pairs from the existing cross-check workspace into `tests/fixtures/fast_cpp/`. `refresh_fast_crosscheck.py` must accept `--cpp-driver` and `--work-dir`; it must not contain a default absolute path.

- [ ] **Step 2: Write the failing 13-case cross-check test**

  Port the record parser and case table into `test_fast_crosscheck.py`. Assert exact integer structures and matrices, identical permutation-stage counts/maps, and float equality within `rel_tol=abs_tol=1e-12` for prediction/bound/baseline/speedup. The test output must name all 13 cases.

- [ ] **Step 3: Write failing schedule-semantic tests**

  In `test_fast_scheduler.py`, assert hardware ns fields are converted to the port's `us` and `us/MB` without changing predictions; phases are one of `balance/intrinsic/inter/dispatch`; pipeline2 overlap starts match `balance -> (inter0 || intrinsic) -> (inter1 || dispatch0) -> ... -> final dispatch`; flow IDs are deterministic; and inter-server emitted bytes equal the original off-diagonal server matrix.

- [ ] **Step 4: Implement the FAST adapter and detailed metadata**

  Keep decomposition code behavior unchanged. Extend only schedule emission to attach phase/stage metadata alongside the legacy `(src,dst,bytes,start_us)` tuples, round starts once to integer ns at the adapter boundary, set logical bytes from the input matrix, physical bytes from emitted flows, and expose predicted makespan/bound/stage starts in `Schedule.metadata`.

- [ ] **Step 5: Run faithful and full Python verification**

  Run: `python3 -m unittest tests.test_fast_crosscheck tests.test_fast_scheduler -v && python3 -m unittest discover -s tests -v && git diff --check`.

  Expected: 13/13 native fixtures PASS, all scheduler tests PASS, and no whitespace errors.

### Task 6: Scenario CLI and versioned artifacts

**Files:**
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/cli.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/__main__.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/scenario.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/configs/hardware/h100_2s2g_100g.json`
- Create: `astra-sim-alibabacloud/experiments/alltoall/configs/smoke.json`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tests/test_cli.py`

**Interfaces:**
- Consumes: JSON scenario/hardware files and Tasks 1–5 APIs.
- Produces: `python3 -m simai_a2a generate`, `schedule`, and `prepare`; scenario directories containing matrix, manifests, four schedules, schedule metadata, and a zero-compute MICRO workload.

- [ ] **Step 1: Write failing CLI tests using a temporary directory**

  Assert `generate` writes one matrix and manifest; `schedule --algorithm` writes one schedule set; and `prepare` caches one matrix then creates all four scheduler outputs referencing the same `matrix_sha256`. Assert unknown keys, missing profile fields, and output-directory reuse without `--new-attempt` fail with nonzero exit status.

- [ ] **Step 2: Run CLI tests and verify failure**

  Run: `python3 -m unittest tests.test_cli -v`; expected: FAIL because the CLI is not implemented.

- [ ] **Step 3: Implement CLI/config parsing and manifests**

  Use schema version `1`; record generator parameters, all skew metrics, input/profile/schedule SHA-256 values, scheduler wall time, logical/physical bytes, prediction fields, Python/NumPy versions, and repository/submodule SHAs when available. A dirty or unavailable SHA is recorded explicitly and never causes an artifact-generation failure.

- [ ] **Step 4: Implement the MICRO workload artifact**

  `prepare` writes one all-to-all layer with header `MICRO model_parallel_NPU_group: 1 ep: <N> all_gpus: <N>` and zero compute columns. The declared collective size is the maximum source row sum and is used only to enter SimAI's all-to-all path; the external schedule remains the source of actual flow sizes.

- [ ] **Step 5: Run all Python tests and deterministic artifact check**

  Run `python3 -m unittest discover -s tests -v`, then run `prepare` twice into two temporary directories and compare matrix/schedule SHA values. Expected: tests PASS and corresponding hashes match byte-for-byte.

### Task 7: Strict C++ schedule parser and SimCCL flow construction

**Files:**
- Create after submodule initialization: `SimCCL/src/mock/v2.30/ExternalSchedule.h`
- Create after submodule initialization: `SimCCL/src/mock/v2.30/ExternalSchedule.cc`
- Modify after submodule initialization: `SimCCL/src/mock/v2.30/MockNcclChannel.h`
- Modify after submodule initialization: `SimCCL/src/mock/v2.30/MockNcclGroup.h`
- Modify after submodule initialization: `SimCCL/src/mock/v2.30/MockNcclGroup.cc`
- Modify: `astra-sim-alibabacloud/astra-sim/network_frontend/ns3/AstraSimNetwork.cc`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tests/cpp/external_schedule_test.cc`

**Interfaces:**
- Consumes: four-column `schedule.csv`, group rank count, and `-s <path>` from the ns-3 executable.
- Produces: `ExternalSchedule::load(path, num_ranks) -> std::vector<ExternalScheduleFlow>` and sender-owned `SingleFlow` records with `external_schedule=true`, row-index `flow_id`, and relative `start_ns`.

- [ ] **Step 1: Initialize and verify only the required pinned submodules**

  Run `git submodule update --init SimCCL ns-3-alibabacloud`, then assert `git -C SimCCL rev-parse HEAD` is `fd7cd57d16f9bd42e3ccb70911c977e18ec294b9` and `git -C ns-3-alibabacloud rev-parse HEAD` is `3e0c7c1bfbbe9f77890ddcf5e5b9c79fc6dd7437`.

- [ ] **Step 2: Write and run the failing standalone parser test**

  Test valid rows including repeated identical tuples; assert flow IDs `0,1,...`; and assert empty files, headers, missing/extra columns, whitespace-only tokens, `src == dst`, zero bytes, negative values, numeric overflow, and endpoints outside `[0,num_ranks)` throw with the line number. Compile/run with `g++ -std=c++11` against only `ExternalSchedule.cc`; expected: compile failure before implementation.

- [ ] **Step 3: Implement the strict parser and external flow fields**

  Add `uint64_t start_ns` and `bool external_schedule` to `SingleFlow`, defaulting to `0/false` so native collectives remain unchanged. Parse once per configured path, fail fast by exception, assign records across existing channels with unique `(channel_id,chunk_id)` tags, and register each external flow only on its sender rank.

- [ ] **Step 4: Add the ns-3 `-s` entry point**

  Extend argument parsing and help in `AstraSimNetwork.cc`. When `-s` is supplied, require default SimCCL v2.30 and configure `MockNcclGroup`; when absent, preserve the native path byte-for-byte. Do not migrate the old NCCL-like changes.

- [ ] **Step 5: Run parser and compile checks**

  Run the standalone parser test, then `SIMAI_NCCL_VERSION=v2.30 ./astra-sim-alibabacloud/build.sh -c ns3`. Expected: parser PASS and the ns-3 executable builds successfully.

### Task 8: Relative release times, MICRO lifecycle, and flow identity

**Files:**
- Modify: `astra-sim-alibabacloud/astra-sim/system/collective/NcclFlowModel.cc`
- Modify: `astra-sim-alibabacloud/astra-sim/system/Sys.hh`
- Modify: `astra-sim-alibabacloud/astra-sim/system/Sys.cc`
- Modify: `astra-sim-alibabacloud/astra-sim/workload/Workload.cc`
- Modify: `astra-sim-alibabacloud/astra-sim/workload/Layer.cc`
- Modify: `astra-sim-alibabacloud/astra-sim/network_frontend/ns3/entry.h`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tests/test_ns3_adapter.py`

**Interfaces:**
- Consumes: Task 7 external `SingleFlow` records.
- Produces: all parentless external flows released at `collective_epoch + start_ns`; FCT rows with a ninth decimal `flow_id` field; clean process exit only after all scheduled sends finish.

- [ ] **Step 1: Write a failing ns-3 adapter integration test**

  Generate a two-flow schedule with identical `(src,dst,size)` and starts separated by `50_000 ns`. Assert the executable exits without polling/killing, writes exactly two FCT rows, reports distinct flow IDs `0` and `1`, preserves a start-time difference of `50_000 ns` within one ns-3 tick, and writes both rows before process exit.

- [ ] **Step 2: Make external flow release independent of native chunk gating**

  In `NcclFlowModel`, only flows marked `external_schedule` bypass native `chunk_id == 0` admission. Give their packet construction a safe sender fallback for empty `prev`, and call `front_end_sim_send` with the flow's relative `start_ns` delay. Leave native flow behavior unchanged.

- [ ] **Step 3: Prevent premature `Sys` destruction**

  Add `Sys::has_active_sends() const`, defined as any true value in `is_there_pending_sends`, and require it to be false in both destruction checks in `call_events()` and the PacketSent handler. Do not use global ns-3 event-count polling, because monitor events may outlive the collective.

- [ ] **Step 4: Fix MICRO's one-shot issue/report path and emit flow IDs**

  Set MICRO's state to `Weight_Gradient` before issuing once, return directly to `check_for_sim_end()` when already waiting, parse `model_parallel_NPU_group/ep/all_gpus` from the MICRO header, guard reporting against absent `pp` by treating it as one, and append `flowTag.current_flow_id` to external FCT rows.

- [ ] **Step 5: Rebuild and run native-regression plus adapter tests**

  Run `SIMAI_NCCL_VERSION=v2.30 ./astra-sim-alibabacloud/build.sh -c ns3`, run a pre-existing native MICRO all-reduce smoke without `-s`, then run `python3 -m unittest tests.test_ns3_adapter -v`. Expected: native smoke still exits, adapter test PASS, and two FCT records map to two flow IDs.

### Task 9: Experiment runner and analyzer

**Files:**
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/runner.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/analysis.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/configs/ns3.conf`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tests/test_analysis.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tests/test_runner.py`

**Interfaces:**
- Consumes: prepared scenario directories, topology path, ns-3 binary path, schedule and metadata, FCT/PFC outputs.
- Produces: `python3 -m simai_a2a run`, `analyze`, per-attempt `run_manifest.json`, `summary.json`, and nonzero process status for incomplete runs.

- [ ] **Step 1: Write failing analyzer tests with synthetic FCT/PFC fixtures**

  Assert one-to-one join by ninth-column flow ID; exact `(src,dst,size)` agreement; collective epoch inferred from `actual_start_ns - scheduled_start_ns`; makespan, logical/physical bytes, algorithm bandwidth, FCT mean/p50/p95/p99/max, and PFC count; and rejection of missing, duplicate, unknown, or mismatched flow IDs.

- [ ] **Step 2: Write failing FAST stage-violation tests**

  Build a synthetic balance/slot timeline and assert `max(0, actual_previous_slot_end - next_scheduled_slot_start)` per boundary, plus count, fraction, maximum, and sum. Assert predicted and actual makespan are separate fields and fidelity is `actual/predicted` only when prediction is positive.

- [ ] **Step 3: Write failing runner tests with a fake executable**

  Assert argv uses `-n/-c/-w/-s`, all run files are local to a unique attempt directory, success requires exit code zero and exact FCT cardinality/content, timeout terminates the child and records `timeout`, and rerun creates a new attempt rather than overwriting. No test may rely on a hardcoded binary path.

- [ ] **Step 4: Implement runner, analyzer, and CLI subcommands**

  Render a relative-path ns-3 config in each attempt directory, capture stdout/stderr and elapsed wall time, never treat file polling as completion, and write manifests atomically. Add `run` and `analyze` to Task 6's CLI.

- [ ] **Step 5: Run the full Python suite**

  Run: `python3 -m unittest discover -s tests -v && git diff --check`; expected: all unit/fake-runner/adapter tests PASS.

### Task 10: Topology profiles and end-to-end ns-3 smoke matrix

**Files:**
- Create: `astra-sim-alibabacloud/experiments/alltoall/src/simai_a2a/topology.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/configs/hardware/h100_1s4g_nvlink.json`
- Create: `astra-sim-alibabacloud/experiments/alltoall/configs/hardware/h100_2s2g_100g_2to1.json`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tests/test_topology.py`
- Create: `astra-sim-alibabacloud/experiments/alltoall/README.md`
- Create: `astra-sim-alibabacloud/experiments/alltoall/tests/test_e2e_smoke.py`

**Interfaces:**
- Consumes: hardware profiles and Tasks 6–9 CLI.
- Produces: validated non-ring ns-3 topology files and a twelve-run smoke result covering three hardware/traffic scenarios by four schedulers.

- [ ] **Step 1: Write failing topology tests**

  For `1 server x 4 GPUs` and `2 servers x 2 GPUs`, assert GPU IDs, per-server NVSwitch IDs, optional ToR ID, header node/link counts, no GPU-as-transit link, profile bandwidth/latency propagation, and distinct nonblocking versus `2:1` oversubscribed uplink capacity. Reject inconsistent device counts and nonpositive rates.

- [ ] **Step 2: Implement topology generation and validate with ns-3 startup**

  Generate topology files into scenario artifacts rather than the source tree. Start ns-3 with the single-server, nonblocking multi-server, and oversubscribed multi-server topologies using an empty diagnostic workload only long enough to verify parsing; expected: all three are accepted without topology assertion failures.

- [ ] **Step 3: Write the end-to-end smoke test**

  Prepare balanced `64 KiB` per pair on the single-server profile, balanced `64 KiB` per pair on the nonblocking multi-server profile, and FAST Zipf `alpha=0.9, bytes_per_unit=16` on the `2:1` profile. For each matrix run all four schedulers; assert clean exit, exact FCT-flow multiset, matching logical bytes across schedulers, deterministic artifact hashes, and FAST stage-violation fields.

- [ ] **Step 4: Run the full vertical smoke matrix**

  Run: `python3 -m unittest tests.test_e2e_smoke -v` from the package directory with `SIMAI_NS3_BINARY` pointing to the just-built executable. Expected: 12/12 runs PASS with no polling-based termination.

- [ ] **Step 5: Document reproducible commands and perform final verification**

  Document submodule initialization, package invocation, build, prepare/run/analyze commands, schemas, open-loop FAST meaning, and exclusions in `README.md`. Then run the 13-case FAST check, complete Python suite, C++ parser test, native MICRO regression, 12-run ns-3 smoke, `git diff --check`, and `git status --short`; expected: every command succeeds and only intentional files/submodule modifications remain.
