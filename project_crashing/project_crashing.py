"""
Project Crashing (Time-Cost Tradeoff) -- versi terintegrasi dengan repo `pysne`.

Perubahan vs versi monolith sebelumnya:
  - Rotation matrix, Sobol sampling, Cluster, spiral dynamics engine (SDOA),
    proses clustering, dan orkestrasi solve_system() SEKARANG memakai modul
    `pysne` yang sama dipakai SNE/Multimodal/Diophantine (tidak duplikat lagi).
  - `ProjectCrashingProblem` sekarang subclass `pysne.problems.base.MultimodalProblem`,
    sehingga bisa langsung dipanggil lewat `pysne.solver.solve_system(problem, params)`.
  - Evaluasi jadwal (`evaluate_schedule`) dibuat FULL VEKTOR lewat
    `evaluate_schedule_batch`: durasi & penjadwalan dengan batasan sumber daya
    dihitung untuk SELURUH populasi sekaligus (bukan loop Python per titik).
    Ini dipakai otomatis oleh fase clustering & SDOA lewat `g_func`/`evaluate_fitness`
    yang polymorphic (terima 1 titik ATAU sekumpulan titik).
  - min_cluster_radius (supaya sub-domain SDOA tidak menyusut di bawah resolusi
    durasi diskrit) sekarang mekanisme GENERIK di `pysne` (properti
    `problem.min_cluster_radius`), bukan parameter khusus yang di-thread manual.
"""

import json
import math
import time
import warnings
from pathlib import Path
from collections import deque

import numpy as np

from pysne.problems.base import MultimodalProblem
from pysne.solver import solve_system


CONFIG = {
    "activity_data_path": "activity_data_v3.json",
    "resource_requirements_path": "resource_requirements_v3.json",
    "resource_capacity_path": "resource_capacity_v3.json",
    "resource_id_map": None,
    "default_capacity_unmapped": 999,
    "max_schedule_horizon": 2000,
    "rounding": "half_up",
    "cost_mode": "crash_slope",
    "m_cluster": 5000,
    "k_cluster": 500,
    "r_cl": 0.95,
    "theta_cl": np.pi / 4,
    "gamma": -float("inf"),
    "num_check_points": 1,
    "sdoa_m": 1000,
    "sdoa_k_max": 100,
    "sdoa_r": 0.95,
    "sdoa_theta": np.pi / 4,
    "delta": 0.05,
    "epsilon": 1e-6,
    "min_radius_safety_factor": 1.0,
}


# ================================================================
#  DATA LOADING & PRECEDENCE GRAPH  (tidak berubah dari versi lama)
# ================================================================

def load_project_data(config=CONFIG):
    activity_data = json.loads(Path(config["activity_data_path"]).read_text())
    resource_requirements = json.loads(Path(config["resource_requirements_path"]).read_text())
    resource_capacity = json.loads(Path(config["resource_capacity_path"]).read_text())
    task_ids = list(activity_data.keys())
    capacity_names = list(resource_capacity.keys())
    raw_ids = {rid for reqs in resource_requirements.values() for rid in reqs.keys()}
    ids_are_numeric = all(rid.isdigit() for rid in raw_ids) if raw_ids else True
    id_map = config.get("resource_id_map")
    if id_map is None:
        if ids_are_numeric:
            id_map = {str(i + 1): name for i, name in enumerate(capacity_names)}
        else:
            id_map = {rid: rid for rid in raw_ids}
    all_resource_ids = sorted(raw_ids, key=(lambda s: int(s)) if ids_are_numeric else (lambda s: s))
    unmapped_ids = [rid for rid in all_resource_ids if rid not in id_map]
    if unmapped_ids:
        warnings.warn(
            f"[project_crashing] Resource id {unmapped_ids} belum punya mapping nama resource; "
            f"diberi kapasitas default {config['default_capacity_unmapped']} (tidak membatasi).",
            UserWarning,
        )
    capacity_by_id = {}
    for rid in all_resource_ids:
        name = id_map.get(rid)
        if name is not None and name in resource_capacity:
            capacity_by_id[rid] = resource_capacity[name]
        else:
            capacity_by_id[rid] = config["default_capacity_unmapped"]
    return {
        "task_ids": task_ids,
        "activity_data": activity_data,
        "resource_requirements": resource_requirements,
        "capacity_by_id": capacity_by_id,
        "id_map": id_map,
    }


def build_topological_order(task_ids, activity_data):
    indegree = {t: 0 for t in task_ids}
    successors = {t: [] for t in task_ids}
    for t in task_ids:
        for pred in activity_data[t]["required_activities"]:
            if pred not in successors:
                continue
            successors[pred].append(t)
            indegree[t] += 1
    queue = deque(t for t in task_ids if indegree[t] == 0)
    order = []
    while queue:
        t = queue.popleft()
        order.append(t)
        for s in successors[t]:
            indegree[s] -= 1
            if indegree[s] == 0:
                queue.append(s)
    if len(order) != len(task_ids):
        missing = set(task_ids) - set(order)
        raise ValueError(f"Precedence graph mengandung siklus atau task tak terjangkau: {missing}")
    return order


def build_project(config=CONFIG):
    data = load_project_data(config)
    data["topo_order"] = build_topological_order(data["task_ids"], data["activity_data"])
    data["task_index"] = {t: i for i, t in enumerate(data["task_ids"])}
    data["rounding"] = config["rounding"]
    data["max_schedule_horizon"] = config["max_schedule_horizon"]
    data["cost_mode"] = config.get("cost_mode", "crash_slope")
    return data


# ================================================================
#  DECODE DURATION (vektor)
# ================================================================

def round_value_batch(x, method="half_up"):
    if method == "half_up":
        return np.floor(x + 0.5)
    if method == "half_even":
        return np.round(x)
    if method == "floor":
        return np.floor(x)
    raise ValueError(f"Metode pembulatan tidak dikenal: {method}")


def decode_duration_batch(x_j, min_time, normal_time, rounding="half_up"):
    """Versi vektor dari decode_duration -- x_j bisa scalar atau array (B,)."""
    x_j = np.clip(x_j, 0.0, 1.0)
    expanded_lo = min_time - 0.5
    expanded_hi = normal_time + 0.5
    scaled = expanded_lo + x_j * (expanded_hi - expanded_lo)
    d = round_value_batch(scaled, rounding)
    return np.clip(d, min_time, normal_time).astype(int)


# ================================================================
#  EVALUASI JADWAL -- FULL VEKTOR (seluruh populasi sekaligus)
# ================================================================

# Kapasitas >= ambang ini dianggap efektif tak terbatas -> tidak perlu dilacak
# (menghemat memori & komputasi, sama seperti unmapped_id di versi lama).
_UNCONSTRAINED_CAPACITY_THRESHOLD = 10_000


def evaluate_schedule_batch(X, project):
    """
    Evaluasi SELURUH populasi `X` (shape (B, n_var)) sekaligus.

    Strategi vektorisasi:
      - loop HANYA di atas topo_order (jumlah task, biasanya puluhan) dan di
        atas hari-hari horizon (ratusan) -- BUKAN di atas populasi B (bisa
        ribuan). Semua operasi per-hari/per-task divektorkan lewat numpy di
        sepanjang sumbu populasi.
      - Kelayakan sumber daya per hari dicek lewat array boolean "blocked"
        (hari yang tidak bisa dipakai task ini karena kapasitas kepenuhan),
        lalu "next_blocked" (hari blocked terdekat >= hari ini, dihitung
        mundur) supaya pencarian start time tercepat per titik populasi bisa
        dilakukan lewat satu scan maju atas hari (bukan loop kandidat awal
        per titik).

    Returns
    -------
    dict dengan key: start, end, duration (masing2 dict task -> array (B,)),
    total_cost (array (B,)), feasible (array bool (B,)), makespan (array (B,)).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    B, n = X.shape

    task_ids = project["task_ids"]
    activity_data = project["activity_data"]
    resource_requirements = project["resource_requirements"]
    capacity_by_id = project["capacity_by_id"]
    topo_order = project["topo_order"]
    rounding = project["rounding"]
    max_horizon = project["max_schedule_horizon"]
    idx = project["task_index"]
    cost_mode = project.get("cost_mode", "crash_slope")

    X = np.clip(X, 0.0, 1.0)

    # Horizon efektif: cukup sepanjang total durasi normal semua task + buffer,
    # tidak perlu sebesar max_schedule_horizon penuh (hemat memori signifikan).
    est_horizon = sum(act["activity_normal_time"] for act in activity_data.values()) + 5
    horizon = max(1, min(max_horizon, est_horizon))

    tracked_resources = set()
    for reqs in resource_requirements.values():
        for rid, qty in reqs.items():
            if qty > 0 and capacity_by_id.get(rid, 0) < _UNCONSTRAINED_CAPACITY_THRESHOLD:
                tracked_resources.add(rid)

    usage = {rid: np.zeros((B, horizon), dtype=np.int32) for rid in tracked_resources}

    start, end, duration = {}, {}, {}
    total_cost = np.zeros(B, dtype=float)
    feasible = np.ones(B, dtype=bool)
    day_idx = np.arange(horizon)

    for t in topo_order:
        act = activity_data[t]
        min_time, normal_time = act["activity_min_time"], act["activity_normal_time"]
        x_j = X[:, idx[t]]
        d = decode_duration_batch(x_j, min_time, normal_time, rounding)

        preds = act["required_activities"]
        earliest_start = np.zeros(B, dtype=int)
        for p in preds:
            if p in end:
                earliest_start = np.maximum(earliest_start, end[p])

        reqs = {rid: qty for rid, qty in resource_requirements.get(t, {}).items()
                if rid in tracked_resources and qty > 0}

        if not reqs:
            s = earliest_start.copy()
        else:
            blocked = np.zeros((B, horizon), dtype=bool)
            for rid, qty in reqs.items():
                cap = capacity_by_id.get(rid, 0)
                blocked |= (usage[rid] + qty > cap)

            # next_blocked[:, day] = hari blocked terdekat >= day (sentinel = horizon
            # berarti "tidak ada"). DULU: loop Python mundur sepanjang `horizon`
            # (satu iterasi Python per hari). SEKARANG: "suffix-min" murni numpy --
            # tandai posisi hari yang blocked (atau sentinel `horizon` bila tidak),
            # lalu ambil running-minimum dari kanan ke kiri sekali jalan
            # (np.minimum.accumulate atas array yang dibalik, lalu dibalik lagi).
            # Hasilnya identik dengan loop lama, tapi 0 iterasi Python.
            blocked_day_pos = np.where(blocked, day_idx[None, :], horizon)
            next_blocked = np.minimum.accumulate(blocked_day_pos[:, ::-1], axis=1)[:, ::-1]

            # Pencarian start time tercepat per titik populasi.
            # DULU: loop Python maju sepanjang `horizon` per task (early-break
            # kalau semua titik sudah "found"), tetap O(horizon) di kasus terburuk.
            # SEKARANG: dibangun matriks boolean (B, horizon) sekali jalan --
            # avail_len[:, day] = panjang jendela kosong berturutan mulai hari
            # `day` (next_blocked - day). Kondisi layak: day >= earliest_start DAN
            # avail_len >= durasi task. Start time tercepat = kolom True pertama
            # per baris, diambil lewat argmax (argmax atas boolean mengembalikan
            # indeks True pertama).
            avail_len = next_blocked - day_idx[None, :]
            feasible_start_mask = (
                (day_idx[None, :] >= earliest_start[:, None]) &
                (avail_len >= d[:, None])
            )
            found = feasible_start_mask.any(axis=1)
            s = np.where(found, feasible_start_mask.argmax(axis=1), -1).astype(int)

            not_found = ~found
            if np.any(not_found):
                # Tidak ketemu slot layak dalam horizon -> jadwalkan longgar di
                # earliest_start (tidak realistis, tapi tetap terdefinisi) dan
                # tandai titik ini infeasible.
                s[not_found] = earliest_start[not_found]
                feasible[not_found] = False

        e = s + d
        feasible &= (e <= max_horizon)

        start[t], end[t], duration[t] = s, e, d

        if reqs:
            mask = (day_idx[None, :] >= s[:, None]) & (day_idx[None, :] < e[:, None])
            for rid, qty in reqs.items():
                usage[rid] += mask * qty

        if cost_mode == "crash_slope":
            total_cost += act["crash_cost"] * (normal_time - d)
        elif cost_mode == "direct":
            total_cost += act["crash_cost"] * d
        else:
            raise ValueError(f"cost_mode tidak dikenal: {cost_mode}")

    makespan = np.zeros(B, dtype=int)
    for t in task_ids:
        makespan = np.maximum(makespan, end[t])

    return {
        "start": start, "end": end, "duration": duration,
        "total_cost": total_cost, "feasible": feasible, "makespan": makespan,
    }


def evaluate_schedule(x, project):
    """Wrapper 1-titik (dipakai laporan/analisis) -- tetap format dict lama."""
    res = evaluate_schedule_batch(np.asarray(x, dtype=float).reshape(1, -1), project)
    task_ids = project["task_ids"]
    return {
        "start": {t: int(res["start"][t][0]) for t in task_ids},
        "end": {t: int(res["end"][t][0]) for t in task_ids},
        "duration": {t: int(res["duration"][t][0]) for t in task_ids},
        "total_cost": float(res["total_cost"][0]),
        "feasible": bool(res["feasible"][0]),
        "makespan": int(res["makespan"][0]),
    }


def compute_min_cluster_radius(project, safety_factor=1.0):
    """Radius minimum cluster: jangan sampai lebih sempit dari resolusi durasi
    diskrit tersempit (task dengan rentang crash paling kecil)."""
    widths = []
    for t, act in project["activity_data"].items():
        w = (act["activity_normal_time"] - act["activity_min_time"]) + 1
        widths.append(w)
    max_width = max(widths) if widths else 1
    return safety_factor * (1.0 / max_width)


# ================================================================
#  PROBLEM CLASS -- subclass MultimodalProblem milik pysne
# ================================================================

class ProjectCrashingProblem(MultimodalProblem):
    """
    Problem project-crashing sebagai MultimodalProblem `pysne`: mencari SEMUA
    kombinasi durasi task (x in [0,1]^n_var) yang sama-sama mencapai biaya
    crash minimum untuk target_duration tertentu (multimodal -> banyak solusi
    optimal setara, bukan cuma satu).
    """
    def __init__(self, project, config=None, penalty=1e12, target_duration=None, penalty_per_day=1e7):
        self.project = project
        self.config = config or CONFIG
        self.penalty = penalty
        self.target_duration = target_duration
        self.penalty_per_day = penalty_per_day
        self.n_evals = 0
        self._min_cluster_radius = compute_min_cluster_radius(
            project, self.config.get("min_radius_safety_factor", 1.0)
        )
        super().__init__()

    @property
    def name(self):
        return "Project Crashing (Time-Cost Tradeoff)"

    @property
    def min_cluster_radius(self):
        """Dibaca otomatis oleh pysne.clustering & pysne.solver supaya radius
        cluster tidak menyusut di bawah resolusi durasi diskrit."""
        return self._min_cluster_radius

    def get_info(self):
        n = len(self.project["task_ids"])
        domain = [(0.0, 1.0)] * n
        return domain, self.config

    def g_func(self, x):
        """
        Polymorphic: terima 1 titik (1D, shape (n_var,)) ATAU sekumpulan titik
        (2D, shape (B, n_var)). Ini memungkinkan fase clustering & SDOA di
        pysne memakai jalur vektor secara otomatis (lihat _batch_evaluate di
        pysne/clustering/modified_clustering_process.py dan usaha vektor di
        pysne/optimizers/sdoa/engine.py) tanpa mengubah kode di sana sama sekali.
        """
        x = np.asarray(x, dtype=float)
        single = (x.ndim == 1)
        X = x.reshape(1, -1) if single else x

        result = evaluate_schedule_batch(X, self.project)
        self.n_evals += X.shape[0]

        cost = result["total_cost"].copy()
        cost = np.where(result["feasible"], cost, cost + self.penalty)

        if self.target_duration is not None:
            overrun_days = np.clip(result["makespan"] - self.target_duration, 0, None)
            cost = cost + self.penalty_per_day * overrun_days

        out_of_box = np.any((X < -1e-9) | (X > 1 + 1e-9), axis=1)
        if np.any(out_of_box):
            overshoot = np.sum(np.clip(-X, 0, None) + np.clip(X - 1, 0, None), axis=1)
            cost = cost + self.penalty * overshoot * out_of_box

        fitness = -cost
        return float(fitness[0]) if single else fitness

    def evaluate_fitness(self, x):
        return self.g_func(x)

    def evaluate_fitness_batch(self, points):
        """Dipertahankan untuk kompatibilitas kode lama yang memanggil nama ini
        secara eksplisit -- secara internal sama saja dengan g_func(points)."""
        return self.g_func(np.asarray(points, dtype=float))


# ================================================================
#  ORKESTRASI -- pakai pysne.solver.solve_system langsung
# ================================================================

def solve_single_problem(problem, config, verbose=True):
    t0 = time.time()
    result = solve_system(problem, config, verbose=verbose)
    candidates = result["optimals"]
    if verbose:
        print(f"  {len(candidates)} kandidat unik ({problem.n_evals} evaluasi objective total)")
    results = []
    for x in candidates:
        sched = evaluate_schedule(np.clip(x, 0, 1), problem.project)
        results.append({
            "x": x, "total_cost": sched["total_cost"], "makespan": sched["makespan"],
            "feasible": sched["feasible"], "start": sched["start"], "end": sched["end"],
            "duration": sched["duration"],
        })
    results.sort(key=lambda r: r["total_cost"])
    elapsed = time.time() - t0
    if verbose:
        print(f"  selesai dalam {elapsed:.2f}s")
    return results


def collect_optimal_solutions(results, target_duration, cost_tol=1e-6):
    valid = [r for r in results if r["feasible"] and r["makespan"] <= target_duration]
    if not valid:
        return []

    min_cost = min(r["total_cost"] for r in valid)
    tied = [r for r in valid if abs(r["total_cost"] - min_cost) <= cost_tol]

    seen = set()
    unique = []
    for r in tied:
        key = tuple(sorted(r["duration"].items()))
        if key not in seen:
            seen.add(key)
            unique.append(r)

    unique.sort(key=lambda r: tuple(sorted(r["duration"].items())))
    return unique


def enforce_monotonic_tradeoff(curve):
    curve_sorted = sorted(curve, key=lambda r: r["target_duration"])
    best_cost = None
    best_solutions = None
    best_makespan = None

    fixed = []
    for row in curve_sorted:
        new_row = dict(row)
        if not row["feasible"]:
            fixed.append(new_row)
            continue

        if best_cost is None or row["best_cost"] < best_cost - 1e-9:
            best_cost = row["best_cost"]
            best_solutions = row["solutions"]
            best_makespan = row["achieved_makespan"]
            new_row["inherited_from_smaller_D"] = False
        else:
            new_row["best_cost"] = best_cost
            new_row["achieved_makespan"] = best_makespan
            new_row["solutions"] = best_solutions
            new_row["n_optimal_solutions"] = len(best_solutions)
            new_row["inherited_from_smaller_D"] = True

        fixed.append(new_row)

    return fixed


def solve_time_cost_tradeoff(config=CONFIG, target_durations=None, penalty_per_day=1e7,
                              sweep_config_overrides=None, cost_tol=1e-6, verbose=True):
    project = build_project(config)
    n = len(project["task_ids"])
    baseline_problem = ProjectCrashingProblem(project, config=config)
    baseline_hi = evaluate_schedule(np.ones(n), project)
    baseline_lo = evaluate_schedule(np.zeros(n), project)
    M_normal, M_min = baseline_hi["makespan"], baseline_lo["makespan"]
    if target_durations is None:
        target_durations = list(range(M_min, M_normal + 1))
    sweep_config = dict(config)
    if sweep_config_overrides:
        sweep_config.update(sweep_config_overrides)
    if verbose:
        print(f"[tradeoff] makespan normal={M_normal} hari, makespan minimum (full crash)={M_min} hari")
        print(f"[tradeoff] sweep {len(target_durations)} target durasi...")
    curve = []
    for D in target_durations:
        problem = ProjectCrashingProblem(project, config=sweep_config, target_duration=D,
                                          penalty_per_day=penalty_per_day)
        results = solve_single_problem(problem, sweep_config, verbose=False)
        optimal_solutions = collect_optimal_solutions(results, D, cost_tol=cost_tol)
        crash_days = M_normal - D

        if optimal_solutions:
            best_cost = optimal_solutions[0]["total_cost"]
            achieved_makespan = optimal_solutions[0]["makespan"]
            feasible = True
        else:
            best_cost = None
            achieved_makespan = None
            feasible = False

        row = {
            "target_duration": D,
            "crash_days": crash_days,
            "best_cost": best_cost,
            "achieved_makespan": achieved_makespan,
            "feasible": feasible,
            "n_optimal_solutions": len(optimal_solutions),
            "solutions": optimal_solutions,
            "schedule": optimal_solutions[0] if optimal_solutions else None,
        }
        curve.append(row)
        if verbose:
            status = "OK" if feasible else "TIDAK TERCAPAI"
            cost_str = f"{best_cost:.2f}" if best_cost is not None else "N/A"
            print(f"  D={D:>4} (crash {crash_days:>3} hari dari normal) -> "
                  f"cost={cost_str}  makespan tercapai={achieved_makespan}  "
                  f"[{status}]  ({row['n_optimal_solutions']} kombinasi optimal ditemukan)")
    curve = enforce_monotonic_tradeoff(curve)
    if verbose:
        n_inherited = sum(1 for r in curve if r.get("inherited_from_smaller_D"))
        if n_inherited:
            print(f"[tradeoff] {n_inherited} dari {len(curve)} titik D diperbaiki "
                  f"(cost diwariskan dari D lebih kecil)")
    return {"project": project, "curve": curve, "M_normal": M_normal, "M_min": M_min}


# ================================================================
#  PELAPORAN  (tidak berubah dari versi lama)
# ================================================================

def format_one_combo_table(sched, project):
    activity_data = project["activity_data"]
    lines = []
    lines.append(f"    {'Task':<14}{'Normal':>7}{'Aktual':>7}{'Crash(hr)':>10}"
                 f"{'Biaya Crash':>14}{'Mulai':>8}{'Selesai':>9}")
    lines.append("    " + "-" * 70)

    task_rows = []
    for t in project["task_ids"]:
        act = activity_data[t]
        normal_t = act["activity_normal_time"]
        actual_d = sched["duration"][t]
        crashed_days = normal_t - actual_d
        task_cost = crashed_days * act["crash_cost"]
        s, e = sched["start"][t], sched["end"][t]
        task_rows.append((t, normal_t, actual_d, crashed_days, task_cost, s, e))

    task_rows_by_start = sorted(task_rows, key=lambda r: (r[5], r[0]))
    for (t, normal_t, actual_d, crashed_days, task_cost, s, e) in task_rows_by_start:
        crash_marker = f"{crashed_days}" if crashed_days > 0 else "-"
        cost_marker = f"{task_cost:,.0f}" if crashed_days > 0 else "-"
        lines.append(f"    {t:<14}{normal_t:>7}{actual_d:>7}{crash_marker:>10}"
                     f"{cost_marker:>14}{s:>8}{e:>9}")

    order_str = " -> ".join(f"{t}(h{s})" for (t, _, _, _, _, s, _) in task_rows_by_start)
    lines.append(f"\n    Urutan pengerjaan: {order_str}")

    crashed_only = [r for r in task_rows if r[3] > 0]
    if crashed_only:
        crash_list = ", ".join(f"{t} (-{c}hr, +Rp{cost:,.0f})"
                                for (t, _, _, c, cost, _, _) in crashed_only)
        lines.append(f"    Task yang di-crash: {crash_list}")
    else:
        lines.append("    Task yang di-crash: (tidak ada, semua durasi normal)")

    return "\n".join(lines)


def format_all_combos_for_duration(row, project):
    lines = []
    n_sol = row["n_optimal_solutions"]

    if not row["feasible"] or n_sol == 0:
        lines.append(f"=== Target durasi = {row['target_duration']} hari "
                      f"-- TIDAK ADA SOLUSI FEASIBLE ditemukan ===")
        return "\n".join(lines)

    inherited_note = (" [solusi diwariskan dari D lebih kecil]"
                       if row.get("inherited_from_smaller_D") else "")
    header = (f"=== Target durasi = {row['target_duration']} hari "
              f"(crash {row['crash_days']} hari dari normal) "
              f"| Biaya crash optimal = Rp {row['best_cost']:,.0f} "
              f"| Makespan tercapai = {row['achieved_makespan']} hari "
              f"| {n_sol} kombinasi sama-sama optimal ditemukan{inherited_note} ===")
    lines.append(header)

    for i, sched in enumerate(row["solutions"], start=1):
        lines.append(f"\n  --- Kombinasi {i} dari {n_sol} ---")
        lines.append(format_one_combo_table(sched, project))

    return "\n".join(lines)


def build_full_tradeoff_report(out):
    lines = []
    lines.append("=" * 78)
    lines.append(" LAPORAN RINCI TIME-COST TRADEOFF PER TARGET DURASI")
    lines.append(f" Makespan normal = {out['M_normal']} hari | "
                 f"Makespan minimum (full crash) = {out['M_min']} hari")
    lines.append("=" * 78)
    for row in out["curve"]:
        lines.append("")
        lines.append(format_all_combos_for_duration(row, out["project"]))
    return "\n".join(lines)


def print_full_tradeoff_report(out):
    print(build_full_tradeoff_report(out))


def save_tradeoff_report_to_txt(out, filepath="laporan_time_cost_tradeoff.txt"):
    report_str = build_full_tradeoff_report(out)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_str)
    abs_path = str(Path(filepath).resolve())
    print(f"\n[info] Laporan lengkap disimpan ke: {abs_path}")
    return abs_path


if __name__ == "__main__":
    project_tmp = build_project(CONFIG)
    n_tmp = len(project_tmp["task_ids"])
    hi = evaluate_schedule(np.ones(n_tmp), project_tmp)
    lo = evaluate_schedule(np.zeros(n_tmp), project_tmp)
    print("makespan normal (x=1):", hi["makespan"], "cost:", hi["total_cost"])
    print("makespan full-crash (x=0):", lo["makespan"], "cost:", lo["total_cost"])
    sample_durations = [241, 242, 243]#list(range(lo["makespan"], hi["makespan"] + 1))
    print("sample_durations:", sample_durations)
    out = solve_time_cost_tradeoff(CONFIG, target_durations=sample_durations, verbose=True)

    print("\n=== RINGKASAN ===")
    for row in out["curve"]:
        print(row["target_duration"], row["crash_days"], row["best_cost"], row["achieved_makespan"])

    print_full_tradeoff_report(out)
    save_tradeoff_report_to_txt(out, filepath="laporan_time_cost_tradeoff.txt")