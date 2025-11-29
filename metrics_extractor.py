import csv
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "output" / "ablations" / "runs"
RUN_PREFIX = "curv_local_three"
DISPLAY_DIGITS = 8


def parse_motion_log(log_path: Path) -> Tuple[List[Tuple[List[float], List[float]]], List[float]]:
    frames: List[Tuple[List[float], List[float]]] = []
    y_means: List[float] = []

    if not log_path.exists():
        return [], []

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            try:
                float(parts[0])
            except ValueError:
                continue

            axis = None
            axes = {"X:": [], "Y:": []}

            for token in parts[1:]:
                if token in axes:
                    axis = token
                elif axis is not None:
                    try:
                        axes[axis].append(float(token))
                    except ValueError:
                        pass

            if axes["X:"] and axes["Y:"]:
                x_vals = axes["X:"]
                y_vals = axes["Y:"]
                frames.append((x_vals, y_vals))
                y_means.append(mean(y_vals))

    return frames, y_means

def compute_lateral_range(y_means: Sequence[float]) -> float:
    if not y_means:
        return math.nan
    return max(y_means) - min(y_means)


def compute_motion_summary(frames: Sequence[Tuple[Sequence[float], Sequence[float]]]) -> Dict[str, float]:
    n = len(frames)
    if n < 2:
        return {"forward_velocity": math.nan}

    com_x: List[float] = []
    for x_vals, _ in frames:
        if x_vals:
            com_x.append(sum(x_vals) / len(x_vals))

    if len(com_x) < 2:
        fwd = math.nan
    else:
        fwd = (com_x[-1] - com_x[0]) / (len(com_x) - 1)

    return {"forward_velocity": fwd}


def locate_simulation_directory(run_path: Path) -> Path:
    log_path = run_path / "worm_motion_log.txt"
    if log_path.is_file():
        return run_path
    for candidate in run_path.iterdir():
        maybe_log = candidate / "worm_motion_log.txt"
        if candidate.is_dir() and maybe_log.is_file():
            return candidate
    return run_path


def pick_baseline(run_paths: Sequence[Path]) -> Path:
    for path in run_paths:
        if "baseline" in path.name.lower():
            return path
    return run_paths[0] if run_paths else None


def format_float(value: float, digits: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def clean_run_name(name: str) -> str:
    parts = name.split("_")
    return parts[-1] if parts else name

run_paths: List[Path] = [
    p for p in RUNS_ROOT.iterdir() if p.is_dir() and p.name.startswith(RUN_PREFIX)
]
run_paths.sort()

baseline_path = pick_baseline(run_paths)
targets = [baseline_path] + [p for p in run_paths if p != baseline_path]

results: List[Dict[str, object]] = []

for target in targets:
    sim_dir = locate_simulation_directory(target)
    log_path = sim_dir / "worm_motion_log.txt"

    frames, y_means = parse_motion_log(log_path)
    lateral_range = compute_lateral_range(y_means)
    motion = compute_motion_summary(frames)

    results.append(
        {
            "run_name": target.name,
            "run_path": target,
            "lateral_range": lateral_range,
            "forward_velocity": motion["forward_velocity"],
        }
    )

baseline = next(r for r in results if r["run_path"] == baseline_path)

rows: List[Dict[str, str]] = []
seen = set()

for entry in results:
    name = clean_run_name(entry["run_name"])
    if name in seen:
        continue
    seen.add(name)

    base = baseline

    lr = entry["lateral_range"]
    fwd = entry["forward_velocity"]

    d_lr = lr - base["lateral_range"]
    d_fwd = fwd - base["forward_velocity"]

    p_lr = (d_lr / base["lateral_range"] * 100.0) if base["lateral_range"] else 0.0
    p_fwd = (d_fwd / base["forward_velocity"] * 100.0) if base["forward_velocity"] else 0.0

    rows.append(
        {
            "lateral_range": format_float(lr, DISPLAY_DIGITS),
            "lateral_range_delta": format_float(d_lr, DISPLAY_DIGITS),
            "lateral_range_percent_change": format_float(p_lr, DISPLAY_DIGITS),
            "forward_velocity": format_float(fwd, DISPLAY_DIGITS),
            "forward_velocity_delta": format_float(d_fwd, DISPLAY_DIGITS),
            "forward_velocity_percent_change": format_float(p_fwd, DISPLAY_DIGITS),
            "run_name": name,
        }
    )

rows.sort(
    key=lambda r: float(r.get("lateral_range_percent_change") or 0.0),
    reverse=True,
)

csv_path = Path.cwd() / "metrics.csv"

fieldnames = [
    "lateral_range",
    "lateral_range_delta",
    "lateral_range_percent_change",
    "forward_velocity",
    "forward_velocity_delta",
    "forward_velocity_percent_change",
    "run_name",
]

with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[metrics] Wrote {len(rows)} rows to {csv_path}")
