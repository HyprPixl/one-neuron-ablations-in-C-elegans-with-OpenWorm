#does the ablations on the specifid neurons and runs everythin

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from connectome_brain_mutation import (
    CONNECTOME_FILENAME,
    ConnectomeMutationError,
    connectome_rows,
    write_mutated_connectome,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SCRIPT = REPO_ROOT / "scripts" / "openworm_evaluator.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "single_ablation"
DEFAULT_RUN_PREFIX = "abl"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero each neuron's outgoing synapses and run ablation simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--duration", type=float, default=150.0, help="Simulation duration (ms).")
    parser.add_argument(
        "--neurons",
        default="",
        help="Comma-separated neuron list.",
    )
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=20,
        help="Maximum number of neurons to ablate.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_RUN_PREFIX,
        help="Run tag prefix.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for ablation outputs.",
    )
    parser.add_argument(
        "--connectome-data",
        type=Path,
        default=REPO_ROOT / "patches" / "c302_data",
        help="Directory containing the canonical c302 connectome CSV/metadata.",
    )
    return parser.parse_args(argv)

def run_evaluator(
    *,
    run_tag: str,
    duration: float,
    output_root: Path,
    connectome_file: Optional[Path] = None,
) -> Tuple[bool, Optional[Dict]]:
    cmd = [
        sys.executable,
        str(EVALUATOR_SCRIPT),
        "--duration",
        f"{duration}",
        "--run-tag",
        run_tag,
        "--output-root",
        str(output_root),
    ]
    if connectome_file is not None:
        cmd.extend(["--connectome-file", str(connectome_file)])
    proc = subprocess.run(cmd, text=True, capture_output=True)
    runs_root = output_root / "runs" / run_tag
    sim_dir = runs_root
    log_path = sim_dir / "worm_motion_log.txt"
    if sim_dir.is_dir():
        for candidate in sim_dir.rglob("worm_motion_log.txt"):
            if candidate.is_file():
                log_path = candidate
                sim_dir = candidate.parent
                break
        else:
            for candidate in sim_dir.iterdir():
                maybe = candidate / "worm_motion_log.txt"
                if maybe.is_file():
                    log_path = maybe
                    sim_dir = candidate
                    break
    ok = log_path.exists() and proc.returncode == 0
    metrics = {
        "run_tag": run_tag,
        "simulation_directory": str(sim_dir),
        "container_return_code": proc.returncode,
        "metrics_source": "log" if ok else "",
    }
    return ok, metrics


def select_neurons(args: argparse.Namespace) -> List[str]:
    explicit = [n.strip() for n in args.neurons.split(",") if n.strip()]
    if not explicit:
        print("[ablation] No neurons supplied via --neurons.", file=sys.stderr)
        return []
    return explicit[: args.max_neurons]


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    args.output_root.resolve().mkdir(parents=True, exist_ok=True)

    neurons = select_neurons(args)

    try:
        base_connectome_rows = connectome_rows(args.connectome_data / CONNECTOME_FILENAME)
    except ConnectomeMutationError as exc:
        print(f"[ablation] error: {exc}", file=sys.stderr)
        return 1

    connectome_output_dir = args.output_root / "connectomes"
    connectome_output_dir.mkdir(parents=True, exist_ok=True)

    baseline_run_tag = f"{args.prefix}_baseline"

    print(f"[ablation] Running baseline ({baseline_run_tag}) …")
    ok, _ = run_evaluator(
        run_tag=baseline_run_tag,
        duration=args.duration,
        output_root=args.output_root,
    )
    if not ok:
        print("[ablation] Baseline simulation failed; aborting.", file=sys.stderr)
        return 1

    for neuron in neurons:
        run_tag = f"{args.prefix}_{neuron}"
        connectome_path = connectome_output_dir / f"{run_tag}.csv"
        try:
            write_mutated_connectome(
                rows=base_connectome_rows,
                output_path=connectome_path,
                scale_by_neuron={neuron: 0.0},
            )
        except ConnectomeMutationError as exc:
            print(f"[ablation] error creating connectome for {neuron}: {exc}", file=sys.stderr)
            continue
        print(f"[ablation] Ablating {neuron} …")
        ok, _ = run_evaluator(
            run_tag=run_tag,
            duration=args.duration,
            output_root=args.output_root,
            connectome_file=connectome_path,
        )
        if not ok:
            print(f"[ablation]  warning: simulation failed for {neuron}", file=sys.stderr)
            continue
        print(f"[ablation]  {neuron}: completed")

    return 0


main()
