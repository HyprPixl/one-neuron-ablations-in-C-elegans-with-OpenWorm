
"""
Launches sibernetic+openworm inside the container, optionally mounts a
mutated connectome CSV, and stores logs under output/runs/<run_tag>/.
probably doesn't work on windows, non intel macs, and req's docker 
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output"
CONTAINER_SHARED_ROOT = "/home/ow/shared"
CONNECTOME_CONTAINER_PATH = "/home/ow/c302/c302/data/herm_full_edgelist_MODIFIED.csv"

DT = 0.005
DT_NRN = 0.05
LOGSTEP = 100
PARAM_SET = "C2"
REFERENCE = "FW"
CONFIGURATION = "worm_crawl_half_resolution"
DEVICE = "CPU"
DATAREADER = "UpdatedSpreadsheetDataReader2"

# Always run headless inside the container.
HEADLESS_ENV = {
    "DISPLAY": "",
    "SDL_VIDEODRIVER": "dummy",
    "LIBGL_ALWAYS_SOFTWARE": "1",
    "NEURON_NO_GUI": "1",
    "PYTHONUNBUFFERED": "1",
    "MESA_GL_VERSION_OVERRIDE": "3.3",
    "GALLIUM_DRIVER": "llvmpipe",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenWorm in Docker and save logs.")
    parser.add_argument("--duration", type=float, default=50.0, help="Simulation duration (ms).")
    parser.add_argument("--run-tag", default="run", help="Run tag for outputs/container name.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Host output directory (bind-mounted to /home/ow/shared).",
    )
    parser.add_argument(
        "--connectome-file",
        type=Path,
        default=None,
        help="mutated connectome CSV to mount over the default c302 data file.",
    )
    parser.add_argument("--verbose", action="store_true", help="print the docker command before launch.")
    return parser.parse_args()


def locate_sim_dir(run_root: Path) -> Path:
    sims = [p for p in run_root.iterdir() if p.is_dir()]
    return max(sims, key=lambda p: p.stat().st_mtime) if sims else run_root


def main() -> None:
    args = parse_args()

    output_root = args.output_root.resolve()
    runs_root = (output_root / "runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    run_tag = args.run_tag
    run_root = runs_root / run_tag
    run_root.mkdir(parents=True, exist_ok=True)

    mounts = ["-v", f"{output_root}:{CONTAINER_SHARED_ROOT}:rw"]
    if args.connectome_file is not None:
        host_connectome = args.connectome_file.resolve()
        if not host_connectome.exists() or host_connectome.is_dir():
            raise SystemExit(f"connectome file not found (or is a directory): {host_connectome}")
        mounts.extend(["-v", f"{host_connectome}:{CONNECTOME_CONTAINER_PATH}:ro"])

    env_exports = " ".join(f"{k}={v}" for k, v in HEADLESS_ENV.items())
    sim_cmd = (
        f"python3 sibernetic_c302.py "
        f"-duration {args.duration} -dt {DT} -dtNrn {DT_NRN} "
        f"-logstep {LOGSTEP} -device {DEVICE} "
        f'-configuration "{CONFIGURATION}" -reference "{REFERENCE}" '
        f'-c302params "{PARAM_SET}" -datareader "{DATAREADER}" '
        f"-outDir {CONTAINER_SHARED_ROOT}/runs/{run_tag}"
    )
    inner = f"export {env_exports} && cd $SIBERNETIC_HOME && {sim_cmd}"

    cmd = ["docker", "run", "--rm", "--name", f"ow-eval-{run_tag}"]
    for k, v in HEADLESS_ENV.items():
        cmd.extend(["-e", f"{k}={v}"])
    cmd += mounts + ["openworm/openworm:latest", "bash", "-lc", inner]

    if args.verbose:
        print("(eval) launching:", " ".join(cmd))

    try:
        proc = subprocess.run(cmd, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise SystemExit("docker not found on PATH") from exc

    sim_dir = locate_sim_dir(run_root)
    (sim_dir / "container_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (sim_dir / "container_stderr.log").write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        err_hint = (proc.stderr or "").strip().splitlines()[-1:] or []
        hint_text = f" ({err_hint[0]})" if err_hint else ""
        print(
            f"(eval) docker run failed with code {proc.returncode}{hint_text}; "
            f"see {sim_dir}/container_stderr.log",
            file=sys.stderr,
        )
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
