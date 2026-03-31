import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "bench_infer_time.py"
CKPT = ROOT / "posneg_run_001" / "best_model.pt"
IMG = ROOT / "20260316_160316.jpg"
OUT = ROOT / "posneg_run_001" / "bench_report.txt"

def run(cmd):
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return result.returncode, result.stdout

def bench(device, include_preprocess):
    cmd = [
        sys.executable, str(BENCH),
        "--checkpoint", str(CKPT),
        "--image", str(IMG),
        "--device", device,
        "--iters", "200",
        "--warmup", "30",
    ]
    if include_preprocess:
        cmd.append("--include_preprocess")
    return run(cmd)

def main():
    sections = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections.append(f"Benchmark Report ({timestamp})")
    sections.append(f"Image: {IMG}")
    sections.append(f"Checkpoint: {CKPT}")
    sections.append("")

    for device in ["cpu", "cuda"]:
        for include in [False, True]:
            title = f"{device.upper()} | {'full (preprocess+model)' if include else 'model only'}"
            code, output = bench(device, include)
            sections.append(title)
            sections.append(output.strip())
            sections.append("")

    OUT.write_text("\n".join(sections))
    print(f"\nSaved report to: {OUT}")

if __name__ == "__main__":
    main()
