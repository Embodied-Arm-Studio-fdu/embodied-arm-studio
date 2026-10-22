"""Optional GPU/Shifu smoke placeholder; never a required CI test."""

import shutil
import subprocess


def main() -> int:
    print("GPU smoke is optional and requires a separately configured Shifu worker.")
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi"], check=False)
    else:
        print("nvidia-smi not found; skipping GPU validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())