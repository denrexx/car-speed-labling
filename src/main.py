import subprocess
import sys
from pathlib import Path


def main():
    src = Path(__file__).parent
    folder = src.parent
    subprocess.run([sys.executable, str(src / "labling.py")], cwd=folder, check=True)
    subprocess.run([sys.executable, str(src / "visualisation.py")], cwd=folder, check=True)


if __name__ == "__main__":
    main()
