import subprocess
import sys
from pathlib import Path


def main():
    folder = Path(__file__).parent
    subprocess.run([sys.executable, "labling.py"], cwd=folder, check=True)
    subprocess.run([sys.executable, "visualisation.py"], cwd=folder, check=True)


if __name__ == "__main__":
    main()
