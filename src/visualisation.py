import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

folder = Path(__file__).parent.parent
file_path = folder / "data" / "cars.csv"
PINK = "pink"
GRID = "white"

plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "legend.facecolor": "black",
    "legend.edgecolor": "white",
})

def load_data(file_path, show_info=True):
    df = pd.read_csv(file_path)
    if show_info:
        print(df.head())
        print(df.describe())
    return df


def line_plot(df):
    car = df.dropna(subset=["speed"]).sort_values("id")
    speed = car["speed"]
    plt.figure(figsize=(15, 6))
    plt.plot(car["id"], speed, color=PINK, marker="o", label="Cars")
    plt.xlabel("Track ID")
    plt.ylabel("Speed (km/h)")
    plt.title("Cars Speed")
    plt.xticks(car["id"], rotation=90)
    plt.ylim(30, speed.max()+5)
    plt.grid(axis="y", color=GRID, alpha=0.3)
    plt.legend()
    plt.savefig(folder / "cars_speed.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    data = load_data(file_path)
    line_plot(data)


if __name__ == "__main__":
    main()
