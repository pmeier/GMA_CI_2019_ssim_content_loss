from os import path
import pandas as pd
from utils import df_to_csv


def process_ncr_benchmark(results_root):
    ncr_benchmark_root = path.join(results_root, "ncr_benchmark")

    df = pd.read_csv(path.join(ncr_benchmark_root, "raw.csv"), index_col=0)

    def create_loss_str(row):
        ssim_loss, ssim_component_weight_ratio = row
        if not ssim_loss:
            return "SE loss"

        return f"SSIM_loss__component_weight_ratio_{ssim_component_weight_ratio:g}"

    loss_columns = ["ssim_loss", "ssim_component_weight_ratio"]
    df = df.assign(loss_type=df[loss_columns].apply(create_loss_str, axis="columns"))
    df = df.drop(columns=loss_columns)

    df = df.groupby(["name", "loss_type"]).median()
    df = df.drop(columns="seed").unstack()

    df_to_csv(df, path.join(results_root, "ncr_benchmark", "processed.csv"))


def process_ssim_window(results_root):
    ncr_benchmark_root = path.join(results_root, "ssim_window")

    df = pd.read_csv(path.join(ncr_benchmark_root, "raw.csv"), index_col=0)
    df = df.groupby(["window_type", "output_shape", "radius"]).median()
    df = df.drop("seed", axis="columns")

    df_to_csv(df, path.join(ncr_benchmark_root, "processed.csv"))


if __name__ == "__main__":
    root = path.dirname(__file__)
    results_root = path.join(root, "results")

    process_ncr_benchmark(results_root)
    process_ssim_window(results_root)
