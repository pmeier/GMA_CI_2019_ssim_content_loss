from os import path
import itertools
import numpy as np
import pandas as pd
import torch
from torchimagefilter import GaussFilter, BoxFilter
from torchssim import SimplifiedMSSIM
from pystiche.image import read_image, write_image, extract_image_size
from pystiche.image.transforms import Resize, RGBToGrayscale
from utils import make_reproducible, intgeomspace, df_to_csv
from images import (
    get_npr_general_files,
    get_npr_general_proxy_file,
    get_style_image_files,
)
from nst import MeierLohweg2019NCRPyramid, MeierLohweg2019NSTPyramid
from recording import record_nst


def get_eval_transform(image):
    eval_transform = Resize(extract_image_size(image)) + RGBToGrayscale()
    return eval_transform.to(image.device)


def get_input_image(target_image, random=True):
    if random:
        return torch.rand_like(target_image)
    else:
        return target_image.clone()


def perform_ncr(
    target_image, seed=0, level_steps=None, quiet=True, print_steps=None, **kwargs
):
    device = target_image.device
    make_reproducible(seed)
    input_image = get_input_image(target_image, random=True)

    ncr_pyramid = MeierLohweg2019NCRPyramid(**kwargs)
    ncr_pyramid = ncr_pyramid.to(device)
    ncr_pyramid.build_levels(level_steps)

    ncr_pyramid.ncr.content_operator.set_target(target_image)

    output_images = ncr_pyramid(input_image, quiet=quiet, print_steps=print_steps)

    return output_images[-1]


def perform_nst(content_image, style_image, quiet=True, print_steps=None, **kwargs):
    device = content_image.device
    make_reproducible()
    input_image = get_input_image(content_image, random=False)

    nst_pyramid = MeierLohweg2019NSTPyramid(**kwargs)
    nst_pyramid = nst_pyramid.to(device)
    nst_pyramid.build_levels()

    nst_pyramid.nst.content_operator.set_target(content_image)
    nst_pyramid.nst.style_operator.set_target(style_image)

    output_images = nst_pyramid(input_image, quiet=quiet, print_steps=print_steps)

    return output_images[-1]


def benchmark_ncr(images_root, data_root, device):
    target_files = get_npr_general_files()
    ssim_component_weight_ratios = (0.0, 3.0, 9.0, np.inf)
    num_seeds = 5

    loss_variations = [
        (True, ssim_component_weight_ratio)
        for ssim_component_weight_ratio in ssim_component_weight_ratios
    ]
    loss_variations = [(False, None)] + loss_variations
    seeds = np.arange(num_seeds)

    calculate_ssim_score = SimplifiedMSSIM().to(device)
    data = []
    for target_file in target_files:
        target_name = path.splitext(path.basename(target_file))[0]
        target_image = read_image(path.join(images_root, target_file)).to(device)

        eval_transform = get_eval_transform(target_image)
        target_image_eval = eval_transform(target_image)

        for loss_variation, seed in itertools.product(loss_variations, seeds):
            ssim_loss, ssim_component_weight_ratio = loss_variation

            output_image = perform_ncr(
                target_image,
                seed=seed,
                ssim_loss=ssim_loss,
                ssim_component_weight_ratio=ssim_component_weight_ratio,
            )
            output_image_eval = eval_transform(output_image)

            mssim = calculate_ssim_score(output_image_eval, target_image_eval)
            ssim_score = mssim.cpu().item()

            data.append(
                (target_name, ssim_loss, ssim_component_weight_ratio, seed, ssim_score)
            )

    columns = ("name", "ssim_loss", "ssim_component_weight_ratio", "seed", "ssim_score")
    df = pd.DataFrame.from_records(data, columns=columns)
    file = path.join(data_root, "ncr_benchmark.csv")
    df_to_csv(df, file)


def evaluate_steady_state(images_root, data_root, device):
    target_file = path.join(images_root, get_npr_general_proxy_file())
    num_steps = 200_000

    target_image = read_image(target_file).to(device)
    level_steps = (0, num_steps)
    print_steps = intgeomspace(1, num_steps, num=1000)

    for ssim_loss in (False, True):
        with record_nst(quiet=True) as recorder:
            perform_ncr(
                target_image,
                level_steps=level_steps,
                quiet=False,
                print_steps=print_steps,
                ssim_loss=ssim_loss,
                diagnose_ssim_score=True,
            )

            df = recorder.extract()

        loss_type = "SSIM" if ssim_loss else "SE"
        df = df.rename(
            columns={f"Content loss ({loss_type})": "loss", "SSIM score": "ssim_score"}
        )
        df = df[["ssim_score", "loss"]]
        df = df.dropna(axis="index", how="all")

        file = f"{loss_type}.csv"
        file = path.join(data_root, "steady_state", file)
        df_to_csv(df, file)


def evaluate_ssim_window(images_root, data_root, device):
    target_file = path.join(images_root, get_npr_general_proxy_file())
    window_types = ("gauss", "box")
    output_shapes = ("same", "valid")
    radii = range(1, 10)
    num_seeds = 5

    target_image = read_image(target_file).to(device)

    eval_transform = get_eval_transform(target_image)
    target_image_eval = eval_transform(target_image)

    def get_image_filter(window_type, output_shape, radius):
        kwargs = {"output_shape": output_shape, "padding_mode": "replicate"}
        if window_type == "gauss":
            return GaussFilter(radius=radius, std=radius / 3.0, **kwargs)
        else:  # filter_type == "box"
            return BoxFilter(radius=radius, **kwargs)

    seeds = range(num_seeds)

    calculate_mssim = SimplifiedMSSIM().to(device)
    data = []

    for image_filter_params in itertools.product(window_types, output_shapes, radii):
        image_filter = get_image_filter(*image_filter_params)

        for seed in seeds:

            kwargs = {"seed": seed, "image_filter": image_filter}
            output_image = perform_ncr(target_image, **kwargs)
            output_image_eval = eval_transform(output_image)

            mssim = calculate_mssim(output_image_eval, target_image_eval)
            ssim_score = mssim.cpu().item()
            data.append((*image_filter_params, seed, ssim_score))

    columns = ("window_type", "output_shape", "radius", "seed", "ssim_score")
    df = pd.DataFrame.from_records(data, columns=columns)
    file = path.join(data_root, "ssim_window.csv")
    df_to_csv(df, file)


def benchmark_nst(images_root, data_root, device):
    def process_image(file):
        name = path.splitext(path.basename(file))[0]
        image = read_image(path.join(images_root, file)).to(device)
        return name, image

    content_files = get_npr_general_files()
    style_files = get_style_image_files()

    for content_file in content_files:
        content_name, content_image = process_image(content_file)
        for style_file in style_files:
            style_name, style_image = process_image(style_file)

            for ssim_loss in (False, True):
                output_image = perform_nst(
                    content_image, style_image, ssim_loss=ssim_loss, quiet=False
                )

                output_file = "__".join(
                    (content_name, style_name, "ssim" if ssim_loss else "se")
                )
                output_file = path.join(
                    data_root, "nst_benchmark", f"{output_file}.jpg"
                )
                write_image(output_image, output_file)


if __name__ == "__main__":
    root = path.dirname(__file__)
    images_root = path.join(root, "images")
    data_root = path.join(root, "data")
    results_root = path.join(root, "results")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # benchmark_ncr(images_root, data_root, device)
    evaluate_steady_state(images_root, data_root, device)
    # evaluate_ssim_window(images_root, data_root, device)
    # benchmark_nst(images_root, data_root, device)
