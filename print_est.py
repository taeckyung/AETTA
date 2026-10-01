import json
import os
import re
import argparse

import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

CORRUPTION_LIST = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur",
                   "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
                   "jpeg_compression"]

METHOD_LIST = ["TENT", "EATA", "SAR", "CoTTA", "RoTTA", "SoTTA"]

BASE_DATASET = "cifar10outdist"

LOG_PREFIX = "eval_results"

SEED_LIST = [0, 1, 2]

DIST = 1

RESET = ""

TARGET_OUTDIST = ["original"]


def get_avg_online_acc(file_path, target):
    with open(file_path) as f:
        json_data = json.load(f)

    to_compare = np.array(json_data[target], dtype=float)
    accuracy = np.array(json_data["current_accuracy"], dtype=float)

    if target != "aetta":
        to_compare *= 100

    mae = np.nanmean(np.abs(accuracy - to_compare))

    return accuracy[-1], mae


def process_path(args, path):
    result = {f"{s}_{t}": pd.DataFrame(columns=CORRUPTION_LIST) for s in args.seed for t in args.outdist}
    method = path.split("/")[-1].replace("_outdist", "")
    for (path, _, _) in os.walk(path):
        for corr in CORRUPTION_LIST:
            for seed in args.seed:
                for outdist in args.outdist:
                    if not args.cont:
                        pattern_of_path = f'.*{corr}.*{outdist}.*/'
                    else:
                        pattern_of_path = f'.*cont_{outdist}.*/'

                    if args.reset:
                        pattern_of_path += f'reset_{args.reset}/'
                    else:
                        if 'reset' in path:
                            continue

                    if method == "SoTTA":
                        if args.dataset == "cifar100outdist":
                            suffix = "_mt0.2_HUS_ht0.66_lr0.001"  # SoTTA
                        elif args.dataset == "cifar10outdist":
                            suffix = "_mt0.2_HUS_ht0.99_lr0.001"  # SoTTA
                        else:
                            suffix = "_mt0.2_HUS_ht0.33_lr0.001"
                    elif method == "RoTTA":
                        suffix = "_mt0.05_CSTU"  # RoTTA
                    else:
                        suffix = ""

                    pattern_of_path += f'{args.prefix}_{seed}_dist{args.dist}{suffix}{args.suffix}'

                    if args.cont:
                        pattern_of_path += f'/{corr}.*'

                    pattern_of_path = re.compile(pattern_of_path)
                    if pattern_of_path.match(path):
                        if not path.endswith('/cp'):  # ignore cp/ dir
                            try:
                                acc, mae = get_avg_online_acc(os.path.join(path, 'online_eval.json'), args.target)
                                if not args.cont:
                                    prefix = outdist
                                    path = path.split('/')[-1]
                                else:
                                    prefix = path.split("/")[-2]
                                    path = '/'.join(path.split('/')[:-2])
                                key = method + "_" + prefix + f"({path})"
                                result[f"{seed}_{outdist}"].loc[key, corr] = float(mae)
                            except Exception as e:
                                pass
    return result


def main(args):
    root = args.log_name + args.dataset
    paths = [os.path.join(root, f"{method}_outdist") for method in args.method]
    with Pool(processes=len(paths)) as p:
        func = partial(process_path, args)
        results = p.map(func, paths)

    for outdist in args.outdist:
        for seed in args.seed:
            print(f"SEED:{seed}, OUTDIST: {outdist}")
            result = pd.concat([results[i][f"{seed}_{outdist}"] for i in range(len(results))]).astype(float).round(3)
            print(result.to_csv())


def parse_arguments():
    """Command line parse."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=BASE_DATASET,
                        help='Base dataset')

    parser.add_argument('--method', nargs="*", type=str, default=METHOD_LIST,
                        help='Method name')

    parser.add_argument('--seed', nargs="*", type=int, default=SEED_LIST,
                        help='Seed')

    parser.add_argument('--outdist', nargs="*", type=str, default=TARGET_OUTDIST,
                        help='Outdist type')

    parser.add_argument('--prefix', type=str, default=LOG_PREFIX,
                        help='Log prefix')

    parser.add_argument('--dist', type=str, default=DIST,
                        help='Distribution')

    parser.add_argument('--reset', type=str, default=RESET,
                        help='Reset function')

    parser.add_argument('--cont', default=False, action='store_true',
                        help='Continual learning')

    parser.add_argument('--suffix', default="", type=str,
                        help='Suffix for folder name')

    parser.add_argument('--target', default="", type=str,
                        help='Target key to calculate MAE. Example: aetta, softmax_score, gde, src_validation, adv_perturb')

    parser.add_argument('--log_name', default="log/", type=str,
                        help='Name of logging directory')


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    print(
        f"DATASET: {args.dataset}\n"
        f"LOG_PREFIX: {args.prefix}\n"
        f"METHOD: {args.method}\n"
        f"SEED: {args.seed}\n"
        f"OUTDIST: {args.outdist}\n"
        f"DIST: {args.dist}\n"
        f"RESET: {args.reset}\n"
        f"CONTINUAL: {args.cont}\n"
    )

    main(args)
