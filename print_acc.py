import json
import os
import re
import argparse

import pandas as pd
from multiprocessing import Pool
from functools import partial

from pathlib import Path

# from tabulate import tabulate

CORRUPTION_LIST = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur",
                   "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
                   "jpeg_compression"]

# METHOD_LIST = ["Src", "TENT", "PseudoLabel", "BN_Stats", "SAR", "RoTTA", "CoTTA", "LAME", "MEMO", "SoTTA", "EATA"]
METHOD_LIST = ["TENT", "EATA", "SAR", "CoTTA", "RoTTA", "SoTTA"]

BASE_DATASET = "cifar10outdist"

LOG_PREFIX = "eval_results"

SEED_LIST = [0, 1, 2]

DIST = 1

RESET = ""

# TARGET_OUTDIST = ["original", "cifar100", "mnist", "uniform", "repeat"]
TARGET_OUTDIST = ["original"]


def get_avg_online_acc(file_path):
    f = open(file_path)
    json_data = json.load(f)
    f.close()
    return json_data['accuracy'][-1]


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
                        pattern_of_path = f'.*cont_{outdist}.*{args.suffix}.*/'

                    if args.reset:
                        pattern_of_path += f'reset_{args.reset}/'
                    else:
                        if 'reset' in path:
                            continue

                    if method == 'Src':
                        pattern_of_path += f'.*{args.prefix}_.*{seed}.*/'
                    elif 'repeat' in outdist:  # attack
                        if args.dataset in ['cifar10outdist', 'cifar100outdist']:
                            attack = "tta_attack_indiscriminate_num20_step10_eps0.1"
                        elif args.dataset == 'imagenetoutdist':
                            attack = "tta_attack_indiscriminate_num20_step1_eps0.2"
                        else:
                            raise NotImplementedError
                        pattern_of_path += f'{attack}/.*{args.prefix}_.*{seed}_dist{args.dist}.*'
                    else:
                        pattern_of_path += f'{args.prefix}_.*{seed}_dist{args.dist}.*'

                    if args.cont:
                        pattern_of_path += f'{corr}.*'

                    pattern_of_path = re.compile(pattern_of_path)
                    if pattern_of_path.match(path):
                        if not path.endswith('/cp'):  # ignore cp/ dir
                            try:
                                acc = get_avg_online_acc(os.path.join(path, 'online_eval.json'))
                                if not args.cont:
                                    prefix = outdist
                                    path = '/'.join(path.split('/')[:-3])
                                else:
                                    prefix = path.split("/")[-2]
                                    path = '/'.join(path.split('/')[:-2])
                                key = method + "_" + prefix + f"({path})"
                                result[f"{seed}_{outdist}"].loc[key, corr] = float(acc)
                            except Exception as e:
                                pass
    return result


def main(args):
    root = args.root_log + "/" + args.dataset
    paths = [os.path.join(root, f"{method}_outdist") for method in args.method]
    with Pool(processes=len(paths)) as p:
        func = partial(process_path, args)
        results = p.map(func, paths)

    for outdist in args.outdist:
        for seed in args.seed:
            print(f"SEED:{seed}, OUTDIST: {outdist}")
            result = pd.concat([results[i][f"{seed}_{outdist}"] for i in range(len(results))])
            print(result.to_csv())
            result.to_csv(args.save_path + f"/{outdist}_{seed}_{args.dist}.csv")


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
    
    parser.add_argument('--root_log', type=str, default="log",
                        help='Reset function')
    
    parser.add_argument('--save_path', type=str, default="csvs",
                        help='Reset function')

    parser.add_argument('--suffix', default="", type=str,
                        help='Suffix for folder name')


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    Path(args.save_path).mkdir(parents=True, exist_ok=True)


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
