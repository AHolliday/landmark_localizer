import argparse
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from appearance_experiment import AppearanceImagePairResult

from landmark_localizer import constants as cs


def merge_results_from_same_config(all_exp_results):
    config_results = {}
    cfg_names_to_configs = {}
    for config, results in all_exp_results:
        # cfg_name = config_name(config, SIMPLE_NAME)
        cfg_name = config['name']
        if cfg_name not in config_results:
            config_results[cfg_name] = []
            cfg_names_to_configs[cfg_name] = config
        config_results[cfg_name] += results

    merged_exp_results = []
    for cfg_name, pair_results in config_results.items():
        cfg = cfg_names_to_configs[cfg_name]
        merged_exp_results.append((cfg, pair_results))
    return merged_exp_results


def accuracy_to_csv(cfg_results):
    names, mean_errors, mean_ious = [], [], []
    # TODO group by configs by network, depth in networks...number of parameters?
    result_data = []
    for config, results in cfg_results:
        names.append(config['name'])
        for result in results:
            ips, jps = [np.array(pts) for pts in zip(*result.point_matches)]
            image_mean_error = np.mean(np.linalg.norm(ips - jps, axis=1))
            mean_errors.append(image_mean_error)

            iboxes = np.array(result.iboxes)
            jboxes = np.array(result.jboxes)
            inter_lefts = np.max((iboxes[:, 0], jboxes[:, 0]), axis=0)
            inter_tops = np.max((iboxes[:, 1], jboxes[:, 1]), axis=0)
            inter_rights = np.min((iboxes[:, 2], jboxes[:, 2]), axis=0)
            inter_bottoms = np.min((iboxes[:, 3], jboxes[:, 3]), axis=0)
            intersection_heights = np.maximum(0, inter_bottoms - inter_tops)
            intersection_widths = np.maximum(0, inter_rights - inter_lefts)
            intersection_areas = intersection_heights * intersection_widths
            iareas = (iboxes[:, 2] - iboxes[:, 0] + 1) * \
                (iboxes[:, 3] - iboxes[:, 1] + 1)
            jareas = (jboxes[:, 2] - jboxes[:, 0] + 1) * \
                (jboxes[:, 3] - jboxes[:, 1] + 1)
            ious = intersection_areas / (iareas + jareas - intersection_areas)
            mean_ious.append(np.mean(ious))

        result_csv_entry = {'type': config['extractor']['type'],
                            'mean error': np.mean(mean_errors),
                            'mean iou': np.mean(mean_ious)}
        if 'oeKwargs' in config['extractor']:
            result_csv_entry['feature'] = config['extractor']['oeKwargs'][
                'feature']
        result_data.append(result_csv_entry)
    return pd.DataFrame(result_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pickle', nargs='+', help='\
Path to pickle files containing results to plot.')
    parser.add_argument('-c', '--csv', required=True, help='\
Path to csv file in which to store the output.')
    args = parser.parse_args()
    all_dfs = []
    for pickle_path in tqdm(args.pickle):
        with open(pickle_path, 'rb') as ff:
            cfg_results = pickle.load(ff)
            df = accuracy_to_csv(cfg_results)
            df['pickle path'] = pickle_path
            all_dfs.append(df)

    all_df = pd.concat(all_dfs)
    if 'feature' in all_df.columns:
        grouped_df = all_df.groupby(['type', 'feature']).mean()
    else:
        grouped_df = all_df.groupby('type').mean()
    grouped_df.to_csv(args.csv)


if __name__ == "__main__":
    main()
