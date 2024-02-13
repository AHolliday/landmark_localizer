import os
import pickle
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from landmark_localizer import experimentUtils as expu
from landmark_localizer import constants as consts


def extract_boxes(cfg):
    obj_engine = expu.getObjectExtractorFromConfig(cfg)
    all_boxes = []
    img_paths = []
    for ds in tqdm(consts.outdoorDatasets):
        for img_path in ds.images:
            img = cv2.imread(img_path)
            all_boxes.append(obj_engine.box_engine.get_boxes(img))
    return np.concatenate(all_boxes)


def collect_pickled_boxes(directory, box_filename):
    all_boxes = []
    box_counts = []
    for root_dir, _, filenames in os.walk(directory):
        if box_filename in filenames:
            with open(os.path.join(root_dir, box_filename), 'rb') as ff:
                # open it and read in the boxes
                box_dict = pickle.load(ff)
            for _, boxes in box_dict.items():
                if type(boxes) is np.ndarray:
                    # convert it to a list of arrays, the expected format
                    boxes = [row for row in boxes]
                if type(boxes[0]) is tuple:
                    # the first element is the box corners, the rest is mask
                    # info
                    boxes = [bb[0] for bb in boxes]
                all_boxes += boxes
                box_counts.append(len(boxes))
    print('average number of boxes:', np.mean(box_counts))
    # plt.hist(box_counts)
    # plt.title(box_filename)
    # plt.show()
    return all_boxes


def plot_box_histogram(boxes, name):
    # plot a histogram of square roots of box areas
    boxes = np.array(boxes)
    widths = (boxes[:, 2] - boxes[:, 0])
    heights = (boxes[:, 3] - boxes[:, 1])
    areas = widths * heights
    sqrt_areas = np.sqrt(areas)
    print(dir, 'mean size:', sqrt_areas.mean())
    print(dir, 'median size:', np.median(sqrt_areas))

    bins = [32 * ii for ii in range(20)]
    plt.xticks(bins)
    plt.hist(sqrt_areas, bins=bins, cumulative=True,
             label=name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = expu.addCommonArgs(parser)
    parser.add_argument('--kitti', action='store_true', help='\
Do kitti, not Montreal.')
    parser.add_argument('--read', action='store_true', help='\
Read pre-computed SS boxes.')
    args = parser.parse_args()

    if args.kitti:
        dirname = '/localdata/aholliday/iros2018/kitti'
        eb_boxes = collect_pickled_boxes(dirname, 'eb.pkl')
        plot_box_histogram(eb_boxes, dirname)
        plt.title('edge boxes')
        plt.legend()
        plt.figure(2)
        ss_boxes = collect_pickled_boxes(dirname, 'ss.pkl')
        plot_box_histogram(ss_boxes, dirname)
        plt.title('selective search')
        plt.legend()
    else:
        if args.read:
            dirname = '/localdata/aholliday/thesis_data/outdoorDataset'
            boxes = collect_pickled_boxes(dirname, 'ss.pkl')
            plot_box_histogram(boxes, dirname)
            plt.legend()
            plt.figure(2)

        for base_cfg in expu.getConfigsFromArgs(args):
            for cfg in expu.expandConfigToParamGrid(base_cfg):
                boxes = extract_boxes(cfg)
                plot_box_histogram(boxes, 'config')
        plt.legend()
    plt.show()
