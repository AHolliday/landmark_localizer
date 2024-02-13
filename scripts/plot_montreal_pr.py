import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from montreal_pr_experiment import MontrealPrResult


def are_from_same_scene(image_path1, image_path2):
    return Path(image_path1).parent == Path(image_path2).parent


def plot_sim_hists(result):
    same_sims = []
    diff_sims = []
    for ii, ipath in enumerate(result.image_paths):
        for jj, jpath in enumerate(result.image_paths[ii + 1:], start=ii + 1):
            if are_from_same_scene(ipath, jpath):
                same_sims.append(result.sim_matrix[ii, jj])
            else:
                diff_sims.append(result.sim_matrix[ii, jj])

    plt.hist(same_sims, histtype='step', label='same', cumulative=True,
             density=True)
    plt.hist(diff_sims, histtype='step', label='different', cumulative=True,
             density=True)
    plt.title('histogram of similarity scores')
    plt.legend()
    plt.show()


def plot_threshold_curve(result, nsamples=20):
    minsim = result.sim_matrix.min()
    maxsim = result.sim_matrix.max()
    wrongs_and_news = np.zeros((nsamples, 2))
    # if nothing is above the threshold, declare it a new scene
    # three possible outcomes:
        # - correct match (true positive)
        # - incorrect match (false positive???)
        # - no match (false negative)
    # there's always a true match that exists for each query, so a false positive isn't well-defined
    # maybe instead of precision and recall, we want error rate and rejection rate
    # TP if max sim is in the same scene AND is the max
    thresholds = np.linspace(minsim, maxsim, nsamples)
    # thresholds = np.linspace(0.1, 1, nsamples)
    for ti, threshold in enumerate(thresholds):
        for ii, mm in enumerate(result.matches):
            iscores = result.sim_matrix[ii].copy()
            iscores[mm] = 0
            next_best = iscores.max()
            if result.sim_matrix[ii, mm] < threshold:
            # if next_best < result.sim_matrix[ii, mm] * threshold:
                wrongs_and_news[ti, 1] += 1
            elif not are_from_same_scene(result.image_paths[ii],
                                         result.image_paths[mm]):
                wrongs_and_news[ti, 0] += 1
    plt.plot(thresholds, wrongs_and_news[:, 0], label='wrong match rate')
    plt.plot(thresholds, wrongs_and_news[:, 1], label='no match rate')
    plt.legend()
    plt.show()


def plot_conf_mat(result):
    # print how many of the images have a match from the same scene
    img_paths = result.image_paths
    are_correct = []
    for i_path, jj in zip(img_paths, result.matches):
        j_path = img_paths[jj]
        are_correct.append(are_from_same_scene(i_path, j_path))
    print('On', cfg_name, 'of', len(result.matches), 'queries,',
          np.sum(are_correct), 'got correct results')

    # plot the similarity matrix
    # start with guidelines to show separate scenes
    plot_extent = [-0.5, len(result.sim_matrix)]
    sep_color = 'white'
    for ii, image_path in enumerate(img_paths[1:]):
        if not are_from_same_scene(img_paths[ii], image_path):
            line_loc = [ii + 0.5, ii + 0.5]
            plt.plot(plot_extent, line_loc, color=sep_color)
            plt.plot(line_loc, plot_extent, color=sep_color)

    # show incorrect matches: y-loc is the query, x-loc is the bad match it got
    print('circles: y-loc is a query that was incorrectly matched, x-loc is \
its incorrect match.')
    wrong_matches = [(ii, mm) for ii, mm in enumerate(result.matches)
                     if not are_correct[ii]]

    # show what the similarity scores for the incorrect examples actually were
    for ii, mm in wrong_matches:
        print('wrong sim for', ii, ':', result.sim_matrix[ii, mm])
        for jj, jpath in enumerate(img_paths):
            if jpath == img_paths[ii]:
                continue
            if are_from_same_scene(img_paths[ii], jpath):
                print('same sim:', result.sim_matrix[ii, jj])

    if len(wrong_matches) > 0:
        wrong_y, wrong_x = list(zip(*wrong_matches))
        plt.scatter(wrong_x, wrong_y, facecolors='none', edgecolors=sep_color)

    # plot the similarity matrix itself
    plt.matshow(result.sim_matrix, 0)

    # plot the scene names as the axis ticks
    scene_names = [Path(ip).parent.stem for ip in img_paths]
    labels = list(set(scene_names))
    locs = np.array([scene_names.index(ll) for ll in labels]) + 0.5
    plt.xticks(locs, labels, rotation=-90)
    plt.yticks(locs, labels)

    # add a color bar, and set some other properties of the figure
    plt.colorbar()
    plt.tick_params(left=False, top=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results', help='\
A pickle file containing a list of experiment results.')
    args = parser.parse_args()
    with open(args.results, 'rb') as ff:
        results = pickle.load(ff)
    for result in results:
        cfg_name = result.config['name']
        print(cfg_name)
        plot_conf_mat(result)
        plot_sim_hists(result)
        plot_threshold_curve(result)
