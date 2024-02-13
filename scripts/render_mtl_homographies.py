import argparse
import pickle
import cv2
from pathlib import Path

from realWorldExperiment import Result
from plotRealWorldResults import groupByImagePair
from landmark_localizer import myCvTools as mycv


def render_homographies(results_with_configs, outsize=None, outdir=None,
                        side_by_side=False):
    for config, results_list in results_with_configs:
        image_pair_results = groupByImagePair(results_list)
        # for each image pair:
        for pair_name, pair_results in image_pair_results.items():
            # produce the mapped image based on the homography
            # arbitrarily pick the first one
            try:
                pr = [pr for pr in pair_results if not pr.is_failure][0]
            except IndexError:
                print("No successes for", pair_results[0].data_pair_name,
                      "with config", config['name'])
                continue

            far_img, near_img = [cv2.resize(im, pr.img_shape)
                                 for im in pr.images]
            Hmat = pr.H
            if side_by_side:
                # render the three images together, side-by-side
                out_img = mycv.renderHomography(far_img, near_img, Hmat, True)
            else:
                out_img = cv2.warpPerspective(far_img, Hmat, pr.img_shape)
            if outsize:
                out_img = mycv.scaleImage(out_img, outsize)

            display_name = config['name'] + '_' + \
                pr.save_friendly_data_pair_name
            if side_by_side:
                display_name += '_sbs'
            if outdir:
                # save the homography
                outpath = Path(outdir) / (display_name + '.png')
                cv2.imwrite(str(outpath), out_img)
            else:
                # show it, wait for key press
                mycv.showNow(out_img, name=display_name, raiseOnEsc=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs='+',
                        help='Pickle file(s) containing saved MTL results')
    parser.add_argument('-o', '--outdir', help='\
Directory in which to save rendered homographies.  If not provided, the \
homographies will be displayed instead of saved.')
    parser.add_argument('-s', '--size', type=int, help='\
Scale the output images so their smaller axis is this size.')
    parser.add_argument('--sbs', action='store_true', help='\
Render the homography and the two original images side-by-side')

    args = parser.parse_args()

    all_results = []
    for data_file in args.data:
        with open(data_file, 'rb') as ff:
            base_cfg, stored_results = pickle.load(ff)
            all_results += stored_results
    render_homographies(all_results, args.size, args.outdir, args.sbs)
