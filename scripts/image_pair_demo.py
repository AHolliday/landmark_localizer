import cv2
import numpy as np

from landmark_localizer import experimentUtils as expu
from landmark_localizer import geometryUtils as geo
from landmark_localizer import myCvTools as mycv
from landmark_localizer import localization as loc
from landmark_localizer import constants as consts


def compare_images(img_path1, img_path2, config, save_images):
    obj_engine = expu.getObjectExtractorFromConfig(config['extractor'])
    loc_config = config['localization']
    scene1 = obj_engine.detectAndComputeScene(img_path1)
    objects1 = scene1.objects
    scene2 = obj_engine.detectAndComputeScene(img_path2)
    objects2 = scene2.objects
    subFeatMetric = None
    if 'subFeatureMetric' in loc_config:
        subFeatMetric = loc_config['subFeatureMetric']
    pts1, pts2, obj_matches = loc.getPointMatchesFromObjects(
        objects1, objects2, loc_config['metric'], subFeatMetric)
    if loc_config['planar']:
        # compute homography (???)
        homography, used_matches = loc.findHomographyFromMatches(
            pts1, pts2, loc_config['ransacThreshold'])
        # display both images and the 'mapped' image between them
        homog_img = mycv.renderHomography(scene1.image, scene2.image,
                                          homography, True)
        homog_img = mycv.scaleImage(homog_img, 500)
        # homog_img = scaleImage
        if not save_images:
            cv2.imshow('homography', homog_img)

    else:
        # attempt to find a valid matrix from these matches
        _, used_matches = loc.findFundamentalMatFromMatches(
            pts1, pts2, loc_config['ransacThreshold'])

    # find used object matches
    # need a function to test if a point pair is in an object pair
    used_obj_matches = set()
    for pt1, pt2 in zip(pts1[used_matches[:, 0]], pts2[used_matches[:, 1]]):
        # check if any object match includes this point match
        for obj1_idx, obj2_idx in obj_matches:
            if objects1[obj1_idx].containsPoint(*pt1) and \
               objects2[obj2_idx].containsPoint(*pt2):
               used_obj_matches.add((obj1_idx, obj2_idx))

    used_obj_matches = np.array(list(used_obj_matches))
    # display matches overlaid on images
    match_img = mycv.renderBoxMatches(
        scene1.image, scene1.boxes, scene2.image, scene2.boxes,
        used_obj_matches, thickness=4)
    match_img = mycv.scaleImage(match_img, 500)
    if save_images:
        cv2.imwrite('matches.png', match_img)
        cv2.imwrite('homography.png', homog_img)
    else:
        cv2.imshow('matches', match_img)
        key = cv2.waitKey()
        if key == ord('s') or key == ord('S'):
            cv2.imwrite('matches.png', match_img)
            cv2.imwrite('homography.png', homog_img)


def main():
    parser = expu.addCommonArgs()
    parser.add_argument('image1', help='path to the first image')
    parser.add_argument('image2', help='path to the second image')
    parser.add_argument('-s', '--save', action='store_true',
                        help='save the image instead of showing it')
    args = parser.parse_args()
    base_configs = expu.getConfigsFromArgs(args)
    for base_config in base_configs:
        for config in expu.expandConfigToParamGrid(base_config):
            compare_images(args.image1, args.image2, config, args.save)


if __name__ == "__main__":
    main()
