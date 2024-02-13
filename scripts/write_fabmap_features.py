import cv2
import argparse
import yaml
from tqdm import tqdm
import numpy as np

# from landmark_localizer import ObjectExtractor as oe
from landmark_localizer import experimentUtils as expu


def write_features_for_fabmap(image_paths, config, out_path, is_for_vocab):
    obj_engine = expu.getObjectExtractorFromConfig(config)
    all_features = []
    fs = cv2.FileStorage(out_path, cv2.FILE_STORAGE_WRITE)
    for ip in tqdm(image_paths):
        # include size ratios? nah...
        scene = obj_engine.detectAndComputeScene(ip)
        if len(scene) == 0:
            # ignore empty scenes
            continue
        if not is_for_vocab:
            key = ip.replace('/', '__').replace('.', ' ')
            fs.write(key, scene._descriptors)
        else:
            all_features.append(scene._descriptors)
    if is_for_vocab:
        # write the features as a single giant matrix
        fs.write("VocabTrainData", np.concatenate(all_features))
    fs.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('list', help='\
Text file listing images from which to extract features.')
    parser.add_argument('cfg', help='Configuration file for the extractor')
    parser.add_argument('outpath', help='Path to which to write the results')
    parser.add_argument('-v', '--vocab', action='store_true', help='\
If provided, write features in the format expected for vocab training data.')
    args = parser.parse_args()
    with open(args.list, 'r') as ff:
        image_paths = [ll.strip() for ll in ff]
    with open(args.cfg, 'r') as ff:
        config = yaml.load(ff)
    write_features_for_fabmap(image_paths, config, args.outpath, args.vocab)
