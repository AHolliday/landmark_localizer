import torch
import numpy as np
from itertools import product

from landmark_localizer import torchUtils as tchu
from landmark_localizer import localization as loc


def testMatchByBruteForce():
    max_vecs = 500
    min_vecs = int(max_vecs * 0.5)
    max_vec_len = 3000
    num_tests = 10
    for _ in range(num_tests):
        # generate some random vectors
        vec_len = np.random.randint(1, max_vec_len + 1)
        feats1_count = np.random.randint(min_vecs, max_vecs)
        feats1 = np.random.randn(feats1_count, vec_len)
        feats2_count = np.random.randint(min_vecs, max_vecs)
        feats2 = np.random.randn(feats2_count, vec_len)

        feats1_t = torch.tensor(feats1)
        feats2_t = torch.tensor(feats2)
        aug_matrix = np.random.random((feats1_count, feats2_count))
        for metric, am in product(['cosine', 'euclidean'], [None, aug_matrix]):
            # make tensors of them
            matches, dists = loc.matchByBruteForce(
                feats1, feats2, metric, crossCheck=True,
                filterZeroVectors=False, augMatrix=am)
            if am is not None:
                am = torch.tensor(am)
            torch_matches, torch_dists = tchu.matchByBruteForce(
                feats1_t, feats2_t, metric, am)
            assert np.all(matches == torch_matches)
            assert np.all(np.isclose(dists, torch_dists))

    # test use of augMatrix

if __name__ == "__main__":
    testMatchByBruteForce()
