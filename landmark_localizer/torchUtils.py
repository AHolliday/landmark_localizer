import torch
import numpy as np
from .ObjectExtractor import WorldObject


def matchByBruteForce(feats1, feats2, metric='cosine', augMatrix=None,
                      returnDists=True):
    """
    Perform brute-force matching between each pair of object sets in a list.
    feats1: a 2D torch tensor of feature vectors for objects in the 1st image
    feats2: a 2D torch tensor of feature vectors for objects in the 2nd image
    metric: either 'euclidean' or 'cosine'
    augMatrix: a matrix by which to multiply the computed disances before
        matching.
    returnDists: if True, return a list of the match distances along
    """
    if metric == 'euclidean':
        dists = []
        for feat in feats1:
            dists.append(((feat - feats2) ** 2).sum(-1).sqrt())
        dist_matrix = torch.stack(dists)
    elif metric == 'cosine':
        vec_lens1 = (feats1 ** 2).sum(-1).sqrt()
        vec_lens2 = (feats2 ** 2).sum(-1).sqrt()
        len_products = vec_lens1[:, None] * vec_lens2[None]
        dot_products = []
        for feat in feats1:
            dot_products.append((feat * feats2).sum(-1))
        dot_products = torch.stack(dot_products)
        dist_matrix = 1 - dot_products / len_products

    if augMatrix is not None:
        dist_matrix *= augMatrix

    # perform matching, store the results of the matches,
    matches_1to2 = torch.argmin(dist_matrix, dim=1)
    matches_2to1 = torch.argmin(dist_matrix, dim=0)
    matches = [(i1, int(i2)) for i1, i2 in enumerate(matches_1to2)
               if matches_2to1[i2] == i1]
    dists = [dist_matrix[i1, i2].item() for i1, i2 in matches]
    # dists = dist_matrix[tuple(zip(*matches))]
    matches_and_dists = sorted(zip(matches, dists), key=lambda x: x[1])
    matches, dists = list(zip(*matches_and_dists))
    # including dists between matches if requested
    matches = np.array(matches)
    if returnDists:
        return matches, dists
    return matches


# def matchByBruteForce_fast(allFeatures, metric='cosine', augMatrix=None,
#                            returnDists=True):
#     """allFeatures: n_images x n_feats_per_img x n_dims torch tensor"""
#     feat_dists_shape = (allFeatures.shape[0], allFeatures.shape[1],
#                         allFeatures.shape[1])
#     dist_matrix = torch.zeros(allFeatures.shape[0], allFeatures.shape[0])
#     all_matches = []
#     if metric == 'cosine':
#         # n_images x n_feats_per_img
#         vec_lens = (allFeatures ** 2).sum(-1).sqrt()
#     for ii, img_feature_mat in enumerate(allFeatures):
#         # n_images x n_feats_per_img x n_feats_per_img
#         img_feat_dists = torch.zeros(feat_dists_shape).to(allFeatures.device)
#         for jj, img_feature in enumerate(img_feature_mat):
#             if metric == 'euclidean':
#                 # n_images x n_feats_per_img
#                 dists = ((allFeatures - img_feature) ** 2).sum(-1).sqrt()
#             elif metric == 'cosine':
#                 # n_images x n_feats_per_img
#                 dists = (allFeatures * img_feature).sum(-1)
#             img_feat_dists[:, jj, :] = dists
#         if metric == 'cosine':
#             img_feat_dists /= vec_lens[ii][None, :, None] * vec_lens[:, None]
#
#         # perform the matching
#         # perform matching, store the results of the matches,
#         matches_itox = torch.argmin(img_feat_dists, dim=2)
#         matches_xtoi = torch.argmin(img_feat_dists, dim=1)
#         matches = [(j1, int(j2)) for j1, j2 in enumerate(matches_itox)
#                    if matches_xtoi[j2] == j1]
#         dists = [img_feat_dists[ii, j1, j2].item() for j1, j2 in matches]
#         # dists = dist_matrix[tuple(zip(*matches))]
#         matches_and_dists = sorted(zip(matches, dists), key=lambda x: x[1])
#         matches, dists = list(zip(*matches_and_dists))
#         # including dists between matches if requested
#         all_matches.append(np.array(matches))
#
#     if returnDists:
#         return matches, dists
#     return matches
#     dist_matrix[ii] = img_feat_dists
