import cv2
import numpy as np
import mahotas as mt
import utils as utl


def extract_feature_vector(img: np.array, bin_size: int = 16, haralick_dist=4) -> (np.ndarray, np.ndarray):
    channels = cv2.split(img)
    features = []
    for c in channels:  # haarlick features
        #texture_feat_mean: np.array = mt.features.haralick(c, compute_14th_feature=True, distance=haralick_dist, return_mean=True)
        texture_feat: np.array = mt.features.haralick(c, compute_14th_feature=True, distance=haralick_dist, return_mean_ptp=True)
        #ht_mean = textures.mean(axis=0)
        #ht_range = np.ptp(textures, axis=0)
        #f = np.hstack((texture_feat_mean, texture_feat_range))
        features.append(texture_feat)

    img = utl.enhance_contrast_image(img, clip_limit=4.0, tile_size=12)

    hist = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [8, 3, 3], [0, 180, 0, 256, 0, 256])  # Histogram features
    print(f'FEAT> Length haralick features {len(np.hstack(features))}, Length histogram features {len(hist.flatten())}')
    combined_features = np.hstack([np.hstack(features), hist.flatten()])
    return combined_features
