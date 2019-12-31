import cv2
import numpy as np
import mahotas as mt
import utils as utl
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed


def extract_feature_vector(X: np.array, bin_size: int = 16, haralick_dist: int = 4) -> np.ndarray:
    channels = cv2.split(X)
    features = []
    for c in channels:  # haarlick features
        # texture_feat_mean: np.array = mt.features.haralick(c, compute_14th_feature=True, distance=haralick_dist, return_mean=True)
        texture_feat: np.array = mt.features.haralick(c, compute_14th_feature=True, distance=haralick_dist, return_mean_ptp=True)
        # ht_mean = textures.mean(axis=0)
        # ht_range = np.ptp(textures, axis=0)
        # f = np.hstack((texture_feat_mean, texture_feat_range))
        features.append(texture_feat)

    img = utl.enhance_contrast_image(X, clip_limit=4.0, tile_size=12)

    hist = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [8, 3, 3], [0, 180, 0, 256, 0, 256])  # Histogram features
    print(f'FEAT> Length haralick features {len(np.hstack(features))}, Length histogram features {len(hist.flatten())}')
    X_trans = np.hstack([np.hstack(features), hist.flatten()])
    return X_trans


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, haralick_dist: int = 4, hist_size=None, clip_limit=4.0):
        if hist_size is None:
            hist_size = [8, 3, 3]
        self.haralick_dist = haralick_dist
        self.hist_size = hist_size
        self.clip_limit = clip_limit

    def fit(self, X, y=None):
        return self

    @staticmethod
    def transform(self, X, y=None):
        X_trans = Parallel(n_jobs=-1)(delayed(self.extract_single_feature_vector)(x, self.haralick_dist, self.hist_size, self.clip_limit) for x in X)
        # X_trans = [self.extract_single_feature_vector(x, self.haralick_dist, self.hist_size, self.clip_limit) for x in X]
        # print(f'FEAT> {np.array(X_trans).shape}')
        return np.array(X_trans)

    def extract_single_feature_vector(self, x, distance, size, limit):
        channels = cv2.split(cv2.cvtColor(x, cv2.COLOR_BGR2LAB))
        features = []
        for c in channels:  # haarlick features
            try:
                texture_feat: np.array = mt.features.haralick(c, compute_14th_feature=True, distance=distance, return_mean_ptp=True)
            except ValueError:
                texture_feat = np.zeros(28, dtype=np.float32)
            features.append(texture_feat)

        img = utl.enhance_contrast_image(x, clip_limit=limit, tile_size=12)
        hist = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, size, [0, 180, 0, 256, 0, 256])  # Histogram features
        feat = np.hstack([np.hstack(features), hist.flatten() ** 0.25])
        return feat

    def get_params(self, deep=True):
        return super().get_params(deep)

    def set_params(self, **params):
        return super().set_params(**params)
