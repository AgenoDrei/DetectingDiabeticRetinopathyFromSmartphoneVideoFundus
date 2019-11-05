import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils import show_image, load_images, show_image_row, enhance_contrast_image, \
    get_retina_mask, get_hsv_colors, show_means, float2gray, print_progress_bar

NUM_CLUSTERS = 5
MODEL = 'gmm_model_4.mod'
ARTIFACT_THRESHOLD = 0.05
MATRIX_TYPE = cv2.ml.EM_COV_MAT_GENERIC


class Glare_Remover:
    def __init__(self, num_cluster: int = 5, masked_class: int = 1, model_path='./gmm_model_3.mod', matrix_type = cv2.ml.EM_COV_MAT_GENERIC):
        self.n = num_cluster
        self.model_path = model_path
        self.data = []
        self.criterion =  (cv2.TERM_CRITERIA_EPS, 100, 0.01)
        self.matrix_type = matrix_type
        self.current_model = None
        self.masked_class = masked_class

    def train(self) -> None:
        assert len(self.data) is not 0

        em: cv2.ml_EM = cv2.ml.EM_create()
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in self.data]
        imgs = [img.reshape(img.shape[0] * img.shape[1], 3) for img in imgs]
        samples = np.vstack(imgs)

        print('INFO> Raw sample size: ', samples.shape)
        # samples = np.array([sample for sample in samples if not np.array_equal(sample, [0, 0, 0])])
        samples = samples[~np.all(samples == 0, axis=1)]
        print('INFO> Reduced sample size: ', samples.shape)

        print('INFO> Training GMM...')
        em.setTermCriteria(self.criterion)
        em.setClustersNumber(self.n)
        em.setCovarianceMatrixType(self.matrix_type)
        retval, logLikelihoods, labels, probs = em.trainEM(samples)

        em.save(self.model_path)
        self.current_model = em
        print(f'INFO> Training done. Saving model to {self.model_path}')
        # print(em.getMeans())
        # print(em.getCovs())
        # print(em.getWeights())
        #show_means(em.getMeans(), em.getWeights())

    def set_training_data(self, imgs: list) -> None:
        self.data.extend(imgs)

    def get_training_data(self) -> list:
        return self.data

    def show_training_data(self) -> None:
        show_image_row(self.data, 'GMM Training Data')

    def show_means(self) -> None:
        if self.current_model is None:
            self.current_model = cv2.ml.EM_load(self.model_path)

        show_means(self.current_model.getMeans(), self.current_model.getWeights())

    def test_thresholds(self) -> None:
        thresholds = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.8, 0.9, 0.99, 0.9999, 0.999999]
        image = self.data[0]
        prop = self.predict(image, use_colors=True, show_result=False)
        for th in thresholds:
            show_single_class(4, image, prop, threshold=th, write_to_file=True)

    def predict(self, img: np.array, use_colors: bool = False, show_result: bool = False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_data = img.reshape(img.shape[0] * img.shape[1], 3)
        em: cv2.ml_EM = cv2.ml.EM_load(self.model_path)
        colors = get_hsv_colors(self.n)

        means = em.getMeans()
        # seg_img = np.zeros(img_data.shape)
        # counts = np.zeros((NUM_CLUSTERS, 1))

        test_data = img_data[~np.all(img_data == 0, axis=1)]
        # print(test_data.shape)

        ret, result = em.predict(np.float32(test_data))
        result_backprojection_data = np.zeros((img_data.shape[0], self.n))
        result_backprojection_data[~np.all(img_data == 0, axis=1)] = result
        best_guess = np.argmax(result_backprojection_data, axis=1)
        seg_img = colors[best_guess] if use_colors else means[best_guess]

        seg_img = seg_img.reshape(img.shape)
        seg_img = np.uint8(seg_img)
        # print(counts)
        if show_result:
            show_image_row([cv2.cvtColor(seg_img, cv2.COLOR_HSV2BGR), cv2.cvtColor(img, cv2.COLOR_HSV2BGR)])
        return result_backprojection_data

    def predict_joined_prob(self, img: np.array):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_data = img.reshape(img.shape[0] * img.shape[1], 3)
        em: cv2.ml_EM = cv2.ml.EM_load(self.model_path)

        means = em.getMeans()
        weights = em.getWeights()
        covs = em.getCovs()
        test_data = img_data[~np.all(img_data == 0, axis=1)]

        joined_prob = np.zeros((test_data.shape[0], 1), dtype=np.float64)
        for w, m, c in zip(weights[0], means, covs):

            for i, d in enumerate(test_data):
                a = (1 / np.sqrt((2 * np.pi)**3 * np.linalg.det(c)))            # 1/sqrt(2pi**k * det(Cov))
                b = -0.5 * np.transpose(d - m) @ np.linalg.inv(c) @ (d - m)
                joined_prob[i] += w * a * np.exp(b)

        result_backprojection_data = np.zeros((img_data.shape[0], 1))
        result_backprojection_data[~np.all(img_data == 0, axis=1)] = joined_prob

        return result_backprojection_data

    @staticmethod
    def get_probability_map(class_idx: int, props: np.array, image: np.array) -> np.array:
        if props.shape[1] != 1:
            props = props[:, class_idx]
        map = props.reshape((image.shape[0], image.shape[1]))
        # map = np.log10(map)

        map = cv2.GaussianBlur(map, (7, 7), 0)
        return float2gray(map)

    def get_glare_mask(self, image: np.array, show_mask: bool = False, joined_prob: bool = False) -> np.array:
        prop = self.predict(image, use_colors=True, show_result=False) if not joined_prob else self.predict_joined_prob(image)
        prop_map = self.get_probability_map(self.masked_class, prop, image)
        # prop_map_th = cv2.adaptiveThreshold(prop_map, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,k
        #                      thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
        ret2, prop_map_th = cv2.threshold(prop_map, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if show_mask:
            show_image_row([prop_map, prop_map_th], name='Probability map + threshold')
        return prop_map_th

    def get_glare_percentage(self, image: np.array, mask: np.array):
        num_pixels = (~np.all(image == 0, axis=2)).sum()
        num_masked_pixels = (mask == 0).sum()
        return num_masked_pixels / (num_pixels + 1) * 100


def run():
    unglarer: Glare_Remover = Glare_Remover(model_path=MODEL, masked_class=3)

    images = load_images('/data/simon/Anomaly Dataset/raw_images/', img_type='png')
    images2 = load_images('./C001R/', img_type='png')

    images_subset = [images[i] for i in range(0, len(images))]
    images_subset.extend(images2)
    images_subset = [enhance_contrast_image(img, clip_limit=3.5, tile_size=12) for img in images_subset]
    images_subset = [cv2.bitwise_and(img, get_retina_mask(img)[0]) for img in images_subset]
    unglarer.set_training_data(images_subset)

    #unglarer.show_training_data()
    #unglarer.train()
    unglarer.show_means()

    masked_images = []
    for img in unglarer.get_training_data():
        mask = unglarer.get_glare_mask(img, show_mask=False, joined_prob=False)
        percentage = unglarer.get_glare_percentage(img, mask)
        masked_images.append((cv2.bitwise_and(img, cv2.cvtColor(mask, code=cv2.COLOR_GRAY2BGR)), percentage))

    [show_image(img, name=f'Masked images - {percentage:.2f}%') for img, percentage in masked_images]




def show_single_class(rel_class: int, img: np.array, props: np.array, threshold=0.01, write_to_file=False):
    used_image = img.copy()
    img_data = used_image.reshape(img.shape[0] * img.shape[1], 3)
    props = props[:, rel_class]

    # img_data = np.array([(0, 0, 255) for (i, pixel) in enumerate(img_data) if props[i] > threshold])
    for i, pixel in enumerate(img_data):
        if props[i] > threshold:
            img_data[i, :] = (0, 255, 0)

    img_data = img_data.reshape(img.shape)
    img_data = np.uint8(img_data)
    show_image(img_data)
    if write_to_file:
        cv2.imwrite(f'output/gmm_class_{rel_class}_threshold_{threshold}.jpg', img_data)


def show_two_classes(classes: (int, int), img: np.array, props: np.array, threshold=(0.01, 0.1), write_to_file=False):
    class_one, class_two = classes
    thres_one, thres_two = threshold
    used_image = img.copy()
    img_data = used_image.reshape(img.shape[0] * img.shape[1], 3)
    props_1 = props[:, class_one]
    props_2 = props[:, class_two]

    # img_data = np.array([(0, 0, 255) for (i, pixel) in enumerate(img_data) if props[i] > threshold])
    for i, pixel in enumerate(img_data):
        if props_1[i] > thres_one:
            img_data[i, :] = (0, 255, 0)
        if props_2[i] > thres_two:
            img_data[i, :] = (0, 191, 0)

    img_data = img_data.reshape(img.shape)
    img_data = np.uint8(img_data)
    show_image(img_data)
    if write_to_file:
        cv2.imwrite(f'output/gmm_class_{classes}_threshold_{threshold}.jpg', img_data)




'''
Experimenting GMM
'''
if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()
