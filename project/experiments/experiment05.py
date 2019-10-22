import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils import show_image, load_images, show_image_row, enhance_contrast_image, \
    get_retina_mask, get_hsv_colors, show_means, float2gray, print_progress_bar

NUM_CLUSTERS = 5
MODEL = 'gmm_model_v2.mod'
ARTIFACT_THRESHOLD = 0.05
MATRIX_TYPE = cv2.ml.EM_COV_MAT_GENERIC


def run():
    images = load_images('/home/simon/Videos/Anomaly Dataset/raw_images/', img_type='png')
    images2 = load_images('./C001R/', img_type='png')

    images_subset = [images[i] for i in range(0, len(images))]
    images_subset.extend([images2[i] for i in range(0, len(images2))])
    images_subset = [enhance_contrast_image(img, clip_limit=4) for img in images_subset]
    images_subset = [cv2.bitwise_and(img, get_retina_mask(img)) for img in images_subset]
    show_image_row(images_subset, name='Raw images')

    #train_gmm(images_subset)

    # props = [segement_image(img, use_colors=True, show_result=False) for img in images_subset]
    # [show_single_class(4, img, props[i], threshold=0.05) for i, img in enumerate(images_subset)]

    masked_images = []
    for i, img in enumerate(images_subset):
        masked_images.append(cv2.bitwise_and(img, get_glare_mask(img)))

    [show_image(img, name='Masked images') for img in masked_images]


def test_thresholds(images:list) -> None:
    thresholds = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.8, 0.9, 0.99, 0.9999, 0.999999]
    prop = segement_image(images[3], use_colors=True, show_result=False)
    for th in thresholds:
        show_single_class(4, images[3], prop, threshold=th, write_to_file=True)


def train_gmm(imgs: list) -> None:
    em: cv2.ml_EM = cv2.ml.EM_create()
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in imgs]
    imgs = [img.reshape(img.shape[0] * img.shape[1], 3) for img in imgs]
    samples = np.vstack(imgs)

    print('INFO> Raw sample size: ',  samples.shape)
    #samples = np.array([sample for sample in samples if not np.array_equal(sample, [0, 0, 0])])
    samples = samples[~np.all(samples == 0, axis=1)]
    print('INFO> Reduced sample size: ', samples.shape)

    print('INFO> No model found, training GMM...')
    criterion = (cv2.TERM_CRITERIA_EPS, 100, 0.01)
    em.setTermCriteria(criterion)
    em.setClustersNumber(NUM_CLUSTERS)
    em.setCovarianceMatrixType(MATRIX_TYPE)
    retval, logLikelihoods, labels, probs = em.trainEM(samples)

    em.save('gmm_model_3.mod')
    print(f'INFO> Training done. Saving model to {MODEL}')
    # print(em.getMeans())
    # print(em.getCovs())
    # print(em.getWeights())

    show_means(em.getMeans(), em.getWeights())


def segement_image(img: np.array, use_colors=False, show_result=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_data = img.reshape(img.shape[0] * img.shape[1], 3)
    em: cv2.ml_EM = cv2.ml.EM_load(MODEL)
    colors = get_hsv_colors(NUM_CLUSTERS)

    means = em.getMeans()
    # seg_img = np.zeros(img_data.shape)
    # counts = np.zeros((NUM_CLUSTERS, 1))

    test_data = img_data[~np.all(img_data == 0, axis=1)]
    #print(test_data.shape)

    ret, result = em.predict(np.float32(test_data))
    result_backprojection_data = np.zeros((img_data.shape[0], NUM_CLUSTERS))
    result_backprojection_data[~np.all(img_data == 0, axis=1)] = result
    best_guess = np.argmax(result_backprojection_data, axis=1)
    seg_img = colors[best_guess] if use_colors else means[best_guess]

    seg_img = seg_img.reshape(img.shape)
    seg_img = np.uint8(seg_img)
    # print(counts)
    if show_result:
        show_image_row([cv2.cvtColor(seg_img, cv2.COLOR_HSV2BGR), cv2.cvtColor(img, cv2.COLOR_HSV2BGR)])
    return result_backprojection_data


def show_single_class(rel_class: int, img: np.array, props: np.array, threshold=ARTIFACT_THRESHOLD, write_to_file=False):
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


def show_two_classes(classes: (int, int), img: np.array, props: np.array, threshold=(ARTIFACT_THRESHOLD, ARTIFACT_THRESHOLD), write_to_file=False):
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


def get_probability_map(class_idx: int, props: np.array, image: np.array) -> np.array:
    prop = props[:, class_idx]
    map = prop.reshape((image.shape[0], image.shape[1]))
    #map = np.log10(map)

    map = cv2.GaussianBlur(map, (7, 7), 0)
    return float2gray(map)


def get_glare_mask(image: np.array, show_mask: bool = False, relevant_segment: int = 4) -> np.array:
    prop = segement_image(image, use_colors=True, show_result=False)
    prop_map = get_probability_map(relevant_segment, prop, image)
    # prop_map_th = cv2.adaptiveThreshold(prop_map, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,k
    #                      thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
    ret2, prop_map_th = cv2.threshold(prop_map, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if show_mask:
        show_image_row([prop_map, prop_map_th], name='Probability map + threshold')
    return cv2.cvtColor(prop_map_th, code=cv2.COLOR_GRAY2BGR)


'''
Experimenting GMM
'''
if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()
