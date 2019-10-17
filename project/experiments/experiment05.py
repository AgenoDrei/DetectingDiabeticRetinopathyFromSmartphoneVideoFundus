from utils import load_image, show_image, load_images, show_image_row, print_progress_bar, enhance_contrast_image, \
    get_retina_mask, get_hsv_colors, show_means
import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys

NUM_CLUSTERS = 5
MODEL = 'gmm_model_2.mod'
ARTIFACT_THRESHOLD = 0.05
MATRIX_TYPE = cv2.ml.EM_COV_MAT_GENERIC


def run():
    images = load_images('/home/simon/Videos/Anomaly Dataset/raw_images/', img_type='png')
    images2 = load_images('./C001R/', img_type='png')

    images_subset = [images[i] for i in range(0, len(images), 12)]
    images_subset.extend([images2[i] for i in range(0, len(images2), 2)])
    images_subset = [enhance_contrast_image(img, clip_limit=4) for img in images_subset]
    images_subset = [cv2.bitwise_and(img, get_retina_mask(img)) for img in images_subset]
    #show_image_row(images_subset, name='Raw images')

    #train_gmm(images_subset)
    #prop = segement_image(images_subset[5], use_colors=True, show_result=True)
    #show_two_classes((4, 2), images_subset[5], prop, threshold=(0.05, 0.9))

    props = [segement_image(img, use_colors=True, show_result=False) for img in images_subset]
    [show_two_classes((4, 2), img, props[i], threshold=(0.01, 0.99)) for i, img in enumerate(images_subset)]


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

    print(samples.shape)
    samples = np.array([sample for sample in samples if not np.array_equal(sample, [0, 0, 0])])

    print('INFO> No model found, training GMM...')
    criterion = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    em.setTermCriteria(criterion)
    em.setClustersNumber(NUM_CLUSTERS)
    em.setCovarianceMatrixType(MATRIX_TYPE)
    retval, logLikelihoods, labels, probs = em.trainEM(samples)

    em.save('gmm_model_2.mod')
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

    ret, result = em.predict(np.float32(img_data))
    best_guess = np.argmax(result, axis=1)
    seg_img = colors[best_guess] if use_colors else means[best_guess]

    # for i, sample in enumerate(img_data):
    #    ret, result = em.predict2(sample)
    #    counts[np.argmax(result)] += 1
    #    color = means[np.argmax(result)] if use_colors == False else colors[np.argmax(result)]
    #    seg_img[i,:] = color

    seg_img = seg_img.reshape(img.shape)
    seg_img = np.uint8(seg_img)
    # print(counts)
    if show_result:
        show_image_row([cv2.cvtColor(seg_img, cv2.COLOR_HSV2BGR), cv2.cvtColor(img, cv2.COLOR_HSV2BGR)])
    return result


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


'''
Experimenting GMM
'''
if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()
