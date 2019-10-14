from utils import  load_image, show_image, load_images, show_image_row, print_progress_bar, enhance_contrast_image, get_retina_mask
import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
NUM_CLUSTERS = 6
MODEL = 'gmm_model_1.mod'

def run():
    images = load_images('/home/simon/Videos/Anomaly Dataset/raw_images/', img_type='png')
    #images = load_images('./C001R_Cut/', img_type='jpg')
    #images = [enhance_contrast_image(img, clip_limit=4) for img in images]

    images_subset = [images[i] for i in range(0, len(images), 6)]
    images_subset = [enhance_contrast_image(img, clip_limit=4) for img in images_subset]
    images_subset = [cv2.bitwise_and(img, get_retina_mask(img)) for img in images_subset]
    show_image_row(images_subset, name='Raw images')

    #train_gmm(images_subset)
    [segement_image(img, use_colors=True) for img in images_subset]


def train_gmm(imgs:list) -> None:
    em:cv2.ml_EM = cv2.ml.EM_create()
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in imgs]
    imgs = [img.reshape(img.shape[0] * img.shape[1], 3) for img in imgs]
    samples = np.vstack(imgs)

    print(samples.shape)
    samples = np.array([sample for sample in samples if not np.array_equal(sample, [0, 0, 0])])

    print('INFO> No model found, training GMM...')
    criterion = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    em.setTermCriteria(criterion)
    em.setClustersNumber(NUM_CLUSTERS)
    retval, logLikelihoods, labels, probs = em.trainEM(samples)

    em.save('gmm_model_1.mod')
    print('INFO> Training done. Saving model to gmm_model_1.mod')
    print(em.getMeans())
    print(em.getCovs())
    print(em.getWeights())

    show_means(em.getMeans(), em.getWeights())


def segement_image(img:np.array, use_colors=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_data = img.reshape(img.shape[0] * img.shape[1], 3)
    em:cv2.ml_EM = cv2.ml.EM_load(MODEL)
    colors = get_hsv_colors(NUM_CLUSTERS)

    means = em.getMeans()
    #seg_img = np.zeros(img_data.shape)
    #counts = np.zeros((NUM_CLUSTERS, 1))

    ret, result = em.predict(np.float32(img_data))
    best_guess = np.argmax(result, axis=1)
    seg_img = colors[best_guess] if use_colors else means[best_guess]

    #for i, sample in enumerate(img_data):
    #    ret, result = em.predict2(sample)
    #    counts[np.argmax(result)] += 1
    #    color = means[np.argmax(result)] if use_colors == False else colors[np.argmax(result)]
    #    seg_img[i,:] = color

    seg_img = seg_img.reshape(img.shape)
    seg_img = np.uint8(seg_img)
    #print(counts)
    show_image_row([cv2.cvtColor(seg_img, cv2.COLOR_HSV2BGR), cv2.cvtColor(img, cv2.COLOR_HSV2BGR)])


def show_means(means:np.array, weights):
    show_strip = np.zeros((100, means.shape[0] * 100, means.shape[1]))
    progress = 0
    for i, mean in enumerate(means):
        start, stop = int(progress), int(progress + weights[0, i] * 100 * means.shape[0])
        show_strip[0:100, start:stop, :] = mean
        progress += weights[0, i] * 100 * means.shape[0]

    show_strip = np.uint8(show_strip)
    print(show_strip.shape)
    show_image(cv2.cvtColor(show_strip, cv2.COLOR_HSV2BGR))


def get_hsv_colors(n:int):
    colors = np.zeros((n, 3), dtype=np.uint8)
    hue = np.arange(0, 180, 180/n)
    colors[:, 0] = hue
    colors[:, 1] = colors[:, 2] = 255
    return colors


'''
Experimenting GMM
'''
if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()