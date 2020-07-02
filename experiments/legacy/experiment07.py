import numpy as np
import cv2
import sys
sys.path.append('/data/simon/Code/MasterThesis/project/include')
import time_wrap
from utils import  load_image, show_image, enhance_contrast_image, show_image_row, float2gray, load_images, get_retina_mask
from sklearn.decomposition import PCA

#np.set_printoptions(threshold=sys.maxsize)

def run() -> None:
    images = load_images('/data/simon/ownCloud/Data/Reflection Dataset/raw_images/', img_type='png')
    #images.extend(load_images('./C001R/', img_type='png'))

    images = [enhance_contrast_image(img, clip_limit=4) for img in images]

    circles = []
    for i, img in enumerate(images):
        mask, circle = get_retina_mask(img)
        images[i] = cv2.bitwise_and(img, mask)
        circles.append(circle)
    print(circles)

    images = normalize_and_register_retians(images, circles)
    show_image_row(images[::5], name='Training data', time=10000)

    perform_pca(images)

    time_wrap.print_prof_data()


@time_wrap.profile
def perform_pca(images:list, expl_var: float = 0.75) -> None:
    images = [img[:,:,1] for img in images]
    images = [cv2.normalize(img.astype('float64'), None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX) for img in images]
    h, w = images[0].shape
    images = [img.flatten() for img in images]
    train_data = np.vstack(images)

    print(f'INFO> Train data specs: {train_data.shape}, {train_data.nbytes}')

    #mean, eigenvectors = cv2.PCACompute(train_data,  mean=np.array([]), maxComponents=10)
    pca = PCA(n_components=expl_var)
    pca.fit(train_data)
    print(f'INFO> Found {len(pca.explained_variance_ratio_)} eigenvectors, explaining {expl_var} of the data variance')

    for i, ev in enumerate(pca.components_):
        show_image(float2gray(ev.reshape(h, w, 1)), f'Eigenvector {i} - weight: {pca.explained_variance_ratio_[i]}')


def normalize_image_sizes(images: list) -> list:
    min_size, max_size = (sys.maxsize, sys.maxsize), (0, 0)
    for img in images:
        min_size = (min_size[0] if min_size[0] <= img.shape[0] else img.shape[0], min_size[1] if min_size[1] <= img.shape[1] else img.shape[1])
        max_size = (max_size[0] if max_size[0] >= img.shape[0] else img.shape[0], max_size[1] if max_size[1] >= img.shape[1] else img.shape[1])
    print(min_size, max_size)

    size = min_size[0] if min_size[0] < min_size[1] else min_size[1]
    return [img[(img.shape[0] - size) // 2:size + (img.shape[0] - size) // 2, (img.shape[1] - size) // 2:size + (img.shape[1] - size) // 2] for img in images]


def normalize_and_register_retians(images: list, circles: list) -> list:
    avg_radius = sum([r for x, y, r in circles if r != 0]) // len(circles)
    print(f'INFO> Average radius of masks: {avg_radius}')

    cropped_images = []
    for img, circle in zip(images, circles):
        if sum(circle) == 0:
            continue
        x, y, r = circle
        cropped_images.append(img[y-r:y+r, x-r:x+r, :])

    cropped_images = [img for img in cropped_images if np.any(img) != 0]
    cropped_images = [cv2.resize(img, (avg_radius, avg_radius)) for img in cropped_images]
    return cropped_images


'''PCA test for segmentation'''
if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()
    sys.exit(0)

