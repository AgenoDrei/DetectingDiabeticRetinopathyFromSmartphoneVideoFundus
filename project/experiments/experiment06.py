import numpy as np
import cv2
import sys
sys.path.append('/data/simon/Code/MasterThesis/project/include')
from utils import  load_image, show_image, enhance_contrast_image, show_image_row, float2gray

np.set_printoptions(threshold=sys.maxsize)

def run() -> None:
    #image = load_image(path='./C001R_Cut/C001R08.jpg')
    image = load_image('/data/simon/ownCloud/Data/SFBI Studie/optimal_quality_control/DN0687.Zeiss.jpg')
    #image = load_image('/home/simon/Downloads/Retinal-blood-vessel-segmentation-in-high-resolution-fundus-images-a-Fundus.jpg')
    show_image(image, 'Raw')
    top = (image.shape[1] - image.shape[0]) // 2
    image = image[:, top:top+image.shape[0]]

    #blue, blue_mean, blue_dev = extract_background_pixel(image[:, :, 0], grid_size=10, t=0.95)
    green, green_mean, green_dev = extract_background_pixel(image[:, :, 1], grid_size=10, t=0.95)
    #red, red_mean, red_dev = extract_background_pixel(image[:, :, 2], grid_size=10, t=0.95)

    #result = cv2.merge([float2gray(blue_dev * blue + blue_mean), float2gray(green_dev * green + green_mean),
    #                     float2gray(red * red_dev + red_mean)])
    #show_image_row([image, result], 'Result')


def extract_background_pixel(img:np.array, grid_size:int=10, t:float=1) -> np.array:
    #img = cv2.copyMakeBorder(img, 500, 500, 500, 500, borderType=cv2.BORDER_CONSTANT, value=0, dst=img)
    img = np.float64(img)
    img = cv2.normalize(img, img, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    print('INFO> Cropped image dimensions: ', img.shape)

    bg_img = np.zeros(img.shape, dtype=np.float64)
    mean_img, dev_img = estimate_mean_dev(img, size=grid_size)

    #show_image_row([float2gray(mean_img), float2gray(dev_img)], name='mean - dev')

    dist_img = cv2.absdiff(img, mean_img)
    dist_img = cv2.divide(dist_img, dev_img)

    mask = dist_img <= t
    bg_img[mask] = 1.0
    show_image(float2gray(bg_img), name='background')

    bg_mean_img, bg_dev_img = estimate_mean_dev(img, mask=bg_img, size=grid_size)
    #print(bg_mean_img.max(), bg_mean_img.min(), bg_dev_img.max(), bg_mean_img.min())
    #show_image_row([float2gray(bg_mean_img), float2gray(bg_dev_img)], name='mean - dev')
    enhanced_img = cv2.subtract(img, bg_mean_img)
    enhanced_img = cv2.divide(enhanced_img, bg_dev_img)
    mask = enhanced_img > 5.0
    enhanced_img[mask] = 5.0
    mask = enhanced_img < -5.0
    enhanced_img[mask] = -5.0
    show_image_row([float2gray(enhanced_img), float2gray(cv2.addWeighted(img, 0.9, bg_img, 0.1, 0))], name='enhanced')

    return enhanced_img, mean_img, dev_img


def estimate_mean_dev(img:np.array, mask: np.array = None, size: int = 10):
    if mask is None:
        mask = np.ones(img.shape)
    step_size = img.shape[0] / size
    print('Sample size: ', step_size ** 2)
    mean_img = np.zeros((size, size), dtype=np.float64)
    dev_img = np.zeros((size, size), dtype=np.float64)

    for y in np.arange(step_size / 2, img.shape[0], step_size):
        for x in np.arange(step_size / 2, img.shape[1], step_size):
            sample = img[int(y - step_size / 2):int(y + step_size / 2), int(x - step_size / 2):int(x + step_size / 2)]
            sample_mask = mask[int(y - step_size / 2):int(y + step_size / 2), int(x - step_size / 2):int(x + step_size / 2)]
            #   show_image_row([float2gray(sample), sample_mask], f'Sample Pos({y}|{x}) Size({step_size})', time=300)
            mean, dev = cv2.meanStdDev(sample[sample_mask.astype('bool')])
            mean_img[int(y / step_size), int(x / step_size)] = mean
            dev_img[int(y / step_size), int(x / step_size)] = dev

    mean_img = cv2.resize(mean_img, img.shape, interpolation=cv2.INTER_CUBIC)
    dev_img = cv2.resize(dev_img, img.shape, interpolation=cv2.INTER_CUBIC)
    return mean_img, dev_img


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()
    sys.exit(0)