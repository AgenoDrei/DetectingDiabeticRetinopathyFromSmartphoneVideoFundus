# Automatic Quality Assessment Images

## Overview

| Method | Description | Examples | Source Type | Source |
|--------|-------------|----------|-------------|--------|
| Sharpness | Using the variance of a laplace filter to get a measurement of the sharpness of a picture, can be easily thresholded. | Automatical removal of blurred camera images | Blog entry | [Link](https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/) |
| Sharpness | Paper describes 11 different ways to measure blurriness of a picture (Spoiler: Laplace based methods usally outperform other methods). | Theoretical | Paper | [Link](https://drive.google.com/open?id=0B1DEg5ii4MOPTnI2aU9IMGNxTEE) |
| Contrast | Using entropy of grayscale histogram to measure contrast, if the histogram has a higher entropy it also will have more contrast. | Detect low contrast images, measure contrast improvement through methods like CLAHE | Forum | [Link](https://stackoverflow.com/questions/13397394/how-to-measure-contrast-in-opencv-visual-c) |
| Brightness | Brightness can be measured through colorspaces. HSV splits colors in Hue, Saturation and Value. Value describes the brightness of your pixels. Average over all pixels and you have a good single value measurement. | Threshold dark images through color space conversion | Blog entry | [Link](https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/) |
| Color | Detect how colorful a image is through a custom color metric. | Easily detect highly saturated images | Blog Entry | [Link](https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/) |
| Color | Reasoning and proof for the color metric described above. | Theoretical | Paper | [Link](https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf) |
| Correction | Thresholding method to remove simple, clearling visible light glares from images. | Remove white reflections from images | Blog entry | [Link](https://towardsdatascience.com/anomaly-detection-in-images-777534980aeb) |
| Detection | Detect image with anomalies (for example wall cracks) with a Neural Network. Also use the network to compute an activation map showing the relevant parts of the image. | Detecting errors in pictures and pinpointing their locations | Blog entry | [Link](https://towardsdatascience.com/anomaly-detection-in-images-777534980aeb) |



