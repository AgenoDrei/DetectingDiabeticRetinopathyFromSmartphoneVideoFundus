# Automatic Quality Assessment Images

## Image Blur / Sharpness
- Use the variance of a laplace-based filter to get a single value measurement for the "edginess" of an image: [Blog entry](https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)
- Paper discusses different methods for focus measurements and compares them: [Paper](https://drive.google.com/open?id=0B1DEg5ii4MOPTnI2aU9IMGNxTEE)

## Artifacts / Anomalies
- Global anomaly detection with CNNs [Blog entry](https://towardsdatascience.com/anomaly-detection-in-images-777534980aeb)
- Remove simple light glares from images [Blog entry](http://www.amphident.de/en/blog/preprocessing-for-automatic-pattern-identification-in-wildlife-removing-glare.html)
- One-class anomaly detection


## Brightness / Contrast

- Using entropy of grayscale histogram to measure contrast: [Stackoverflow](https://stackoverflow.com/questions/13397394/how-to-measure-contrast-in-opencv-visual-c)
- Brightness can be measured through different colorspaces. HSV splits colors in Hue, Saturation and Value. Value describes the brightness of your pixels. Average over all your pixels and you have a good single value measurement
- Colorfullness of an image: [Paper](https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf), [Blog entry](https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/)
