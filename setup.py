from setuptools import setup
import os

VERSION = "1.1"


def get_long_description():
    with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
            encoding="utf8",
    ) as fp:
        return fp.read()


setup(
    name="SmartphoneDR",
    description="Detecting Diabetic Retinopathy from Smartphone Fundus Videos",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Simon Müller",
    url="https://github.com/AgenoDrei/DetectingDiabeticRetinopathyFromSmartphoneVideoFundus",
    project_urls={
        "Issues": "https://github.com/AgenoDrei/DetectingDiabeticRetinopathyFromSmartphoneVideoFundus/issues",
        "CI": "https://github.com/AgenoDrei/DetectingDiabeticRetinopathyFromSmartphoneVideoFundus/actions",
        "Changelog": "https://github.com/AgenoDrei/DetectingDiabeticRetinopathyFromSmartphoneVideoFundus/releases",
    },
    license="Apache License, Version 2.0",
    version=VERSION,
    packages=["experiments", "include", "scripts"],
    include_package_data=True,
    entry_points="""
        [console_scripts]
        smartphone-dr=scripts.cli:cli
    """,
    install_requires=["click", "pandas", "openpyxl", "requests", "jinja2", "importlib_resources", "tqdm", "numpy",
                      "toml", "scikit-learn", "scikit-video", "scikit-image", "opencv-contrib-python", "albumentations",
                      "scipy", "torch", "tensorboard", "torchvision", "pretrainedmodels", "joblib", "mahotas"],
    extras_require={
        "test": ["pytest"]
    },
    python_requires=">=3.6",
)
