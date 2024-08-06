from setuptools import setup, find_packages
setup(
    name='Kazier',
    version='0.0.1',
    packages=find_packages(),
    author='Sumit Kumar Singh',
    author_email='sumitsingh9441@gmail.com',
    description='A comprehensive package for face detection, hand tracking, pose estimation, and more using MediaPipe, designed to simplify your project development.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RAJPUTRoCkStAr/Kazier.git',
     keywords=[
        'computer vision', 
        'mediapipe', 
        'face detection', 
        'hand tracking', 
        'pose estimation', 
        'opencv', 
        'image processing', 
        'machine learning', 
        'AI', 
        'artificial intelligence'
    ],
    install_requires=[
        'numpy','opencv-python','mediapipe','requests'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)