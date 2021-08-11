from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='deform_conv2d_onnx_exporter',
    version='1.0.0',
    description='A library to support onnx export of deform_conv2d in PyTorch.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter',
    author='Masamitsu MURASE',
    author_email='masamitsu.murase@gmail.com',
    license='MIT',
    keywords='deform_conv2d PyTorch ONNX',
    py_modules=["deform_conv2d_onnx_exporter"],
    package_dir={"": "src"},
    zip_safe=True,
    python_requires='>=3.6.*, <4',
    install_requires=['torch>=1.8.0', 'torchvision>=0.9.0'],
    project_urls={
        'Bug Reports':
        'https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter/issues',
        'Source': 'https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter',
    },
)
