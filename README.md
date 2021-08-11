# deform_conv2d_onnx_exporter

## Overview

At this moment, in August 2021, PyTorch 1.9.0 and torchvision 0.10.0 does not support exporting `deform_conv2d` into ONNX.

This module adds ONNX conversion of `deform_conv2d` to support the export.

This module implements Deformable Convolution v2, described in a paper, `Deformable ConvNets v2: More Deformable, Better Results <https://arxiv.org/abs/1811.11168>`, using ONNX operators.  
The implementation is straightforward, but may not be efficient.


