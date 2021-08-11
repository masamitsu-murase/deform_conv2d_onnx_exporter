# deform_conv2d_onnx_exporter

## Overview

This module enables to export `deform_conv2d` in PyTorch.

At this moment, in August 2021, PyTorch 1.9.0 and torchvision 0.10.0 does not support exporting `deform_conv2d` into ONNX, so I implemented this module.

This module implements Deformable Convolution v2, described in a paper, `Deformable ConvNets v2: More Deformable, Better Results <https://arxiv.org/abs/1811.11168>`, using ONNX operators.  
The implementation is straightforward, but may not be efficient.

## Installation

```sh
$ pip install deform_conv2d_onnx_exporter
```

## Usage

```python
import torch.onnx
from torchvision.ops.deform_conv import DeformConv2d
import deform_conv2d_onnx_exporter

deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()

model = DeformConv2d(...)
input_names = ["input", "offset"]
output_names = ["output"]
torch.onnx.export(model,
                  input_params,
                  "output.onnx",
                  input_names=input_names,
                  output_names=output_names,
                  opset_version=12)
```

Note that you have to set `opset_version` to `12` or later.

## Tests

1. Install dependent libraries.  
   ```sh
   $ pip install -r requirements.txt
   ```
2. Run `unittest`.  
   ```sh
   $ python -m unittest tests/test_deform_conv2d_onnx_exporter.py
   ```

## License

You can use this module under the MIT License.

Copyright 2021 Masamitsu MURASE

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
