[![PyPI version](https://badge.fury.io/py/deform-conv2d-onnx-exporter.svg)](https://badge.fury.io/py/deform-conv2d-onnx-exporter) [![Test and Release](https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter/actions/workflows/ci.yml/badge.svg)](https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter/actions/workflows/ci.yml)

# deform\_conv2d\_onnx\_exporter

## Overview

This module enables you to export `deform_conv2d` to ONNX in PyTorch.

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
input_params = (
    torch.rand([1, x, x, x]),  # input
    torch.randn([1, x, x, x]), # offset
)
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
   $ python -m unittest discover -s tests
   ```

## Development notes

### Options for `deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()`

You can specify 2 options for this function.

* `use_gathernd`:  
  If `True`, use `GatherND` operator. Otherwise, use `GatherElements` operator.
* `enable_openvino_patch`:  
  If `True`, enable patch for OpenVINO.

### Referenced paper

This module implements Deformable Convolution v2, described in a paper, "[Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168)", using ONNX operators.

Some of the variable names in the module, such as `p` and `p_0`, are based on the paper.

### Memory layout of `offset`

The detail of `deform_conv2d` implementation in PyTorch is not fully documented.  
Therefore, I investigated the [implementation](https://github.com/pytorch/vision/blob/19ad0bbc5e26504a501b9be3f0345381d6ba1efc/torchvision/csrc/ops/cpu/deform_conv2d_kernel.cpp) to understand memory layout of some variables, such as `offset`.

* `offset`  
  The shape is `(batch, 2 * group * kernel_h * kernel_w, out_h, out_w)` according to the [reference](https://pytorch.org/vision/stable/ops.html#torchvision.ops.deform_conv2d).  
  The internal memory layout of `2 * group * kernel_h * kernel_w` is not clear.  
  According to the source code, it seems to be `(batch, group, kernel_h, kernel_w, 2, out_h, out_w)`.  
  The size `2` means "y-coords and x-coords".

### Padding of `input`

Even if `padding` is set to `0`, this module adds at least 1 padding internally.  
This is necessary to handle out-of-bounds `offset` appropriately.

### Performance

To be honest, the performance is not so good because the current version of ONNX, version 15, does not support `deform_conv2d` natively.  
Therefore, this module implements it using `GatherND` and other operators.  
As a result, the performance is not so good, but good enough for me.

Of course, I implemented this module carefully to reduce unnecessary or duplicated calculations.

### Opset version

Version **12** or later is required because of the following reasons:

- `Clip`:  
  Version 12 and later supports `Clip` operaetor for `tensor(int64)`. This module uses it.
- `GatherND`:  
  Versoin 12 and later supports `GatherND` operator with `batch_dims` attribute. This module also uses it.

## License

You can use this module under the MIT License.

Copyright 2021 Masamitsu MURASE

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
