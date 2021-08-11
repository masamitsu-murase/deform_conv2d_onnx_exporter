import io
import random
import unittest

import numpy as np
import onnxruntime
import torch
import torch.onnx
from torchvision.ops.deform_conv import DeformConv2d

import deform_conv2d_onnx_exporter

deform_conv2d_onnx_exporter.register_deform_conv2d_op()


def tonumpy(tensor):
    return tensor.to('cpu').detach().numpy().copy()


class DeformConv2dOpTestCase(unittest.TestCase):
    OPSET_VERSION = 11

    def create_input_params(self, dcn_params):
        size = [
            dcn_params["batch"],
            dcn_params["input_ch"],
            dcn_params["input_h"],
            dcn_params["input_w"],
        ]
        input = torch.rand(size, dtype=torch.float)

        size = [
            dcn_params["batch"],
            2 * dcn_params["n_offset_grps"] * dcn_params["kernel_h"] *
            dcn_params["kernel_w"],
            dcn_params["output_h"],
            dcn_params["output_w"],
        ]
        offset = torch.randn(size, dtype=torch.float) * dcn_params["kernel_w"]

        size = [
            dcn_params["batch"],
            dcn_params["n_offset_grps"] * dcn_params["kernel_h"] *
            dcn_params["kernel_w"],
            dcn_params["output_h"],
            dcn_params["output_w"],
        ]
        mask = torch.rand(size, dtype=torch.float)

        return input, offset, mask

    def create_pytorch_model(self, dcn_params):
        model = DeformConv2d(
            in_channels=dcn_params["input_ch"],
            out_channels=dcn_params["output_ch"],
            kernel_size=(dcn_params["kernel_h"], dcn_params["kernel_w"]),
            stride=(dcn_params["stride_h"], dcn_params["stride_w"]),
            padding=(dcn_params["padding_h"], dcn_params["padding_w"]),
            dilation=(dcn_params["dilation_h"], dcn_params["dilation_w"]),
            groups=dcn_params["groups"],
            bias=dcn_params["bias"],
        )
        return model

    def convert_to_onnx_model(self, pytorch_model, input, offset, mask=None):
        if mask is not None:
            input_params = (input, offset, mask)
            input_names = ["input", "offset", "mask"]
        else:
            input_params = (input, offset)
            input_names = ["input", "offset"]
        buffer = io.BytesIO()
        torch.onnx.export(pytorch_model,
                          input_params,
                          buffer,
                          input_names=input_names,
                          output_names=["output"],
                          opset_version=self.OPSET_VERSION)
        onnx_model = onnxruntime.InferenceSession(buffer.getvalue())
        return onnx_model

    def run_pytorch_model(self, model, input, offset, mask=None):
        return model(input, offset, mask)

    def run_onnx_model(self, model, input, offset, mask=None):
        input_params = {
            "input": tonumpy(input),
            "offset": tonumpy(offset),
        }
        if mask is not None:
            input_params["mask"] = tonumpy(mask)
        return model.run(["output"], input_params)[0]

    def run_with_dcn_params(self, dcn_params, message=""):
        print(dcn_params)
        input, offset, mask = self.create_input_params(dcn_params)
        pytorch_model = self.create_pytorch_model(dcn_params)
        if dcn_params["use_mask"]:
            pytorch_output = self.run_pytorch_model(pytorch_model, input,
                                                    offset, mask)
            onnx_model = self.convert_to_onnx_model(pytorch_model, input,
                                                    offset, mask)
            onnx_output = self.run_onnx_model(onnx_model, input, offset, mask)
        else:
            pytorch_output = self.run_pytorch_model(pytorch_model, input,
                                                    offset)
            onnx_model = self.convert_to_onnx_model(pytorch_model, input,
                                                    offset)
            onnx_output = self.run_onnx_model(onnx_model, input, offset)

        pytorch_output_np = tonumpy(pytorch_output)
        self.assertTrue(
            np.allclose(pytorch_output_np, onnx_output, rtol=1e-03,
                        atol=1e-05), message)

    def test_full_parameters(self):
        dcn_params = {
            "batch": 3,
            "input_ch": 32,
            "input_h": 224,
            "input_w": 200,
            "output_ch": 16,
            "output_h": 112,
            "output_w": 66,
            "kernel_h": 3,
            "kernel_w": 4,
            "stride_h": 2,
            "stride_w": 3,
            "padding_h": 1,
            "padding_w": 2,
            "dilation_h": 1,
            "dilation_w": 2,
            "groups": 2,
            "n_offset_grps": 2,
            "bias": True,
            "use_mask": True,
        }
        self.run_with_dcn_params(dcn_params)

    def generate_random_parameters(self):
        dcn_params = {
            "batch": random.randrange(1, 6),
            # "input_ch": 0,
            "input_h": random.randrange(100, 201),
            "input_w": random.randrange(100, 201),
            # "output_ch": 0,
            # "output_h": 0,
            # "output_w": 0,
            "kernel_h": random.randrange(1, 8),
            "kernel_w": random.randrange(1, 8),
            "stride_h": random.randrange(1, 5),
            "stride_w": random.randrange(1, 5),
            "padding_h": random.randrange(1, 5),
            "padding_w": random.randrange(1, 5),
            "dilation_h": random.randrange(1, 4),
            "dilation_w": random.randrange(1, 4),
            "groups": random.randrange(1, 4),
            "n_offset_grps": random.randrange(1, 4),
            "bias": random.choice([True, False]),
            "use_mask": random.choice([True, False]),
        }
        dcn_params["input_ch"] = dcn_params["groups"] * dcn_params[
            "n_offset_grps"] * random.randrange(1, 17)
        dcn_params["output_ch"] = dcn_params["groups"] * random.randrange(
            1, 17)
        ker_h = dcn_params["dilation_h"] * (dcn_params["kernel_h"] - 1) + 1
        dcn_params["output_h"] = (
            (dcn_params["input_h"] + 2 * dcn_params["padding_h"] - ker_h) //
            dcn_params["stride_h"]) + 1
        ker_w = dcn_params["dilation_w"] * (dcn_params["kernel_w"] - 1) + 1
        dcn_params["output_w"] = (
            (dcn_params["input_w"] + 2 * dcn_params["padding_w"] - ker_w) //
            dcn_params["stride_w"]) + 1
        return dcn_params

    def test_random_parameters(self):
        for _ in range(100):
            dcn_params = self.generate_random_parameters()
            self.run_with_dcn_params(dcn_params)
