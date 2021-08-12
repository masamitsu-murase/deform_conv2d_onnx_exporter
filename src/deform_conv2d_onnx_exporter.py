"""This module adds ONNX conversion of `deform_conv2d`.

This module implements Deformable Convolution v2,
described in a paper, `Deformable ConvNets v2: More Deformable, Better Results
<https://arxiv.org/abs/1811.11168>`, using ONNX operators.
The implementation is straightforward, but may not be efficient.
"""

import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx import symbolic_helper as sym_help

__all__ = ["register_deform_conv2d_op"]

onnx_opset_version = 11


def add(g, lhs, rhs):
    return g.op("Add", lhs, rhs)


def sub(g, lhs, rhs):
    return g.op("Sub", lhs, rhs)


def mul(g, lhs, rhs):
    return g.op("Mul", lhs, rhs)


def reshape(g, x, shape):
    if isinstance(shape, list):
        shape = tensor(g, shape, dtype=torch.int64)
    return g.op("Reshape", x, shape)


def slice(g, x, axes, starts, ends, *, steps=None):
    axes = tensor(g, axes, dtype=torch.int64)
    starts = tensor(g, starts, dtype=torch.int64)
    ends = tensor(g, ends, dtype=torch.int64)
    if steps is not None:
        steps = tensor(g, steps, dtype=torch.int64)
        return g.op("Slice", x, starts, ends, axes, steps)
    else:
        return g.op("Slice", x, starts, ends, axes)


def unsqueeze(g, input, dims):
    return sym_help._unsqueeze_helper(g, input, axes_i=dims)


def get_tensor_dim_size(tensor, dim):
    return sym_help._get_tensor_dim_size(tensor, dim)


def tensor(g, value, dtype):
    return g.op("Constant", value_t=torch.tensor(value, dtype=dtype))



def calculate_p_0(dcn_params):
    """Calculate p_0 value in equation (1) in the paper.
    """
    h = dcn_params["out_h"]
    w = dcn_params["out_w"]
    stride_h = dcn_params["stride_h"]
    stride_w = dcn_params["stride_w"]
    K = dcn_params["kernel_area_size"]
    additional_pad_h = dcn_params["additional_pad_h"]
    additional_pad_w = dcn_params["additional_pad_w"]

    p_0_y, p_0_x = torch.meshgrid(torch.arange(0, h * stride_h, stride_h),
                                  torch.arange(0, w * stride_w, stride_w))
    p_0_y = p_0_y.view(1, 1, 1, 1, h, w).repeat(1, 1, K, 1, 1, 1)
    p_0_y += additional_pad_h
    p_0_x = p_0_x.view(1, 1, 1, 1, h, w).repeat(1, 1, K, 1, 1, 1)
    p_0_x += additional_pad_w
    return torch.cat([p_0_y, p_0_x], dim=3)


def calculate_p_k(dcn_params):
    """Calculate p_k value in equation (1) in the paper.
    """
    kernel_h = dcn_params["kernel_h"]
    kernel_w = dcn_params["kernel_w"]
    dilation_h = dcn_params["dilation_h"]
    dilation_w = dcn_params["dilation_w"]
    K = dcn_params["kernel_area_size"]

    p_k_y, p_k_x = torch.meshgrid(
        torch.arange(0, kernel_h * dilation_h, step=dilation_h),
        torch.arange(0, kernel_w * dilation_w, step=dilation_w),
    )
    p_k_y = p_k_y.reshape(1, 1, K, 1, 1, 1)
    p_k_x = p_k_x.reshape(1, 1, K, 1, 1, 1)
    return torch.cat([p_k_y, p_k_x], dim=3)


def calculate_p(g, dcn_params, offset):
    """Calculate p_0 + p_k + Delta(p_k) in equation (1) in the paper.
    """
    b = dcn_params["batch"]
    K = dcn_params["kernel_area_size"]
    h = dcn_params["out_h"]
    w = dcn_params["out_w"]
    group = dcn_params["n_offset_grps"]
    offset_dtype = dcn_params["offset_dtype_pytorch"]

    offset = reshape(g, offset, [b, group, K, 2, h, w])

    p_0 = calculate_p_0(dcn_params)
    p_k = calculate_p_k(dcn_params)
    p = p_0 + p_k
    p = add(g, tensor(g, p.tolist(), dtype=offset_dtype), offset)
    # => p.shape is (b, group, K, 2, h, w)
    return p


def gather_elements(g, dcn_params, input, p_y, p_x):
    """Gather elements specified p_y and p_x.
    """
    b = dcn_params["batch"]
    group = dcn_params["n_offset_grps"]
    ch = dcn_params["in_ch_per_group"]
    out_h = dcn_params["out_h"]
    out_w = dcn_params["out_w"]
    K = dcn_params["kernel_area_size"]

    # If an index value is out of bounds, clear it to avoid error.
    p_y = reshape(g, p_y, [b, group, out_h * out_w * K, 1])
    p_x = reshape(g, p_x, [b, group, out_h * out_w * K, 1])
    index = g.op("Concat", p_y, p_x, axis_i=3)
    # => index.shape is (b, group, out_h * out_w * K, 2)

    v = g.op("GatherND", input, index, batch_dims_i=2)
    # => v.shape is (b, group, out_h * out_w * K, ch)
    v = g.op("Transpose", v, perm_i=[0, 1, 3, 2])
    return reshape(g, v, [b, group, ch, K, out_h, out_w])


def gather_elements_tlbr(g, dcn_params, input, p_tlbr):
    patterns = ["tl", "br", "bl", "tr"]
    v_tlbr = {}
    for key in patterns:
        key_y = key[0]  # "t" or "b"
        key_x = key[1]  # "l" or "r"
        p_y = p_tlbr[key_y]
        p_x = p_tlbr[key_x]
        v = gather_elements(g, dcn_params, input, p_y, p_x)
        v_tlbr[key] = v
    return v_tlbr



def reshape_v_for_conv(g, dcn_params, v):
    """Reshape v for convolution.

    e.g. If kernel_size is 3, reshape
       1  2  3  4  5  6  7  8  9
      10 11 12 13 14 15 16 17 18
      19 20 21 22 23 24 25 26 27
      ...
    to
       1  2  3 10 11 12 19 20 21
       4  5  6 13 14 15 22 23 24
       7  8  9 16 17 18 25 26 27
      28 29 30 37 38 39 46 47 48
      ...
    """
    b = dcn_params["batch"]
    h = dcn_params["out_h"]
    w = dcn_params["out_w"]
    ch = dcn_params["in_ch"]
    kernel_h = dcn_params["kernel_h"]
    kernel_w = dcn_params["kernel_w"]
    K = dcn_params["kernel_area_size"]

    #    (batch, n_offset_grps, in_ch_per_group, K, out_h, out_w)
    v = g.op("Transpose", v, perm_i=[0, 1, 2, 4, 5, 3])

    if kernel_h == 1:
        # Split returnes not list of tensor but a tensor
        # if kernel_h == 1.
        return reshape(g, v, [b, ch, h * kernel_h, w * kernel_w])

    v = reshape(g, v, [b, ch, h, w, K])
    items = g.op("Split",
                 v,
                 split_i=[kernel_w] * kernel_h,
                 axis_i=4,
                 outputs=kernel_h)
    shape = tensor(g, [b, ch, h, w * kernel_w], dtype=torch.int64)
    items = [reshape(g, item, shape) for item in items]
    concatnated_v = g.op("Concat", *items, axis_i=3)
    return reshape(g, concatnated_v, [b, ch, h * kernel_h, w * kernel_w])


def calculate_p_tl(g, dcn_params, p):
    """Calculate floor of p.
    """
    p_tl = g.op("Floor", p)
    return p_tl


def calculate_p_tlbr(g, dcn_params, p_tl):
    """Calculate floor and ceil of p.
    """
    h = dcn_params["in_h"]
    w = dcn_params["in_w"]
    index_dtype_onnx = dcn_params["index_dtype_onnx"]
    index_dtype_pytorch = dcn_params["index_dtype_pytorch"]

    p_tl = g.op("Cast", p_tl, to_i=index_dtype_onnx)
    one = tensor(g, 1, dtype=index_dtype_pytorch)

    p_t = slice(g, p_tl, [3], [0], [1])
    p_l = slice(g, p_tl, [3], [1], [2])
    p_b = add(g, p_t, one)
    p_r = add(g, p_l, one)

    # Clip out-of-bounds coords.
    # Clipped coords point to padding area, which is filled with 0.
    p_t = g.op("Clip", p_t, tensor(g, 0, dtype=index_dtype_pytorch),
               tensor(g, h - 1, dtype=index_dtype_pytorch))
    p_l = g.op("Clip", p_l, tensor(g, 0, dtype=index_dtype_pytorch),
               tensor(g, w - 1, dtype=index_dtype_pytorch))
    p_b = g.op("Clip", p_b, tensor(g, 0, dtype=index_dtype_pytorch),
               tensor(g, h - 1, dtype=index_dtype_pytorch))
    p_r = g.op("Clip", p_r, tensor(g, 0, dtype=index_dtype_pytorch),
               tensor(g, w - 1, dtype=index_dtype_pytorch))
    return {
        "t": p_t,
        "l": p_l,
        "b": p_b,
        "r": p_r,
    }


def calculate_weight(g, dcn_params, p, p_tl):
    """Calculate weight value for bilinear interpolation.
    """
    b = dcn_params["batch"]
    group = dcn_params["n_offset_grps"]
    h = dcn_params["out_h"]
    w = dcn_params["out_w"]
    K = dcn_params["kernel_area_size"]
    offset_dtype = dcn_params["offset_dtype_pytorch"]

    one = tensor(g, 1, dtype=offset_dtype)

    diff = sub(g, p, p_tl)
    diff_y = slice(g, diff, [3], [0], [1])
    diff_x = slice(g, diff, [3], [1], [2])
    diff_y_inv = sub(g, one, diff_y)
    diff_x_inv = sub(g, one, diff_x)

    # bilinear kernel (b, group, K, 1, h, w)
    # (1 - (p_x - p_l)) * (1 - (p_y - p_t))
    weight_tl = mul(g, diff_x_inv, diff_y_inv)
    weight_tl = reshape(g, weight_tl, [b, group, 1, K, h, w])
    # (p_x - p_l) * (p_y - p_t)
    weight_br = mul(g, diff_x, diff_y)
    weight_br = reshape(g, weight_br, [b, group, 1, K, h, w])
    # (1 - (p_x - p_l)) * (p_y - p_t)
    weight_bl = mul(g, diff_x_inv, diff_y)
    weight_bl = reshape(g, weight_bl, [b, group, 1, K, h, w])
    # (p_x - p_l) * (1 - (p_y - p_t))
    weight_tr = mul(g, diff_x, diff_y_inv)
    weight_tr = reshape(g, weight_tr, [b, group, 1, K, h, w])

    return {
        "tl": weight_tl,
        "br": weight_br,
        "bl": weight_bl,
        "tr": weight_tr,
    }


@sym_help.parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i",
                     "i", "i", "b")
def deform_conv2d(g, input, weight, offset, mask, bias, stride_h, stride_w,
                  pad_h, pad_w, dilation_h, dilation_w, n_weight_grps,
                  n_offset_grps, use_mask):
    additional_pad_h = additional_pad_w = 0
    if pad_h == 0:
        additional_pad_h = 1
    if pad_w == 0:
        additional_pad_w = 1

    batch = get_tensor_dim_size(input, 0)
    in_ch = get_tensor_dim_size(input, 1)
    in_h = get_tensor_dim_size(input, 2) + 2 * (pad_h + additional_pad_h)
    in_w = get_tensor_dim_size(input, 3) + 2 * (pad_w + additional_pad_w)
    in_ch_per_group = in_ch // n_offset_grps

    out_ch = get_tensor_dim_size(weight, 0)
    kernel_h = get_tensor_dim_size(weight, 2)
    kernel_w = get_tensor_dim_size(weight, 3)
    kernel_area_size = kernel_h * kernel_w

    out_h = get_tensor_dim_size(offset, 2)
    out_w = get_tensor_dim_size(offset, 3)

    offset_dtype = sym_help._try_get_scalar_type(offset)
    offset_dtype_onnx = sym_help.cast_pytorch_to_onnx[offset_dtype]
    dtype_index = sym_help.scalar_type_to_onnx.index(offset_dtype_onnx)
    offset_dtype_pytorch = sym_help.scalar_type_to_pytorch_type[dtype_index]

    index_dtype = "Long"
    index_dtype_onnx = sym_help.cast_pytorch_to_onnx[index_dtype]
    dtype_index = sym_help.scalar_type_to_onnx.index(index_dtype_onnx)
    index_dtype_pytorch = sym_help.scalar_type_to_pytorch_type[dtype_index]

    dcn_params = {
        # batch and kernel
        "batch": batch,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "kernel_area_size": kernel_area_size,

        # input size
        "in_ch": in_ch,
        "in_ch_per_group": in_ch_per_group,
        "in_h": in_h,
        "in_w": in_w,

        # output size
        "out_ch": out_ch,
        "out_h": out_h,
        "out_w": out_w,

        # other parameters
        "stride_h": stride_h,
        "stride_w": stride_w,
        "dilation_h": dilation_h,
        "dilation_w": dilation_w,
        "n_offset_grps": n_offset_grps,

        # offset data type
        "offset_dtype": offset_dtype,
        "offset_dtype_onnx": offset_dtype_onnx,
        "offset_dtype_pytorch": offset_dtype_pytorch,

        # index data type
        "index_dtype": index_dtype,
        "index_dtype_onnx": index_dtype_onnx,
        "index_dtype_pytorch": index_dtype_pytorch,

        # additional pads
        "additional_pad_h": additional_pad_h,
        "additional_pad_w": additional_pad_w,
    }

    pad_size = [
        0,
        0,
        (pad_h + additional_pad_h),
        (pad_w + additional_pad_w),
        0,
        0,
        (pad_h + additional_pad_h),
        (pad_w + additional_pad_w),
    ]
    pad = tensor(g, pad_size, dtype=torch.int64)
    input = g.op("Pad", input, pad, mode_s="constant")

    p = calculate_p(g, dcn_params, offset)
    # p.shape is (b, n_offset_grps, kernel_area_size, 2, out_h, out_w)

    p_tl = calculate_p_tl(g, dcn_params, p)
    p_tlbr = calculate_p_tlbr(g, dcn_params, p_tl)
    weight_tlbr = calculate_weight(g, dcn_params, p, p_tl)
    # ratio_tlbr.shape is (b, n_offset_grps, 1, kernel_area_size, out_h, out_w)

    input = reshape(g, input,
                    [batch, n_offset_grps, in_ch_per_group, in_h, in_w])
    input = g.op("Transpose", input, perm_i=[0, 1, 3, 4, 2])
    # input.shape is (b, n_offset_grps, in_h, in_w, in_ch_per_group)

    v_tlbr = gather_elements_tlbr(g, dcn_params, input, p_tlbr)
    # v_tlbr[x].shape is
    #  (batch, n_offset_grps, in_ch_per_group, K, out_h, out_w)

    weighted_v_list = [mul(g, weight_tlbr[key], v_tlbr[key]) for key in v_tlbr]

    v = g.op("Sum", *weighted_v_list)
    #  v.shape is
    #   (batch, n_offset_grps, in_ch_per_group, K, out_h, out_w)

    if use_mask:
        mask = reshape(
            g, mask, [batch, n_offset_grps, 1, kernel_area_size, out_h, out_w])
        v = mul(g, v, mask)

    v = reshape_v_for_conv(g, dcn_params, v)
    # v.shape is (batch, in_ch, out_h * kernel_h, out_w * kernerl_w)
    output = g.op("Conv",
                  v,
                  weight,
                  group_i=n_weight_grps,
                  kernel_shape_i=[kernel_h, kernel_w],
                  strides_i=[kernel_h, kernel_w])
    bias = unsqueeze(g, bias, [0, 2, 3])
    output = add(g, output, bias)
    return output


def register_deform_conv2d_op():
    """Register custom operator for torchvision::deform_conv2d.
    """
    register_custom_op_symbolic('torchvision::deform_conv2d', deform_conv2d,
                                onnx_opset_version)
