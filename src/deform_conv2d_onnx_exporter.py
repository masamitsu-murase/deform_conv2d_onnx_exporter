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


def create_clipping_mask(g, dcn_params, p, end):
    index_dtype_pytorch = dcn_params["index_dtype_pytorch"]
    return g.op("And",
                g.op("Greater", p, tensor(g, -1, dtype=index_dtype_pytorch)),
                g.op("Less", p, tensor(g, end, dtype=index_dtype_pytorch)))


def calculate_p_0(dcn_params):
    """Calculate p_0 value in equation (1) in the paper.
    """
    h = dcn_params["out_h"]
    w = dcn_params["out_w"]
    stride_h = dcn_params["stride_h"]
    stride_w = dcn_params["stride_w"]
    K = dcn_params["kernel_area_size"]

    p_0_y, p_0_x = torch.meshgrid(torch.arange(0, h * stride_h, stride_h),
                                  torch.arange(0, w * stride_w, stride_w))
    p_0_y = p_0_y.view(1, 1, 1, h, w).repeat(1, 1, K, 1, 1)
    p_0_x = p_0_x.view(1, 1, 1, h, w).repeat(1, 1, K, 1, 1)
    return p_0_y, p_0_x


def calculate_p_k(dcn_params):
    """Calculate p_k value in equation (1) in the paper.
    """
    kernel_h = dcn_params["kernel_h"]
    kernel_w = dcn_params["kernel_w"]
    dilation_h = dcn_params["dilation_h"]
    dilation_w = dcn_params["dilation_w"]
    K = dcn_params["kernel_area_size"]

    p_n_y, p_n_x = torch.meshgrid(
        torch.arange(0, kernel_h * dilation_h, step=dilation_h),
        torch.arange(0, kernel_w * dilation_w, step=dilation_w),
    )
    return p_n_y.reshape(1, 1, K, 1, 1), p_n_x.reshape(1, 1, K, 1, 1)


def calculate_p_helper(g, dcn_params, p_0, p_k, offset):
    b = dcn_params["batch"]
    K = dcn_params["kernel_area_size"]
    h = dcn_params["out_h"]
    w = dcn_params["out_w"]
    group = dcn_params["n_offset_grps"]
    offset_dtype = dcn_params["offset_dtype_pytorch"]

    p = p_0 + p_k
    # => p.shape is (1, 1, K, h, w)
    p = add(g, tensor(g, p.tolist(), dtype=offset_dtype), offset)
    p = g.op("Transpose", p, perm_i=[0, 1, 3, 4, 2])
    # => p.shape is (b, group, h, w, K)
    p = reshape(g, p, [b, group, 1, h, w, K])
    return p


def calculate_p(g, dcn_params, offset):
    """Calculate p_0 + p_k + Delta(p_k) in equation (1) in the paper.
    """
    b = dcn_params["batch"]
    K = dcn_params["kernel_area_size"]
    h = dcn_params["out_h"]
    w = dcn_params["out_w"]
    group = dcn_params["n_offset_grps"]

    offset = reshape(g, offset, [b, group, 2 * K, h, w])

    # Gather offset x and offset y.
    offset_y = slice(g, offset, [2], [0], [2 * K], steps=[2])
    offset_x = slice(g, offset, [2], [1], [2 * K + 1], steps=[2])

    p_0_y, p_0_x = calculate_p_0(dcn_params)
    p_k_y, p_k_x = calculate_p_k(dcn_params)

    p_y = calculate_p_helper(g, dcn_params, p_0_y, p_k_y, offset_y)
    p_x = calculate_p_helper(g, dcn_params, p_0_x, p_k_x, offset_x)
    return p_y, p_x


def gather_elements(g, dcn_params, input, p_y, p_x, mask_y, mask_x):
    """Gather elements specified p_y and p_x.
    """
    b = dcn_params["batch"]
    group = dcn_params["n_offset_grps"]
    h = dcn_params["in_h"]
    w = dcn_params["in_w"]
    ch = dcn_params["in_ch_per_group"]
    out_h = dcn_params["out_h"]
    out_w = dcn_params["out_w"]
    K = dcn_params["kernel_area_size"]
    index_dtype_onnx = dcn_params["index_dtype_onnx"]

    # index = add(g, mul(g, p_y, tensor(g, w, dtype=index_dtype_pytorch)), p_x)
    p_y = reshape(g, p_y, [b, group, out_h * out_w * K, 1])
    p_x = reshape(g, p_x, [b, group, out_h * out_w * K, 1])
    index = g.op("Concat", p_y, p_x, axis_i=3)
    mask_y = reshape(g, mask_y, [b, group, out_h * out_w * K, 1])
    mask_x = reshape(g, mask_x, [b, group, out_h * out_w * K, 1])
    mask = g.op("And", mask_y, mask_x)
    mask = g.op("Cast", mask, to_i=index_dtype_onnx)

    # If an index value is out of bounds, clear it to avoid error.
    index = mul(g, index, mask)
    # => index.shape is (b, group, out_h * out_w * K, 2)

    input = reshape(g, input, [b, group, ch, h, w])
    input = g.op("Transpose", input, perm_i=[0, 1, 3, 4, 2])
    # => input.shape is (b, group, h, w, ch)

    v = g.op("GatherND", input, index, batch_dims_i=2)
    # => v.shape is (b, group, out_h * out_w * K, ch)
    v = mul(g, v, mask)
    v = g.op("Transpose", v, perm_i=[0, 1, 3, 2])
    return reshape(g, v, [b, group, ch, out_h, out_w, K])


def gather_elements_org(g, dcn_params, input, p_y, p_x, mask_y, mask_x):
    """Gather elements specified p_y and p_x.
    """
    b = dcn_params["batch"]
    group = dcn_params["n_offset_grps"]
    h = dcn_params["in_h"]
    w = dcn_params["in_w"]
    ch = dcn_params["in_ch_per_group"]
    out_h = dcn_params["out_h"]
    out_w = dcn_params["out_w"]
    K = dcn_params["kernel_area_size"]
    index_dtype_onnx = dcn_params["index_dtype_onnx"]
    index_dtype_pytorch = dcn_params["index_dtype_pytorch"]

    index = add(g, mul(g, p_y, tensor(g, w, dtype=index_dtype_pytorch)), p_x)
    mask = g.op("And", mask_y, mask_x)
    mask = g.op("Cast", mask, to_i=index_dtype_onnx)

    # If an index value is out of bounds, clear it to avoid error.
    index = mul(g, index, mask)

    # index.shape is (b, group, 1, out_h, out_w, K)
    # Expand and reshape it to (b, group, ch, out_h * out_w * K)
    # to use GatherElements.
    index = g.op("Expand", index,
                 tensor(g, [b, group, ch, out_h, out_w, K], dtype=torch.int64))
    index = reshape(g, index, [b, group, ch, out_h * out_w * K])
    mask = reshape(g, mask, [b, group, 1, out_h * out_w * K])

    input = reshape(g, input, [b, group, ch, h * w])
    v = g.op("GatherElements", input, index, axis_i=3)
    v = mul(g, v, mask)  # Clear out-of-bounds values.
    v = reshape(g, v, [b, group, ch, out_h, out_w, K])
    return v


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


def calculate_p_tlbr(g, dcn_params, p_y, p_x):
    """Calculate floor and ceil of p.
    """
    offset_dtype = dcn_params["offset_dtype_pytorch"]

    # p_y/p_x.shape is (b, n_offset_grps, 1, out_h, out_w, kernel_area_size)
    p_t = g.op("Floor", p_y)
    p_l = g.op("Floor", p_x)
    one = tensor(g, 1, dtype=offset_dtype)

    p_b = add(g, p_t, one)
    p_r = add(g, p_l, one)

    return p_t, p_l, p_b, p_r


def calculate_ratio(g, dcn_params, p_y, p_x, p_t, p_l):
    """Calculate ratio value for bilinear interpolation.
    """
    offset_dtype = dcn_params["offset_dtype_pytorch"]

    one = tensor(g, 1.0, dtype=offset_dtype)

    diff_y = sub(g, p_y, p_t)
    diff_x = sub(g, p_x, p_l)
    diff_y_inv = sub(g, one, diff_y)
    diff_x_inv = sub(g, one, diff_x)

    # bilinear kernel (b, group, ch, out_h, out_w, kernel_area_size)
    # (1 - (p_x - p_l)) * (1 - (p_y - p_t))
    ratio_lt = mul(g, diff_x_inv, diff_y_inv)
    # (p_x - p_l) * (p_y - p_t)
    ratio_rb = mul(g, diff_x, diff_y)
    # (1 - (p_x - p_l)) * (p_y - p_t)
    ratio_lb = mul(g, diff_x_inv, diff_y)
    # (p_x - p_l) * (1 - (p_y - p_t))
    ratio_rt = mul(g, diff_x, diff_y_inv)

    return ratio_lt, ratio_rb, ratio_lb, ratio_rt


@sym_help.parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i",
                     "i", "i", "b")
def deform_conv2d(g, input, weight, offset, mask, bias, stride_h, stride_w,
                  pad_h, pad_w, dilation_h, dilation_w, n_weight_grps,
                  n_offset_grps, use_mask):
    if pad_h or pad_w:
        pad = tensor(g, [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w],
                     dtype=torch.int64)
        input_with_pad = g.op("Pad", input, pad, mode_s="constant")
    else:
        input_with_pad = input

    batch = get_tensor_dim_size(input, 0)
    in_ch = get_tensor_dim_size(input, 1)
    in_h = get_tensor_dim_size(input, 2) + 2 * pad_h
    in_w = get_tensor_dim_size(input, 3) + 2 * pad_w
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
    }

    p_y, p_x = calculate_p(g, dcn_params, offset)
    # => p_y/p_x.shape is (b, n_offset_grps, 1, out_h, out_w, kernel_area_size)

    p_t, p_l, p_b, p_r = calculate_p_tlbr(g, dcn_params, p_y, p_x)
    ratio_lt, ratio_rb, ratio_lb, ratio_rt = calculate_ratio(
        g, dcn_params, p_y, p_x, p_t, p_l)

    p_t = g.op("Cast", p_t, to_i=index_dtype_onnx)
    mask_t = create_clipping_mask(g, dcn_params, p_t, in_h)
    p_l = g.op("Cast", p_l, to_i=index_dtype_onnx)
    mask_l = create_clipping_mask(g, dcn_params, p_l, in_w)
    p_b = g.op("Cast", p_b, to_i=index_dtype_onnx)
    mask_b = create_clipping_mask(g, dcn_params, p_b, in_h)
    p_r = g.op("Cast", p_r, to_i=index_dtype_onnx)
    mask_r = create_clipping_mask(g, dcn_params, p_r, in_w)

    # (b, group, in_ch_per_group, out_h, out_w, kernel_area_size)
    v_lt = gather_elements(g, dcn_params, input_with_pad, p_t, p_l, mask_t,
                           mask_l)
    v_rb = gather_elements(g, dcn_params, input_with_pad, p_b, p_r, mask_b,
                           mask_r)
    v_lb = gather_elements(g, dcn_params, input_with_pad, p_b, p_l, mask_b,
                           mask_l)
    v_rt = gather_elements(g, dcn_params, input_with_pad, p_t, p_r, mask_t,
                           mask_r)

    weighted_v_lt = mul(g, ratio_lt, v_lt)
    weighted_v_rb = mul(g, ratio_rb, v_rb)
    weighted_v_lb = mul(g, ratio_lb, v_lb)
    weighted_v_rt = mul(g, ratio_rt, v_rt)
    v = g.op("Sum", weighted_v_lt, weighted_v_rb, weighted_v_lb, weighted_v_rt)
    # => v.shape is
    #    (batch, n_offset_grps, in_ch_per_group out_h, out_w, kernel_area_size)

    if use_mask:
        mask = reshape(
            g, mask, [batch, n_offset_grps, 1, kernel_area_size, out_h, out_w])
        mask = g.op("Transpose", mask, perm_i=[0, 1, 2, 4, 5, 3])
        v = mul(g, v, mask)

    v = reshape_v_for_conv(g, dcn_params, v)
    # => v.shape is (batch, in_ch, in_h * kernel_h, in_w * kernerl_w)
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
