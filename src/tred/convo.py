#!/usr/bin/env python
'''
Support for DFT-based convolutions.

'''

from .util import to_tuple, to_tensor, tenstr, getattr_first, deprecated
from .types import IntTensor, Tensor, Shape, index_dtype
from .blocking import Block, batchify
from .partitioning import deinterlace, deinterlace_pairs
import torch
from torch.nn.functional import pad

def dft_shape(tshape: Shape, kshape: Shape) -> Shape:
    '''
    Return the shape required to convolve tensor of shape tshape with kernel of shape kshape.

    The tensor is considered batched in the first dimension if its size is one
    more than the size of kshape.

    A tuple is returned.
    '''
    device = getattr_first("device", tshape, kshape)
    tshape = to_tensor(tshape, device=device)
    kshape = to_tensor(kshape, device=device)

    if len(tshape) == len(kshape) + 1:
        tshape = tshape[1:]     # remove batching

    if len(tshape) != len(kshape):
        raise ValueError(f'shape mismatch {tshape=} != {kshape=} (after possible unbatching of tshape)')


    return to_tuple(tshape + kshape - 1)


def zero_pad(ten : Tensor, shape: Shape|None = None) -> Tensor:
    '''
    Zero-pad tensor ten to be of given shape.

    If the number of tensor dimensions of ten is one more than the length of
    shape then an extra first dimension is assumed to run along a batch.

    Zero-padding is applied to the high-side of each non-batch dimension.
    '''
    batched = True
    if len(shape) == len(ten.shape):
        batched = False
        ten = torch.unsqueeze(ten, 0) # add batch dim.

    have = to_tensor(ten.shape[1:], dtype=index_dtype, device=ten.device)
    want = to_tensor(shape, dtype=index_dtype, device=ten.device)
    zzzz = torch.zeros_like(want)

    diff = want - have
    padding = torch.vstack((diff, zzzz)).T.flatten().tolist()
    padding.reverse()
    padded = pad(ten, tuple(padding))
    if not batched:
        padded = padded[0]
    return padded


@deprecated('Please use zero_pad instead.')
def symmetric_pad(ten: Tensor, shape: Shape, symmetry: tuple) -> Tensor:
    '''
    Zero-pad tensor to shape in a symmetric fashion.

    - ten :: an N-dimensional tensor or N+1 if batched.

    - shape :: an N-tensor giving desired sizes of the padded tensor.  Each
      dimension size must at least that of the corresponding input size.

    - symmetry :: a tuple of symmetry types corresponding to each (non-batch)
      dimension.

    Supported symmetry types that govern where the zero padding is placed
    w.r.t. the input tensor dimension.  For tensor dimension size n_t and pad
    dimension size n_p, the layouts are also given.

    - prepend :: padding is inserted on the low side of the dimension.
                 Layout: [n_p, n_t].

    - append :: padding is inserted on the high side of the dimension.
                Layout: [n_t, n_p].

    - center :: padding is inserted between two "halves" of the dimension.
                Layout: [front_half(n_t), n_p, back_half(n_t)]

    - edge :: "half" the padding is prepended, "half" appended
               Layout: [front_half(n_p), n_t, back_half(n_p)]

    When the argument is odd, the front_half() and back_half() give the "extra"
    element to the front.

    Note, tred applies "center" padding for spatial dimensions to a "response"
    and "edge" padding to spatial dimensions of a "signal" and "append" padding
    to time/drift dimension of either.
    '''

    # input validation
    if len(symmetry) != len(shape):
        raise ValueError(f'symmetric_pad: shape and symmetry must be same size got {len(symmetry)} != {len(shape)}')

    if not all([s in 'prepend append center edge'.split() for s in symmetry]):
        raise ValueError(f'symmetric_pad: unsupported symmetries in {symmetry}')

    ten, squeeze = batchify(ten, len(shape))
    o_shape = to_tensor(ten.shape[1:], device=ten.device)
    vdim = len(o_shape)
    if not all([t < s for t,s in zip(o_shape, shape)]):
        raise ValueError(f'symmetric_pad: truncation {o_shape} to {shape} is not supported')

    # We will always apply the zero padding as an "append" and achieve the
    # desired symmetry by rolling the dimension before and/or after the padding.
    pre_shift = [0]*vdim
    post_shift = [0]*vdim
    for idim, sym in enumerate(symmetry): # iterate over volume dimensions

        if sym == "append":
            continue
        # all others require some kind of shift

        # the original size of the dimension
        o_siz = o_shape[idim]
        # the final target size
        f_siz = shape[idim]
        # the amount of padding to add
        p_siz = f_siz - o_siz

        if sym == "prepend":
            # shift the appended padding to the front
            pre_shift[idim] = 0
            post_shift[idim] = p_siz
            continue

        if sym == "center":
            # Shift the latter half of the tensor to the front, pad, shift back.
            # This leaves the "1+n//2" prior to the central padding.
            half = o_siz//2
            pre_shift[idim] = half
            post_shift[idim] = -half
            continue

        if sym == "edge":
            # Append, then shift half the padding to the front.  This puts the
            # tensor data following "1+n//2" padding.
            half = (1+p_siz)//2
            pre_shift[idim] = 0
            post_shift[idim] = half
            continue

    # Only apply shifts to non-batch dimensions
    dims = to_tuple(torch.arange(vdim) + 1)

    if any(pre_shift):
        ten = torch.roll(ten, shifts=tuple(pre_shift), dims=dims)

    ten = zero_pad(ten, shape)

    if any(post_shift):
        ten = torch.roll(ten, shifts=tuple(post_shift), dims=dims)

    if squeeze:
        ten = ten.squeeze(0)

    return ten


@deprecated("Not used in interlaced_symm_v2")
def response_pad(response: Tensor, shape: Shape) -> Tensor:
    '''
    Apply tred "response style" padding to the tensor.
    '''
    return zero_pad(response, shape)


@deprecated("Not used in interlaced_symm_v2")
def signal_pad(signal: Block, shape: Shape, taxis: int = -1) -> Block:
    '''
    Apply tred "signal style" padding to the block.

    The returned block has its location adjusted to reflect the edge type
    padding so that the original signal content remains at its original
    location.
    '''
    data = zero_pad(signal.data, shape)
    # signal is batched
    nrm1 = [i - j for i,j in zip(shape, signal.data.size()[1:])] # Nr - 1
    nrm1[taxis] = 0
    if any(x % 2 != 0 for x in nrm1):
        raise ValueError(f"Length of response tensor must always be odd. {nrm1} + 1 is given.")
    nrm1 = to_tensor(nrm1, device=signal.data.device)
    fh = nrm1 // 2
    return Block(signal.location - fh, data=data)


@deprecated("Not used in interlaced_symm_v2")
def convolve_spec(signal: Block, response_spec: Tensor, taxis: int = -1) -> Block:
    '''
    As convolve() but provide response in padded, Fourier representation.
    '''
    iscomplex = False
    if signal.data.dtype == torch.complex64 or signal.data.dtype == torch.complex128:
        iscomplex = True
    signal = signal_pad(signal, response_spec.shape, taxis)

    # exclude first batched dimension
    dims = to_tuple(torch.arange(signal.vdim) + 1)

    spec = torch.fft.fftn(signal.data, dim=dims)
    signal.data = None # manual release
    measure = spec * response_spec
    measure = torch.fft.ifftn(measure, dim=dims)
    if not iscomplex:
        measure = measure.real
    return Block(location = signal.location, data = measure) # fixme: normalization


@deprecated("Not used in interlaced_symm_v2")
def convolve(signal: Block, response: Tensor, taxis: int = -1) -> Block:
    '''
    Return a tred simple convolution of signal and response.

    Both are interval space representations, not padded nor interlaced.

    This is NOT a general-purpose convolution.

    - signal :: a data block providing signal (charge) on some rectangular span
      of grid points.

    - response :: an N-dimension tensor providing a response.  This response
      must be "spatially centered" and "temporarily causal" (see docs).

    - taxis :: the dimension of response that is interpreted as being along
      drift time.

    The non-batch signal dimensions must be coincident with the response
    dimensions.

    The convolved block is returned.  Its location reflects the size increase
    induced by the convolution.  Along the spatial dimensions, the location is
    reduced by "half" the size of the response dimensions.  Along the drift
    dimension, the location is unchanged.

    See tred/docs/response.org and tred/docs/convo.org for descriptions of
    "spatially centered" and "temporarily causal" requirements and other
    details.
    '''
    c_shape = dft_shape(signal.shape, response.shape)
    response = response_pad(response, c_shape)
    dims = to_tuple(torch.arange(len(c_shape)))
    response_spec = torch.fft.fftn(response, dim=dims)
    return convolve_spec(signal, response_spec, taxis)

@deprecated("Use interlaced_symm_v2 instead")
def interlaced(signal: Block, response: Tensor, steps: IntTensor, taxis: int = -1) -> Block:
    '''
    Return a tred interlaced convolution of signal and response.

    - signal :: an N-D block (holding batched 1+N-D tensor) holding signal
    - response :: an N-D tensor holding response
    - steps :: an integer N-tensor giving the number of steps performed by the interlacing.
    - taxis :: the dimension of N that is considered the time/drift axis.

    FIXME: taxis is not actually used nor tested.
    WARNING: steps[taxis] should always be 1.

    The signal block must be aligned to the lower-left corner of its pixel grid.
    The response tensor must be aligned to the lower-left corner of the collection pixel
    where the single ionizing electron is collected.

    Both signal and response tensors represent interval-space samples.  They are
    not padded but are interlaced.  The interlace spacing is given by steps.

    This is similar to depth-wise convolution in neural network terminology.
    Each channel—representing an impact position—is convolved with its own kernel
    (the field response), where each response element has a fixed offset relative
    to the pixel corner. The resulting output channels (impact positions) are then
    summed to compute the current at a pixel.

    Alternatively, the process can be viewed as performing a convolution with
    a stride equal to the number of impact positions along each spatial dimension.

    The response tensor is expected to exhibit mirror symmetry with respect to the
    collection wire and pixel positions.
    This symmetry informs the determination of the resulting block’s location.

    The convolution() function is applied to each matching interlaced tensor in
    signal and response and the sum over "laces" is returned as a Block.  The
    Block.location of the returned Block positions the result in the
    "super-grid" and is signal.location/steps.
    '''

    super_location = signal.location / steps

    batched_steps = torch.cat([torch.tensor([1], device=steps.device), steps])
    sig_laces = deinterlace(signal.data, batched_steps)
    res_laces = deinterlace(response, steps)

    meas = None
    for sig_lace, res_lace in zip(sig_laces, res_laces):
        sig_lace_block = Block(super_location, data=sig_lace)
        meas_lace_block = convolve(sig_lace_block, res_lace)
        if meas is None:
            meas = meas_lace_block
            continue
        meas.data += meas_lace_block.data
    return meas

@deprecated("Use interlaced_symm_v2 instead")
def interlaced_symm(signal: Block, response: Tensor, steps: IntTensor, taxis: int = -1, symm_axis: int = 0) -> Block:
    '''
    Return a tred interlaced convolution of signal and response, similar to `interlaced`,
    with the additional use of reflection symmetry in the response tensor.

    symm_axis :: The axis (excluding batch) along which mirror symmetry is present in `response`.
    '''

    symm_axis = symm_axis if symm_axis >=0 else steps.shape[0] + symm_axis

    super_location = signal.location / steps

    batched_steps = torch.cat([torch.tensor([1], device=steps.device), steps])
    sig_laces = deinterlace_pairs(signal.data, batched_steps, 1+symm_axis) # one extra dim for batch dim
    res_laces = deinterlace_pairs(response, steps, symm_axis)

    flipdims = (1+symm_axis,) # batch dim == 0

    meas = None
    for sig_lace, res_lace in zip(sig_laces, res_laces):
        sig_lace_block = Block(super_location, data=torch.complex(sig_lace[0], sig_lace[1].flip(dims=flipdims)))
        # res_lace[1] is not used as it is a flipped copy, given the reflection symmetry
        meas_lace_block = convolve(sig_lace_block, res_lace[0])
        meas_lace_block = Block(location = meas_lace_block.location,
                                data = meas_lace_block.data.real + torch.flip(meas_lace_block.data.imag, dims=flipdims))
        if meas is None:
            meas = meas_lace_block
            continue
        meas.data += meas_lace_block.data
    return meas


def interlaced_symm_v2(signal: Block, response: Tensor, steps: IntTensor, taxis: int = -1, symm_axis: int = 0, o_shape=None) -> Block:
    '''
    Return a tred interlaced convolution of signal and response, similar to `interlaced`,
    with the additional use of reflection symmetry in the response tensor.

    symm_axis :: The axis (excluding batch) along which mirror symmetry is present in `response`.
    '''
    # DFT shape
    c_shape = dft_shape(torch.tensor(signal.data.shape[1:]).to(steps.device)//steps, torch.tensor(response.shape).to(steps.device)//steps)
    if o_shape is None:
        o_shape = c_shape
    elif any(tuple(s1 >= s2 for s1,s2 in zip(c_shape, o_shape))):
        raise ValueError(f"Invalid output shape after padding: expected at least Ns + Nr - 1, {c_shape}, but got {o_shape}")

    # output locations
    nrm1 = to_tensor(response.shape, device='cpu') // steps.cpu() - 1
    nrm1[taxis] = 0
    if torch.any(nrm1 % 2):
        raise ValueError(f"Length of response tensor must always be odd. {nrm1} + 1 is given.")
    nrm1 = to_tensor(nrm1, device=signal.data.device)
    fh = nrm1 // 2
    super_location = signal.location // steps - fh

    # deinterlace
    symm_axis = symm_axis if symm_axis >=0 else steps.shape[0] + symm_axis
    flipdims = (1+symm_axis,) # batch dim == 0, dims for flipping

    batched_steps = torch.cat([torch.tensor([1], device=steps.device), steps])
    sig_laces = deinterlace_pairs(signal.data, batched_steps, 1+symm_axis) # one extra dim for batch dim
    res_laces = deinterlace_pairs(response, steps, symm_axis)

    # FFT preparation
    dims = tuple(i+1 for i in range(len(c_shape))) # dims for FFT

    meas = None

    for sig_lace, res_lace in zip(sig_laces, res_laces):
        sig_lace_complex = torch.complex(sig_lace[0], sig_lace[1].flip(dims=flipdims))
        # res_lace[1] is not used as it is a flipped copy, given the reflection symmetry
        res_  = zero_pad(res_lace[0], o_shape)
        meas_ = zero_pad(sig_lace_complex, o_shape)
        meas_ = torch.fft.fftn(res_[None,...], dim=dims) * torch.fft.fftn(meas_, dim=dims)
        if meas is None:
            meas = meas_
        else:
            meas += meas_
    meas = torch.fft.ifftn(meas, dim=dims)
    meas.real = meas.real + torch.flip(meas.imag, dims=flipdims)
    return Block(data=meas.real, location=super_location)
