from torch import Tensor, IntTensor, Size, int32, iinfo

# The torch dtype to use for representing indices.
index_dtype = int32

MAX_INDEX = iinfo(index_dtype).max
MIN_INDEX = iinfo(index_dtype).min

# An allowed type for a tensor shape.  Typically this is converted to an IntTensor.
Shape = IntTensor|tuple|Size


