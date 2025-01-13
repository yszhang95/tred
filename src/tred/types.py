from torch import Tensor, IntTensor, Size, int32

# The torch dtype to use for representing indices.
index_dtype = int32


# An allowed type for a tensor shape.  Typically this is converted to an IntTensor.
Shape = IntTensor|tuple|Size


