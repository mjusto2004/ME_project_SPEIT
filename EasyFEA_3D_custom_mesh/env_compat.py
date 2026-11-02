"""Compatibility shim for the local Anaconda ``Fea`` environment.

On this Windows machine the MKL 2026.1.0 Intel/TBB threading layer aborts
with a native exception (0xc06d007f) on the first dense matrix product
(``np.matmul`` / BLAS ``dgemm``).  The sequential layer works correctly.

The variable must be set before NumPy/EasyFEA are imported, hence this module
is imported first by every entry point.  An externally supplied value wins.
"""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
