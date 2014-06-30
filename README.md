Python Toolbox for Nonnegative Matrix Factorization
====================

This package includes Python implementations (with Numpy and Scipy) of
numerical algorithms for computing nonnegative matrix factorization.

Requirements
------------

Numpy (http://www.numpy.org) and Scipy (http://www.scipy.org) need to be
installed.  Versions of Numpy and Scipy tested with this code were 1.6.1 and
0.9.0, respectively.

Installation
------------

Use **setup.py** to install this package:

    sudo python setup.py install

Usage Instructions
------------

When A is a dense (numpy.array) or a sparse (scipy.sparse) matrix, the
following code returns W and H as factor matrices of A with 10 as the lower
rank.

    from nonnegfac.nmf import NMF
    W, H, info = NMF().run(A, 10)

Try to execute **example.py** to see simple usage.  Function `run()` executes
an NMF algorithm once, and Function `run_repeat()` executes an NMF algorithm
for the specified number of times and returns the best result based on the
norm of the error matrix. See **nmf.py** for the optional arguments and the
return information of `run()` and `run_repeat()`.

There are several algorithms implemented and included as separate classes. A
specific algorithm can be used by creating an instance of one of the following
classes. By default, `NMF()` creates an instance of `NMF_ANLS_BLOCKPIVOT`;
another fast algorithm is `NMF_HALS`.  Examples of using each of these
algorithms are also included in **example.py**.  See **nmf.py** and the
following references for more information of algorithms.

* `NMF_ANLS_BLOCKPIVOT` - ANLS with block principal pivoting
* `NMF_ANLS_AS_NUMPY`   - ANLS with scipy.optimize.nnls solver
* `NMF_ANLS_AS_GROUP`   - ANLS with active-set method and column grouping
* `NMF_HALS`            - Hierarchical alternating least squares
* `NMF_MU`              - Multiplicative updating

References
----------
1. Jingu Kim, Yunlong He, and Haesun Park.
   Algorithms for Nonnegative Matrix and Tensor Factorizations:
   A Unified View Based on Block Coordinate Descent Framework.
   Journal of Global Optimization, 58(2), pp. 285-319, 2014.
   http://link.springer.com/content/pdf/10.1007%2Fs10898-013-0035-4.pdf

2. Jingu Kim and Haesun Park.
   Fast Nonnegative Matrix Factorization: An Active-set-like Method
   And Comparisons.
   SIAM Journal on Scientific Computing (SISC), 33(6), pp. 3261-3281, 2011.
   https://sites.google.com/site/jingukim/2011_paper_sisc_nmf.pdf

Feedback
--------
Please send bug reports, comments, or questions to [Jingu Kim](mailto:jingu.kim@nokia.com).
Contributions and extensions with new algorithms are welcome.
