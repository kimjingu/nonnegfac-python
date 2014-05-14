import numpy as np
import scipy.sparse as sps
import scipy.optimize as opt
import numpy.linalg as nla
import matrix_utils as mu
import time
import json
from numpy import random
from nnls import nnlsm_activeset
from nnls import nnlsm_blockpivot


class NMF_Base(object):

    """ Base class for NMF algorithms

    Specific algorithms need to be implemented by deriving from this class.
    """
    default_max_iter = 100
    default_max_time = np.inf

    def __init__(self):
        raise NotImplementedError(
            'NMF_Base is a base class that cannot be instantiated')

    def set_default(self, default_max_iter, default_max_time):
        self.default_max_iter = default_max_iter
        self.default_max_time = default_max_time

    def run(self, A, k, init=None, max_iter=None, max_time=None, verbose=0):
        """ Run a NMF algorithm

        Parameters
        ----------
        A : numpy.array or scipy.sparse matrix, shape (m,n)
        k : int - target lower rank

        Optional Parameters
        -------------------
        init : (W_init, H_init) where
                    W_init is numpy.array of shape (m,k) and
                    H_init is numpy.array of shape (n,k).
                    If provided, these values are used as initial values for NMF iterations.
        max_iter : int - maximum number of iterations.
                    If not provided, default maximum for each algorithm is used.
        max_time : int - maximum amount of time in seconds.
                    If not provided, default maximum for each algorithm is used.
        verbose : int - 0 (default) - No debugging information is collected, but
                                    input and output information is printed on screen.
                        -1 - No debugging information is collected, and
                                    nothing is printed on screen.
                        1 (debugging/experimental purpose) - History of computation is
                                        returned. See 'rec' variable.
                        2 (debugging/experimental purpose) - History of computation is
                                        additionally printed on screen.
        Returns
        -------
        (W, H, rec)
        W : Obtained factor matrix, shape (m,k)
        H : Obtained coefficient matrix, shape (n,k)
        rec : dict - (debugging/experimental purpose) Auxiliary information about the execution
        """
        info = {'k': k,
                'alg': str(self.__class__),
                'A_dim_1': A.shape[0],
                'A_dim_2': A.shape[1],
                'A_type': str(A.__class__),
                'max_iter': max_iter if max_iter is not None else self.default_max_iter,
                'verbose': verbose,
                'max_time': max_time if max_time is not None else self.default_max_time}
        if init != None:
            W = init[0].copy()
            H = init[1].copy()
            info['init'] = 'user_provided'
        else:
            W = random.rand(A.shape[0], k)
            H = random.rand(A.shape[1], k)
            info['init'] = 'uniform_random'

        if verbose >= 0:
            print '[NMF] Running: '
            print json.dumps(info, indent=4, sort_keys=True)

        norm_A = mu.norm_fro(A)
        total_time = 0

        if verbose >= 1:
            his = {'iter': [], 'elapsed': [], 'rel_error': []}

        start = time.time()
        # algorithm-specific initilization
        (W, H) = self.initializer(W, H)

        for i in range(1, info['max_iter'] + 1):
            start_iter = time.time()
            # algorithm-specific iteration solver
            (W, H) = self.iter_solver(A, W, H, k, i)
            elapsed = time.time() - start_iter

            if verbose >= 1:
                rel_error = mu.norm_fro_err(A, W, H, norm_A) / norm_A
                his['iter'].append(i)
                his['elapsed'].append(elapsed)
                his['rel_error'].append(rel_error)
                if verbose >= 2:
                    print 'iter:' + str(i) + ', elapsed:' + str(elapsed) + ', rel_error:' + str(rel_error)

            total_time += elapsed
            if total_time > info['max_time']:
                break

        W, H, weights = mu.normalize_column_pair(W, H)

        final = {}
        final['norm_A'] = norm_A
        final['rel_error'] = mu.norm_fro_err(A, W, H, norm_A) / norm_A
        final['iterations'] = i
        final['elapsed'] = time.time() - start

        rec = {'info': info, 'final': final}
        if verbose >= 1:
            rec['his'] = his

        if verbose >= 0:
            print '[NMF] Completed: '
            print json.dumps(final, indent=4, sort_keys=True)
        return (W, H, rec)

    def run_repeat(self, A, k, num_trial, max_iter=None, max_time=None, verbose=0):
        """ Run an NMF algorithm several times with random initial values 
            and return the best result in terms of the Frobenius norm of
            the approximation error matrix

        Parameters
        ----------
        A : numpy.array or scipy.sparse matrix, shape (m,n)
        k : int - target lower rank
        num_trial : int number of trials

        Optional Parameters
        -------------------
        max_iter : int - maximum number of iterations for each trial.
                    If not provided, default maximum for each algorithm is used.
        max_time : int - maximum amount of time in seconds for each trial.
                    If not provided, default maximum for each algorithm is used.
        verbose : int - 0 (default) - No debugging information is collected, but
                                    input and output information is printed on screen.
                        -1 - No debugging information is collected, and
                                    nothing is printed on screen.
                        1 (debugging/experimental purpose) - History of computation is
                                        returned. See 'rec' variable.
                        2 (debugging/experimental purpose) - History of computation is
                                        additionally printed on screen.
        Returns
        -------
        (W, H, rec)
        W : Obtained factor matrix, shape (m,k)
        H : Obtained coefficient matrix, shape (n,k)
        rec : dict - (debugging/experimental purpose) Auxiliary information about the execution
        """
        for t in xrange(num_trial):
            if verbose >= 0:
                print '[NMF] Running the {0}/{1}-th trial ...'.format(t + 1, num_trial)
            this = self.run(A, k, verbose=(-1 if verbose is 0 else verbose))
            if t == 0:
                best = this
            else:
                if this[2]['final']['rel_error'] < best[2]['final']['rel_error']:
                    best = this
        if verbose >= 0:
            print '[NMF] Best result is as follows.'
            print json.dumps(best[2]['final'], indent=4, sort_keys=True)
        return best

    def iter_solver(self, A, W, H, k, it):
        raise NotImplementedError

    def initializer(self, W, H):
        return (W, H)


class NMF_ANLS_BLOCKPIVOT(NMF_Base):

    """ NMF algorithm: ANLS with block principal pivoting

    J. Kim and H. Park, Fast nonnegative matrix factorization: An active-set-like method and comparisons,
    SIAM Journal on Scientific Computing, 
    vol. 33, no. 6, pp. 3261-3281, 2011.
    """

    def __init__(self, default_max_iter=50, default_max_time=np.inf):
        self.set_default(default_max_iter, default_max_time)

    def iter_solver(self, A, W, H, k, it):
        Sol, info = nnlsm_blockpivot(W, A, init=H.T)
        H = Sol.T
        Sol, info = nnlsm_blockpivot(H, A.T, init=W.T)
        W = Sol.T
        return (W, H)


class NMF_ANLS_AS_NUMPY(NMF_Base):

    """ NMF algorithm: ANLS with scipy.optimize.nnls solver
    """

    def __init__(self, default_max_iter=50, default_max_time=np.inf):
        self.set_default(default_max_iter, default_max_time)

    def iter_solver(self, A, W, H, k, it):
        if not sps.issparse(A):
            for j in xrange(0, H.shape[0]):
                res = opt.nnls(W, A[:, j])
                H[j, :] = res[0]
        else:
            for j in xrange(0, H.shape[0]):
                res = opt.nnls(W, A[:, j].toarray()[:, 0])
                H[j, :] = res[0]

        if not sps.issparse(A):
            for j in xrange(0, W.shape[0]):
                res = opt.nnls(H, A[j, :])
                W[j, :] = res[0]
        else:
            for j in xrange(0, W.shape[0]):
                res = opt.nnls(H, A[j, :].toarray()[0,:])
                W[j, :] = res[0]
        return (W, H)


class NMF_ANLS_AS_GROUP(NMF_Base):

    """ NMF algorithm: ANLS with active-set method and column grouping

    H. Kim and H. Park, Nonnegative matrix factorization based on alternating nonnegativity 
    constrained least squares and active set method, SIAM Journal on Matrix Analysis and Applications, 
    vol. 30, no. 2, pp. 713-730, 2008.
    """

    def __init__(self, default_max_iter=50, default_max_time=np.inf):
        self.set_default(default_max_iter, default_max_time)

    def iter_solver(self, A, W, H, k, it):
        if it == 1:
            Sol, info = nnlsm_activeset(W, A)
            H = Sol.T
            Sol, info = nnlsm_activeset(H, A.T)
            W = Sol.T
        else:
            Sol, info = nnlsm_activeset(W, A, init=H.T)
            H = Sol.T
            Sol, info = nnlsm_activeset(H, A.T, init=W.T)
            W = Sol.T
        return (W, H)


class NMF_HALS(NMF_Base):

    """ NMF algorithm: Hierarchical alternating least squares

    A. Cichocki and A.-H. Phan, Fast local algorithms for large scale nonnegative matrix and tensor factorizations,
    IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences,
    vol. E92-A, no. 3, pp. 708-721, 2009.
    """

    def __init__(self, default_max_iter=100, default_max_time=np.inf):
        self.eps = 1e-16
        self.set_default(default_max_iter, default_max_time)

    def initializer(self, W, H):
        W, H, weights = mu.normalize_column_pair(W, H)
        return W, H

    def iter_solver(self, A, W, H, k, it):
        AtW = A.T.dot(W)
        WtW = W.T.dot(W)
        for kk in xrange(0, k):
            temp_vec = H[:, kk] + AtW[:, kk] - H.dot(WtW[:, kk])
            H[:, kk] = np.maximum(temp_vec, self.eps)

        AH = A.dot(H)
        HtH = H.T.dot(H)
        for kk in xrange(0, k):
            temp_vec = W[:, kk] * HtH[kk, kk] + AH[:, kk] - W.dot(HtH[:, kk])
            W[:, kk] = np.maximum(temp_vec, self.eps)
            ss = nla.norm(W[:, kk])
            if ss > 0:
                W[:, kk] = W[:, kk] / ss

        return (W, H)


class NMF_MU(NMF_Base):

    """ NMF algorithm: Multiplicative updating 

    Lee and Seung, Algorithms for non-negative matrix factorization, 
    Advances in Neural Information Processing Systems, 2001, pp. 556-562.
    """

    def __init__(self, default_max_iter=500, default_max_time=np.inf):
        self.eps = 1e-16
        self.set_default(default_max_iter, default_max_time)

    def iter_solver(self, A, W, H, k, it):
        AtW = A.T.dot(W)
        HWtW = H.dot(W.T.dot(W)) + self.eps
        H = H * AtW
        H = H / HWtW

        AH = A.dot(H)
        WHtH = W.dot(H.T.dot(H)) + self.eps
        W = W * AH
        W = W / WHtH

        return (W, H)


class NMF(NMF_ANLS_BLOCKPIVOT):

    """ Default NMF algorithm: NMF_ANLS_BLOCKPIVOT
    """

    def __init__(self, default_max_iter=50, default_max_time=np.inf):
        self.set_default(default_max_iter, default_max_time)


def _mmio_example(m=100, n=100, k=10):
    print '\nTesting mmio read and write ...\n'
    import scipy.io.mmio as mmio

    W_org = random.rand(m, k)
    H_org = random.rand(n, k)
    X = W_org.dot(H_org.T)
    X[random.rand(n, k) < 0.5] = 0
    X_sparse = sps.csr_matrix(X)

    filename = '_temp_mmio.mtx'
    mmio.mmwrite(filename, X_sparse)
    A = mmio.mmread(filename)

    alg = NMF_ANLS_BLOCKPIVOT()
    rslt = alg.run(X_sparse, k, max_iter=50)


def _compare_nmf(m=300, n=300, k=10):
    from pylab import plot, show, legend, xlabel, ylabel

    W_org = random.rand(m, k)
    H_org = random.rand(n, k)
    A = W_org.dot(H_org.T)

    print '\nComparing NMF algorithms ...\n'

    names = [NMF_MU, NMF_HALS, NMF_ANLS_BLOCKPIVOT,
             NMF_ANLS_AS_NUMPY, NMF_ANLS_AS_GROUP]
    iters = [2000, 1000, 100, 100, 100]
    labels = ['mu', 'hals', 'anls_bp', 'anls_as_numpy', 'anls_as_group']
    styles = ['-x', '-o', '-+', '-s', '-D']

    results = []
    init_val = (random.rand(m, k), random.rand(n, k))

    for i in range(len(names)):
        alg = names[i]()
        results.append(
            alg.run(A, k, init=init_val, max_iter=iters[i], verbose=1))

    for i in range(len(names)):
        his = results[i][2]['his']
        plot(np.cumsum(his['elapsed']), his['rel_error'],
             styles[i], label=labels[i])

    xlabel('time (sec)')
    ylabel('relative error')
    legend()
    show()


def _test_nmf(m=300, n=300, k=10):
    W_org = random.rand(m, k)
    H_org = random.rand(n, k)
    A = W_org.dot(H_org.T)

    alg_names = [NMF_ANLS_BLOCKPIVOT, NMF_ANLS_AS_GROUP,
                 NMF_ANLS_AS_NUMPY, NMF_HALS, NMF_MU]
    iters = [50, 50, 50, 500, 1000]

    print '\nTesting with a dense matrix...\n'
    for alg_name, i in zip(alg_names, iters):
        alg = alg_name()
        rslt = alg.run(A, k, max_iter=i)

    print '\nTesting with a sparse matrix...\n'
    A_sparse = sps.csr_matrix(A)
    for alg_name, i in zip(alg_names, iters):
        alg = alg_name()
        rslt = alg.run(A_sparse, k, max_iter=i)


if __name__ == '__main__':
    _test_nmf()
    _mmio_example()

    # To see an example of comparisons of NMF algorithms, execute
    # _compare_nmf() with X-window enabled.
    # _compare_nmf()
