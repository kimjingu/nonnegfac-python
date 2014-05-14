from numpy import random
from nonnegfac.nmf import NMF
from nonnegfac.nmf import NMF_ANLS_BLOCKPIVOT
from nonnegfac.nmf import NMF_ANLS_AS_NUMPY
from nonnegfac.nmf import NMF_ANLS_AS_GROUP
from nonnegfac.nmf import NMF_HALS
from nonnegfac.nmf import NMF_MU

if __name__ == '__main__':
    W_org = random.rand(300, 10)
    H_org = random.rand(300, 10)
    A = W_org.dot(H_org.T)
    print '\nTesting NMF().run() ...\n'
    W, H, info = NMF().run(A, 10)
    print '\nTesting NMF().run_repeat() ...\n'
    W, H, info = NMF().run_repeat(A, 10, 10)

    print '\nTesting NMF_ANLS_BLOCKPIVOT ...\n'
    W, H, info = NMF_ANLS_BLOCKPIVOT().run(A, 10, max_iter=50)
    print '\nTesting NMF_ANLS_AS_NUMPY ...\n'
    W, H, info = NMF_ANLS_AS_NUMPY().run(A, 10, max_iter=50)
    print '\nTesting NMF_ANLS_AS_GROUP ...\n'
    W, H, info = NMF_ANLS_AS_GROUP().run(A, 10, max_iter=50)
    print '\nTesting NMF_HALS ...\n'
    W, H, info = NMF_HALS().run(A, 10, max_iter=500)
    print '\nTesting NMF_MU ...\n'
    W, H, info = NMF_MU().run(A, 10, max_iter=500)
