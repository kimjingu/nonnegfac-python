info = {'name':'nonnegfac',
        'description':'Python Implementations of Nonnegative Matrix Factorization Algorithms',
        'version':'0.1',
        'author':'Jingu Kim',
        'author_email':'jingu.kim@nokia.com',
        'license':'new BSD',
        'packages':['nonnegfac']
        }

if __name__ == '__main__':
    try:
        import numpy
        import scipy
    except ImportError:
        print 'This package requires Numpy (http://www.numpy.org) and Scipy (http://www.scipy.org) installed.'
        print 'Exiting without installation.'
        import sys
        sys.exit(-1)

    from distutils.core import setup
    setup(**info)
