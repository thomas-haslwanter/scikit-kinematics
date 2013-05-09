#!/usr/bin/env python
descr = """\
Example package.

This is a do nothing package, to show how to organize a scikit.
"""

DISTNAME            = 'scikits-example'
DESCRIPTION         = 'Scikits example package'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'David Cournapeau',
MAINTAINER_EMAIL    = 'david@ar.media.kyoto-u.ac.jp',
URL                 = 'http://projects.scipy.org/scipy/scikits'
LICENSE             = 'MIT'
DOWNLOAD_URL        = URL

import os
import sys

import setuptools
from numpy.distutils.core import setup

def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(package_name, parent_package, top_path,
                           version=get_version(),
                           maintainer=MAINTAINER,
                           maintainer_email=MAINTAINER_EMAIL,
                           description=DESCRIPTION,
                           license=LICENSE,
                           url=URL,
                           download_url=DOWNLOAD_URL,
                           long_description=LONG_DESCRIPTION)
    return config

def get_version():
    """Obtain the version number"""
    import imp
    mod = imp.load_source('version', 'skexample/version.py')
    return mod.__version__

if __name__ == "__main__":
    setup(configuration = configuration,
        install_requires = 'numpy',
        include_package_data = True,
        #test_suite="tester", # for python setup.py test
        classifiers =
            [ 'Development Status :: 1 - Planning',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering'])
