#!/usr/bin/env python
descr = """\
Example package.

This is a do nothing package, to show how to organize a scikit.
"""

DISTNAME            = 'scikit-kinematics'
DESCRIPTION         = 'Python utilites for movements in 3d space'
LONG_DESCRIPTION    = open('README.rst').read()
MAINTAINER          = 'Thomas Haslwanter'
MAINTAINER_EMAIL    = 'thomas.haslwanter@fh-linz.at'
URL                 = 'http://work.thaslwanter.at/sklearn/html'
LICENSE             = 'http://opensource.org/licenses/BSD-2-Clause'
DOWNLOAD_URL        = 'https://github.com/thomas-haslwanter/scikit-kinematics'
PACKAGE_NAME        = 'skinematics'


import os
import sys
import subprocess

import setuptools
from numpy.distutils.core import setup

def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg: "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage(PACKAGE_NAME)
    return config

def get_version():
    """Obtain the version number"""
    import imp
    mod = imp.load_source('version', os.path.join(PACKAGE_NAME, 'version.py'))
    return mod.__version__

# Documentation building command
try:
    from sphinx.setup_command import BuildDoc as SphinxBuildDoc
    class BuildDoc(SphinxBuildDoc):
        """Run in-place build before Sphinx doc build"""
        def run(self):
            ret = subprocess.call([sys.executable, sys.argv[0], 'build_ext', '-i'])
            if ret != 0:
                raise RuntimeError("Building Scipy failed!")
            SphinxBuildDoc.run(self)
    cmdclass = {'build_sphinx': BuildDoc}
except ImportError:
    cmdclass = {}

# Call the setup function
if __name__ == "__main__":

    metadata= dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        long_description=LONG_DESCRIPTION,
        install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'easygui'],
        classifiers=['Development Status :: 4 - Beta',
                     'Programming Language :: Python :: 2',
                     'Programming Language :: Python :: 3',
                     'Intended Audience :: Developers',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved :: BSD License',
                     'Topic :: Scientific/Engineering'],
        test_suite='nose.collector',
        include_package_data=True,
        configuration=configuration,
        cmdclass=cmdclass,
        version=get_version()
    )
    setup(**metadata)
