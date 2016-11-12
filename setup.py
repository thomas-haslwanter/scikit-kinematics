from setuptools import setup, find_packages

setup(
    name='scikit-kinematics',
    version="0.3.3",
    packages=find_packages(),

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['docutils>=0.3', 'easygui'],

    # metadata for upload to PyPI
    author       = "Thomas Haslwanter",
    author_email = "thomas.haslwanter@fh-linz.at",
    description  = 'Python utilites for movements in 3d space',
    long_description=open('README.rst').read(),
    license      = 'http://opensource.org/licenses/BSD-2-Clause',
    download_url = 'https://github.com/thomas-haslwanter/scikit-kinematics',
    keywords     = 'quaterions rotations',
    url          = 'http://work.thaslwanter.at/skinematics/html',
    classifiers  = ['Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 3',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Topic :: Scientific/Engineering'],
    test_suite   = 'nose.collector',
    tests_require=['nose'],
)
