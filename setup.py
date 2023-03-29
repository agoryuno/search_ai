from setuptools import setup, find_packages

PACKAGE_NAME = 'search_ai'
DESCRIPTION = 'A test of a simple instructor AI'
VERSION = '0.1.0'
AUTHOR = 'Alex Goryunov'
AUTHOR_EMAIL = 'alex.goryunov@gmail.com'
URL = 'https://github.com/agoryuno/search_ai'
LICENSE = 'MIT'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
]

INSTALL_REQUIRES = [
    'numpy',
    'torch'
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    packages=[package for package in find_packages() if package.startswith("search_ai")],
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.8',
)
