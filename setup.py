from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    url="https://github.com/helenala/qi-relaxometry",
    author="Helena La",
    author_email="heelenala@gmail.com",
    name='qir',
    version='0.1.1',
    description='Quantum Impurity Relaxometry (QIR) is a package for QI relaxation rates calculations.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["qir"],
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "matplotlib ~= 3.5.2",
        "numpy ~= 1.22.3",
    ],
)
