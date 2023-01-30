"""Setup.py file for tf_grain."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='tf_grain',
    version='0.1.0',
    description='Grain',
    author='Grain team',
    author_email='no-reply@google.com',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.so']},
    python_requires='>=3.8',
    install_requires=[
        'absl-py',
        'array_record',
        'etils',
        'google-re2',
        'jax',
        'numpy',
        'orbax',
        'tensorflow',
        'seqio',
        'typing_extensions',
    ],
    url='https://github.com/google/grain/tree/main/_src/tensorflow',
    license='Apache-2.0',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    zip_safe=False,
)
