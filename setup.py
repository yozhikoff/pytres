import sys

from setuptools import find_packages
from setuptools import setup


def setup_package():
    install_requires = ['numpy>=1.20.1', 'pandas>=1.2.3', 'scipy>=1.6.1', 'tqdm>=4.59.0', 'torch==1.8.1+cpu',
                        'pyro-ppl>=1.5.2', 'matplotlib>=3.3.4']
    metadata = dict(
        name='pytres',
        version='0.01',
        description='Package for TRES data decomposition',
        url='https://github.com/yozhikoff/pytres/',
        author='Artem Shmatko',
        author_email='artem_shmatko@protonmail.com',
        packages=find_packages(),
        install_requires=install_requires,
        dependency_links=['https://download.pytorch.org/whl/torch_stable.html']
    )

    setup(**metadata)


if __name__ == '__main__':
    if sys.version_info < (2, 8):
        sys.exit('Sorry, Python 2 is not supported')

    setup_package()
