from setuptools import setup, find_packages

setup(
    name='StarNet',
    readme = "README.md",
    version = '0.dev0',
    author='Francesco Turini',
    author_email='fturini.turini7@gmail.com',
    description='Library for identifing the properties of a star\'s simple population usning a convolutional Neural Network',
    url='https://github.com/fturini98/StraNet',
    license='GNU General Public License (GPL)',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'json',
        'pandas',
        'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Operating System :: Microsoft :: Windows',
    ],
)