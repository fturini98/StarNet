from setuptools import setup, find_packages

setup(
    name='StarNet',
    readme = "README.md",
    version = '0.dev0',
    author='Francesco Turini',
    author_email='fturini.turini7@gmail.com',
    description='Library for identifying the properties of a star\'s simple population usning a convolutional Neural Network',
    url='https://github.com/fturini98/StarNet',
    license='GNU General Public License (GPL)',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'ezpadova @ git+https://github.com/mfouesneau/ezpadova.git@v2.0#egg=ezpadova',
        'tensorflow',
        'scikit-learn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Operating System :: Microsoft :: Windows',
    ],
    long_description=open('README.md').read(),  # Aggiunto per una descrizione lunga
    long_description_content_type='text/markdown',  # Specifica il formato
)