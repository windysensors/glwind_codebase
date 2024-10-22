from setuptools import setup, find_packages

setup(
    name='windprofiles',
    version='0.2.1',
    packages=find_packages(),
    py_modules=[
        'atmo_calc',
        'stat_calc',
        'objects',
        'units',
        'exceptions',
        'roses',
        'temporal'
    ],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'tqdm'
    ],
    author='Elliott Walker',
    author_email='walker.elliott.j@gmail.com',
    description='A package for wind profile calculations and analysis, part of the 2024 GLWind codebase',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Intergalactyc/glwind_codebase',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
