from setuptools import setup

requirements = [
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
    'seaborn',
    'scikit-learn',
    'moviepy',
    'xlrd'  # required for pandas to load excel files
]

setup(
    name='Fiber_Photometry_Analysis',
    version='1.0.0',
    install_requires=requirements,
    packages=['fiber_photometry_analysis'],
    python_requires='>=3.1',
    # entry_points={
    #     'console_scripts': [
    #         'photometry_analysis=fiber_photometry_analysis.pipeline:main',
    #     ],
    # },
    url='',
    license='GPLv2',
    author='Tom-top',
    author_email='thomas.topilko@icm-institute.org',
    description='Python toolkit for the analysis of fiber photometry data and related video recordings. '
)
