from setuptools import setup, find_packages

# Used for installing test dependencies directly
tests_require = [
    'flake8',
    'mock',
    'nose',
    'nose-timer',
    'coverage<4.1',
]

setup(
    name='peak-forevaster',
    version='1.0.0',
    description="Building Load Peak Forecaster",
    author="Sam Hollenbach",
    author_email="shollenbach@axiomexergy.com",
    packages=find_packages(exclude=['test', 'test_*', 'fixtures']),
    install_requires=[
            'pandas',
            'numpy',
            'sklearn',
            'seaborn',
            'tensorflow',
            'optimizer-engine',
        ],
    test_suite='nose.collector',
    tests_require=tests_require,
    # For installing test dependencies directly
    extras_require={'test': tests_require},
)
