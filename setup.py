from setuptools import setup


setup(
    name='bayesian_free_energies',
    author='Gregory A. Ross',
    author_email='gregory.ross@choderalab.org',
    version=0.0,
    url='https://github.com/choderalab/bayesian-free-energies',
    packages=['bams'],
    license='MIT',
    long_description=open('README.md').read(),
    platforms=[
            'Linux',
            'Mac OS-X',
            'Unix'],
    zip_safe=False,
)