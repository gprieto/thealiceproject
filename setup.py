'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='the_alice_project',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='Alice text generator keras model on Cloud ML Engine',
      author='Guillermo Prieto',
      author_email='guillermo.prieto.santero@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'numpy',
          'h5py'],
      zip_safe=False)
