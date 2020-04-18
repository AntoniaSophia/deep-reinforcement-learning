import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 5):
    sys.exit('Sorry, Python < 3.5 is not supported!')

setup(name='gym_wheel',
      version='1.0.dev0',
      install_requires=['gym', 'numpy'],
      packages=find_packages()
      )
