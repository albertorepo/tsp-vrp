#!/usr/bin/env python
# coding=utf-8
import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='route_metaheuristic',
      version="0.0.1",
      author='Alberto Castaño',
      author_email="bertocast@gmail.com",
      description='Metaheuristic algorithms for the resolution of route problems',
      packages=find_packages(exclude='tests'),
      long_description=read('README.rst'),
      keywords='tsp, cvrp, metaheuristic, local_search, simulated_annealing, tabu_search',
      classifiers=[
          "Development Status :: 3 - Alpha",
          'Intended Audience :: Developers',
          'Natural Language :: English',
          'Operating System :: Unix',
          'Programming Language :: Python :: 2.7'
      ],
      include_package_data=True,
      install_requires=[
          'numpy',
      ],
      tests_require=[
          'nose',
      ],
      entry_points={
          'console_scripts': [
              'solve_tsp = route_metaheuristic.tsp.__main__:main',
              'solve_cvrp = route_metaheuristic.cvrp.__main__:main']
      }
      )
