import os
from setuptools import setup, find_packages


req = ['mysql-connector-python-rf',
       'matplotlib',
       'pyDOE',
       'scipy',
]

setup(name="autotune",
      version="0.0.1",
      description="Database configuration tuning EA",
      author="autotune",
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Software Development',
      ],
      packages=find_packages(),
      include_package_data=True,
      install_requires=req,
)
