#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

"""
List:
https://pypi.python.org/pypi?%3Aaction=list_classifiers
"""

from setuptools import setup, find_packages
from os.path import abspath, dirname, join

setup(
	name="tiny_ai_helper",
	version="0.0.9",
	description="Tiny AI Helper for PyTorch",
	long_description=open(join(abspath(dirname(__file__)), 'README.md'), encoding='utf-8').read(),
	long_description_content_type='text/markdown',
	author="Ildar Bikmamatov",
	author_email="ildar@bayrell.org",
	license="MIT License",
	url = "https://github.com/tiny-php/ai_helper",
	packages=find_packages(),
	include_package_data = True,
	classifiers=[
		'License :: OSI Approved :: MIT License',
		'Operating System :: POSIX :: Linux',
		'Operating System :: Microsoft :: Windows :: Windows 10',
		'Programming Language :: Python',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.8',
		'Topic :: Scientific/Engineering :: Artificial Intelligence',
	],
	keywords = [
		"ai helper", "pytorch"
	],
)