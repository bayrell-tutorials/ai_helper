#!/bin/bash

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`
BASE_PATH=`dirname $SCRIPT_PATH`
version="0.1.6"

case "$1" in
	
	build)
		python3 setup.py sdist bdist_wheel
	;;
	
	install)
		python3 setup.py sdist bdist_wheel
		pip3 install dist/tiny_ai_helper-$version.tar.gz
	;;
	
	uninstall)
		pip3 uninstall tiny_ai_helper
	;;
	
	install-dev)
		python3 setup.py develop --prefix=~/.local
	;;
	
	uninstall-dev)
		python3 setup.py develop -u --prefix=~/.local
	;;
	
	upload)
		twine check dist/tiny_ai_helper-$version.tar.gz
		twine check dist/tiny_ai_helper-$version-py3-none-any.whl
		twine upload -r pypi dist/tiny_ai_helper-$version.tar.gz
		twine upload -r pypi dist/tiny_ai_helper-$version-py3-none-any.whl
	;;
	
	clean)
		rm -rf dist/*
	;;
	
	*)
		echo "Usage: $0 {build|clean|install|install-dev|uninstall|uninstall-dev|upload}"
		RETVAL=1

esac

exit $RETVAL
