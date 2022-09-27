#!/bin/bash

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`
BASE_PATH=`dirname $SCRIPT_PATH`
version="0.0.1"

case "$1" in
	
	build)
		python3.8 setup.py sdist bdist_wheel
	;;
	
	install)
		python3.8 setup.py sdist bdist_wheel
		echo "Install. Need root password"
		sudo pip3.8 install dist/ai_helper-$version.tar.gz
	;;
	
	uninstall)
		echo "Uninstall. Need root password"
		sudo pip3.8 uninstall ai_helper
	;;
	
	install-dev)
		python3.8 setup.py develop
	;;
	
	uninstall-dev)
		python3.8 setup.py develop -u
	;;
	
	upload)
		twine check dist/ai_helper-$version.tar.gz
		twine check dist/ai_helper-$version-py3-none-any.whl
		twine upload -r pypi dist/ai_helper-$version.tar.gz
		twine upload -r pypi dist/ai_helper-$version-py3-none-any.whl
	;;
	
	clean)
		rm -rf dist/*
	;;
	
	*)
		echo "Usage: $0 {build|clean|install|install-dev|uninstall|uninstall-dev|upload}"
		RETVAL=1

esac

exit $RETVAL