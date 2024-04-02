#!/bin/bash

if [ ! -d "./venv" ]; then
	./setup_linux.sh
else
	source ./venv/bin/activate
fi

./venv/bin/python3 -m app.main $@
