# !/bin/bash

ERROR="
[Expected Parameter]
Please run this shellscript with the path to the python file you want to run as parameter.
After that the execution will ask for CLI parameters, if needed.
Ex: $ ./docker_run.sh PATH/TO/PYTHON/FILE.py
"

if [[ $1 == "" ]]; then
    echo "$ERROR"
else

    echo "[INSTRUCTION] Enter parameters CLI parameters, else press enter!"
    echo "[TIP] If you don't know about CLI parameters type -h or --help"
    read PARAMETERS

    docker run --rm -it \
    -v $(pwd):/workdir/ \
    diogo364/dl4cv:latest \
    python3 "$1" $PARAMETERS
fi