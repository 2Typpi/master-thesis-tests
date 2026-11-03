#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
CXX="${CXX:-g++}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    printf 'CONDA_PREFIX is not set; activate the environment containing Eigen first.\n' >&2
    exit 1
fi

EIGEN_INCLUDE="${EIGEN_INCLUDE:-$CONDA_PREFIX/include/eigen3}"
if [[ ! -f "$EIGEN_INCLUDE/Eigen/Core" ]]; then
    EIGEN_INCLUDE="$CONDA_PREFIX/include"
fi
if [[ ! -f "$EIGEN_INCLUDE/Eigen/Core" ]]; then
    printf 'Eigen headers not found under %s. Install them with: conda install -c conda-forge eigen\n' "$CONDA_PREFIX" >&2
    exit 1
fi

read -r -a PYTHON_INCLUDES <<< "$("$PYTHON" -c 'import sysconfig; print(" ".join("-I" + sysconfig.get_path(p, scheme="posix_prefix") for p in ("include", "platinclude") if sysconfig.get_path(p, scheme="posix_prefix")))')"
NUMPY_INCLUDE="$("$PYTHON" -c 'import numpy; print(numpy.get_include())')"
EXT_SUFFIX="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')"

cd "$SCRIPT_DIR"
"$CXX" -c Dlo_iso.cpp Dlo_iso_wrap.cpp Dlo_utils.cpp \
    -I"$EIGEN_INCLUDE" -I"$NUMPY_INCLUDE" "${PYTHON_INCLUDES[@]}" \
    -fPIC -std=c++14 -O3 -DNDEBUG
"$CXX" -shared Dlo_iso.o Dlo_iso_wrap.o Dlo_utils.o -o "_Dlo_iso${EXT_SUFFIX}" -fPIC
PYTHONPATH="$(cd "$SCRIPT_DIR/../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -c 'from dlo.cpp import Dlo_iso; print("Built", Dlo_iso.DLO_iso)'
