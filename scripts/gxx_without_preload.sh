#!/bin/bash
set -euo pipefail

unset LD_PRELOAD
exec /home/HPCBase/compilers/gcc/11.3.0/bin/g++ "$@"
