#!/bin/bash
set -euo pipefail

# Python needs libc10 preloaded on this aarch64 host to reserve static TLS.
# Native compiler children must not inherit it: GNU as otherwise loads libc10
# against the host's older libstdc++ and fails on GLIBCXX_3.4.26/3.4.29.
unset LD_PRELOAD
exec /home/HPCBase/compilers/gcc/11.3.0/bin/gcc "$@"
