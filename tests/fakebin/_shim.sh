#!/bin/sh
# Every fake tool is a symlink to this shim; fake_tool.py decides what to print by the link's name.
# Uses parameter expansion only: tests run with PATH limited to this directory.
exec "${GFV_FAKE_PYTHON:?set GFV_FAKE_PYTHON}" "${0%/*}/fake_tool.py" "${0##*/}" "$@"
