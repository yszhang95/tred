#!/usr/bin/env python
'''
Some basic utility functions
'''
import subprocess
from pathlib import Path

try:
    import magic
except ImportError:
    magic = None

def mime_type(path):
    path = Path(path)
    if not path.exists():
        return None

    if magic:
        return magic.from_file(path, mime=True)
    return subprocess.check_output(f"file --brief --mime-type {file}", shell=True).decode().strip()
