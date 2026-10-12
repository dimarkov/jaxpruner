"""Imports every module below the given directories, each in its own process.

Usage: python tools/import_check.py DIRECTORY [DIRECTORY ...]

Run from the repository root with the interpreter of the environment to check.
Separate processes keep one module's absl flags from clashing with another's.
"""
import pathlib
import subprocess
import sys


def main(directories):
  failures = 0
  for directory in directories:
    for path in sorted(pathlib.Path(directory).rglob('*.py')):
      if path.name == '__init__.py':
        continue
      module = '.'.join(path.with_suffix('').parts)
      result = subprocess.run(
          [sys.executable, '-c', f'import {module}'],
          capture_output=True,
          text=True,
          check=False,
      )
      if result.returncode == 0:
        print(f'OK    {module}')
      else:
        failures += 1
        last_line = (result.stderr.strip().splitlines() or ['no output'])[-1]
        print(f'FAIL  {module}\n      {last_line}')
  return 1 if failures else 0


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
