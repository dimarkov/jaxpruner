"""Checks that notebooks are valid, fully executed and free of error outputs.

Usage: python tools/check_notebooks.py NOTEBOOK [NOTEBOOK ...]
"""
import json
import sys


def check(path):
  """Returns 'ok' or a short description of what is wrong with a notebook."""
  try:
    with open(path, encoding='utf-8') as f:
      notebook = json.load(f)
  except json.JSONDecodeError as e:
    return f'invalid JSON: {e}'
  code_cells = [
      cell
      for cell in notebook['cells']
      if cell['cell_type'] == 'code' and ''.join(cell['source']).strip()
  ]
  errors = [
      output
      for cell in code_cells
      for output in cell.get('outputs', [])
      if output.get('output_type') == 'error'
  ]
  if errors:
    return f'{len(errors)} error output(s), first: {errors[0].get("ename")}'
  unexecuted = [c for c in code_cells if c.get('execution_count') is None]
  if unexecuted:
    return f'{len(unexecuted)} of {len(code_cells)} code cells not executed'
  return 'ok'


def main(paths):
  failed = False
  for path in paths:
    status = check(path)
    print(f'{path}: {status}')
    failed = failed or status != 'ok'
  return 1 if failed else 0


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
