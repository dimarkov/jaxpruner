# coding=utf-8
# Copyright 2024 Jaxpruner Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Names for the leaves of arbitrary parameter PyTrees.

JAX describes the position of a leaf with a key path such as
`(DictKey('dense'), GetAttrKey('weight'), SequenceKey(0))`. The functions here
turn such a path into a tuple of plain dict keys, attribute names and indices,
e.g. `('dense', 'weight', 0)`. For nested dicts this is the same tuple that
`flax.traverse_util.flatten_dict` produces.
"""
import jax


def _entry_name(entry):
  """Returns the plain key, attribute name or index held by a key-path entry."""
  for attribute in ('key', 'name', 'idx'):
    if hasattr(entry, attribute):
      return getattr(entry, attribute)
  return str(entry)


def path_name(key_path):
  """Converts a JAX key path into a tuple of plain keys, names and indices."""
  return tuple(_entry_name(entry) for entry in key_path)


def flatten_with_names(tree, is_leaf=None):
  """Flattens a tree into a list of `(name, leaf)` pairs and its treedef."""
  paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(
      tree, is_leaf=is_leaf
  )
  named_leaves = [(path_name(path), leaf) for path, leaf in paths_and_leaves]
  return named_leaves, treedef


def map_with_names(fn, tree, is_leaf=None):
  """Maps `fn(name, leaf)` over the leaves of a tree, keeping its structure."""
  return jax.tree_util.tree_map_with_path(
      lambda path, leaf: fn(path_name(path), leaf), tree, is_leaf=is_leaf
  )
