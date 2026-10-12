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

"""Tests for the leaf naming helpers in tree_paths.py."""
from typing import NamedTuple

from absl.testing import absltest
import equinox as eqx
import flax
import jax
import jax.numpy as jnp
from jaxpruner import tree_paths


class _Pair(NamedTuple):
  first: jax.Array
  second: jax.Array


class TreePathsTest(absltest.TestCase):

  def testNestedDictNamesMatchFlaxFlattenDict(self):
    tree = {
        'b': {'kernel': jnp.ones((2, 3)), 'bias': jnp.ones(3)},
        'a': jnp.ones(1),
    }
    named_leaves, _ = tree_paths.flatten_with_names(tree)
    self.assertEqual(
        sorted(name for name, _ in named_leaves),
        sorted(flax.traverse_util.flatten_dict(tree).keys()),
    )

  def testFrozenDictNamesMatchPlainDict(self):
    tree = flax.core.freeze({'b': {'kernel': jnp.ones((2, 3))}, 'a': jnp.ones(1)})
    names = [name for name, _ in tree_paths.flatten_with_names(tree)[0]]
    self.assertEqual(names, [('a',), ('b', 'kernel')])

  def testSequenceAndNamedTupleNames(self):
    tree = [_Pair(first=jnp.ones(2), second=jnp.ones(3)), (jnp.ones(1),)]
    names = [name for name, _ in tree_paths.flatten_with_names(tree)[0]]
    self.assertEqual(names, [(0, 'first'), (0, 'second'), (1, 0)])

  def testEquinoxModuleNames(self):
    mlp = eqx.nn.MLP(4, 2, 3, 1, key=jax.random.PRNGKey(0))
    params = eqx.filter(mlp, eqx.is_inexact_array)
    names = [name for name, _ in tree_paths.flatten_with_names(params)[0]]
    self.assertEqual(
        names,
        [
            ('layers', 0, 'weight'),
            ('layers', 0, 'bias'),
            ('layers', 1, 'weight'),
            ('layers', 1, 'bias'),
        ],
    )

  def testSingleArrayHasEmptyName(self):
    named_leaves, _ = tree_paths.flatten_with_names(jnp.ones(3))
    self.assertEqual([name for name, _ in named_leaves], [()])

  def testNoneIsSkippedUnlessMarkedAsLeaf(self):
    tree = {'a': jnp.ones(2), 'b': None}
    names = [name for name, _ in tree_paths.flatten_with_names(tree)[0]]
    self.assertEqual(names, [('a',)])
    named_leaves, _ = tree_paths.flatten_with_names(
        tree, is_leaf=lambda x: x is None
    )
    self.assertEqual([name for name, _ in named_leaves], [('a',), ('b',)])

  def testMapWithNamesKeepsStructure(self):
    tree = {'a': jnp.ones(2), 'b': {'c': jnp.ones(3)}}
    result = tree_paths.map_with_names(lambda name, leaf: '/'.join(name), tree)
    self.assertEqual(result, {'a': 'a', 'b': {'c': 'b/c'}})

  def testMapWithNamesMayReturnNone(self):
    tree = flax.core.freeze({'a': jnp.ones(2), 'b': jnp.ones((2, 2))})
    result = tree_paths.map_with_names(
        lambda name, leaf: 0.5 if leaf.ndim > 1 else None, tree
    )
    self.assertIsInstance(result, flax.core.FrozenDict)
    self.assertEqual(dict(result), {'a': None, 'b': 0.5})


if __name__ == '__main__':
  absltest.main()
