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

"""Tests for global pruning algorithms."""
import dataclasses
from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized

import chex
import equinox as eqx
import flax
import jax
import jax.numpy as jnp
from jaxpruner.algorithms import global_pruners


class _Pair(NamedTuple):
  first: jax.Array
  second: jax.Array


@dataclasses.dataclass
class _ScaledMagnitude(global_pruners.GlobalMagnitudePruning):
  """Magnitude scores multiplied by a `scale` tree passed as extra argument."""

  def calculate_scores(self, params, sparse_state=None, grads=None, scale=None):
    del sparse_state, grads
    return jax.tree.map(lambda p, s: jnp.abs(p) * s, params, scale)


class GlobalPrunerTest(parameterized.TestCase, absltest.TestCase):

  def testSETDefaultArguments(self):
    pruner = global_pruners.GlobalMagnitudePruning()
    self.assertIsNone(pruner.custom_sparsity_map)
    self.assertTrue(pruner.use_normalization)
    self.assertEqual(pruner.sparsity, 0)

  @parameterized.parameters([
      global_pruners.GlobalMagnitudePruning,
      global_pruners.GlobalSaliencyPruning,
  ])
  def testGlobalPruning(self, updater):
    # Sparsity is passed here for the topk_function to work.
    updater = updater(filter_fn=lambda *_: True, sparsity=0.5)
    score_tree = {
        'w': jnp.array([0.1, 0.9, -0.2, 0.5]),
        'w2': jnp.array([0.7, 0.8]),
    }
    expected_tree = {'w': jnp.array([0, 1, 0, 0]), 'w2': jnp.array([1, 1])}

    chex.assert_trees_all_close(
        updater.create_masks(score_tree, target_sparsity=0.5), expected_tree
    )

  def testInstantSparsifyForwardsExtraArguments(self):
    updater = _ScaledMagnitude(
        filter_fn=lambda *_: True, sparsity=0.5, use_normalization=False
    )
    params = {'w': jnp.array([1.0, 2.0, 3.0, 4.0])}
    scale = {'w': jnp.array([10.0, 10.0, 0.1, 0.1])}
    pruned_params, masks = updater.instant_sparsify(params, scale=scale)
    chex.assert_trees_all_close(masks, {'w': jnp.array([1, 1, 0, 0])})
    chex.assert_trees_all_close(
        pruned_params, {'w': jnp.array([1.0, 2.0, 0.0, 0.0])}
    )

  def testGlobalPruningKeepsTreeType(self):
    updater = global_pruners.GlobalMagnitudePruning(
        filter_fn=lambda *_: True, sparsity=0.5
    )
    w, w2 = jnp.array([0.1, 0.9, -0.2, 0.5]), jnp.array([0.7, 0.8])
    mask, mask2 = jnp.array([0, 1, 0, 0]), jnp.array([1, 1])
    for score_tree, expected_tree in (
        ([w, w2], [mask, mask2]),
        (_Pair(first=w, second=w2), _Pair(first=mask, second=mask2)),
        (
            flax.core.freeze({'w': w, 'w2': w2}),
            flax.core.freeze({'w': mask, 'w2': mask2}),
        ),
    ):
      masks = updater.create_masks(score_tree, target_sparsity=0.5)
      self.assertIs(type(masks), type(score_tree))
      chex.assert_trees_all_close(masks, expected_tree)

  def testFilterAndCustomMapUseLeafNames(self):
    mlp = eqx.nn.MLP(4, 2, 8, 1, key=jax.random.PRNGKey(0))
    params = eqx.filter(mlp, eqx.is_inexact_array)
    updater = global_pruners.GlobalMagnitudePruning(
        sparsity=0.5, custom_sparsity_map={('layers', 1, 'weight'): 0.25}
    )
    masks = updater.create_masks(params, target_sparsity=0.5)
    # One dimensional leaves are filtered out by default.
    self.assertIsNone(masks.layers[0].bias)
    self.assertIsNone(masks.layers[1].bias)
    # The only leaf left for the global ranking gets the global sparsity.
    self.assertEqual(float(jnp.mean(masks.layers[0].weight)), 0.5)
    self.assertEqual(float(jnp.mean(masks.layers[1].weight)), 0.75)

  def testCustomMapMayCoverEveryPrunableLeaf(self):
    updater = global_pruners.GlobalMagnitudePruning(
        sparsity=0.5, custom_sparsity_map={('w',): 0.5}
    )
    score_tree = {'w': jnp.array([[0.1, 0.9], [-0.2, 0.5]]), 'b': jnp.ones(2)}
    masks = updater.create_masks(score_tree, target_sparsity=0.5)
    chex.assert_trees_all_close(masks['w'], jnp.array([[0, 1], [0, 1]]))
    self.assertIsNone(masks['b'])


if __name__ == '__main__':
  absltest.main()
