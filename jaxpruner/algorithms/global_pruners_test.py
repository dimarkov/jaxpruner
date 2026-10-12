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
import functools
from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized

import chex
import equinox as eqx
import flax
import jax
import jax.numpy as jnp
import optax
from jaxpruner import sparsity_distributions
from jaxpruner import sparsity_schedules
from jaxpruner.algorithms import global_pruners
from jaxpruner.algorithms import pruners


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

  def testInstantSparsifyWithPackedMasks(self):
    updater = global_pruners.GlobalMagnitudePruning(
        sparsity=0.5, use_packed_masks=True
    )
    w = jnp.array([[0.1, 0.9, -0.2, 0.5], [0.7, 0.8, 0.3, -0.4]])
    params = {'w': w, 'b': jnp.ones(4)}
    mask = jnp.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=jnp.uint8)
    pruned_params, masks = updater.instant_sparsify(params)
    chex.assert_trees_all_close(
        pruned_params, {'w': w * mask, 'b': params['b']}
    )
    chex.assert_trees_all_equal(masks['w'], jnp.packbits(mask))
    self.assertIsNone(masks['b'])

  @parameterized.parameters(
      lambda scheduler: pruners.MagnitudePruning(
          scheduler=scheduler,
          sparsity_distribution_fn=functools.partial(
              sparsity_distributions.uniform, sparsity=0.5
          ),
      ),
      lambda scheduler: global_pruners.GlobalMagnitudePruning(
          scheduler=scheduler, sparsity=0.5
      ),
  )
  def testTrainingLoopOnEquinoxModule(self, make_updater):
    updater = make_updater(
        sparsity_schedules.PolynomialSchedule(
            update_freq=1, update_start_step=1, update_end_step=3
        )
    )
    mlp = eqx.nn.MLP(4, 2, 8, 1, key=jax.random.PRNGKey(0))
    params = eqx.filter(mlp, eqx.is_inexact_array)
    tx = updater.wrap_optax(optax.sgd(0.01))
    state = tx.init(params)

    def loss_fn(params):
      return sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(params))

    @jax.jit
    def train_step(params, state):
      grads = jax.grad(loss_fn)(params)
      updates, state = tx.update(grads, state, params)
      params = optax.apply_updates(params, updates)
      return updater.post_gradient_update(params, state), state

    for _ in range(5):
      params, state = train_step(params, state)

    self.assertIsInstance(params, eqx.nn.MLP)
    self.assertIsInstance(state.masks, eqx.nn.MLP)
    weights = [layer.weight for layer in params.layers]
    masks = [layer.weight for layer in state.masks.layers]
    for weight, mask in zip(weights, masks):
      self.assertEqual(mask.shape, weight.shape)
      chex.assert_trees_all_equal(weight != 0, mask != 0)
    num_zeros = sum(int(jnp.sum(weight == 0)) for weight in weights)
    self.assertEqual(num_zeros, sum(weight.size for weight in weights) // 2)
    for params_layer, masks_layer in zip(params.layers, state.masks.layers):
      self.assertIsNone(masks_layer.bias)
      self.assertTrue(bool(jnp.all(params_layer.bias != 0)))

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
