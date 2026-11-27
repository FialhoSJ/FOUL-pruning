"""Checks for controlled subclass unlearning experiments."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from test_pet_experiment import clientFOUL, FOUL, FixedPredictions
from utils.pet_protocol import split_indices, load_pet_split, apply_encoder_policy
from flcore.trainmodel.hierarchical_cnn import HierarchicalCNN
from flcore.trainmodel.hierarchical_resnet import HierarchicalResNet18
from flcore.servers.foul_pruning import prune_model


class ProtocolTests(unittest.TestCase):
    def test_full_example_retain_excludes_target_from_both_heads(self):
        class Heads(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.family = torch.nn.Linear(2, 2)
                self.breed = torch.nn.Linear(2, 3)
            def forward(self, x):
                return self.family(x), self.breed(x)
        client = object.__new__(clientFOUL)
        client.device = 'cpu'
        client.args = SimpleNamespace(pet_forget_scope='examples', breed_loss_weight=2.)
        client.loss = torch.nn.CrossEntropyLoss()
        model = Heads()
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        target_only = (torch.tensor([[1., 2.]]), torch.tensor([1]), torch.tensor([0]))
        self.assertEqual(client._hierarchical_step(model, optimizer, target_only, 0, 'retain'), 0)
        self.assertTrue(all(torch.equal(p, before[k]) for k, p in model.named_parameters()))
        mixed = (torch.tensor([[1., 2.], [2., 1.]]), torch.tensor([1, 0]), torch.tensor([0, 2]))
        client._hierarchical_step(model, optimizer, mixed, 0, 'retain')
        retained_family = model.family.weight.detach().clone()
        alternate = Heads()
        alternate.load_state_dict({k: v.clone() for k, v in before.items()})
        client._hierarchical_step(alternate, torch.optim.SGD(alternate.parameters(), lr=.1),
                                  (mixed[0][1:], mixed[1][1:], mixed[2][1:]), 0, 'retain')
        self.assertTrue(torch.allclose(retained_family, alternate.family.weight))
        self.assertTrue(torch.allclose(model.breed.weight, alternate.breed.weight))

    def test_magnitude_pruning_uses_weights_without_target_summaries(self):
        model = HierarchicalCNN(2, 3, image_size=32, channels=(4, 8, 8), hidden_dim=8, pool_grid=1)
        with torch.no_grad():
            for index in range(4):
                model.conv1.weight[index].fill_(index + 1)
        compact, _ = prune_model(model, {}, 0, .5, strategy='magnitude')
        self.assertEqual(compact.conv1.out_channels, 2)
        self.assertTrue(torch.equal(compact.conv1.weight[0], model.conv1.weight[2]))
        self.assertTrue(torch.equal(compact.conv1.weight[1], model.conv1.weight[3]))

    def test_final_test_is_logged_without_influencing_stopping_history(self):
        server = object.__new__(FOUL)
        server.args = SimpleNamespace(pet_validation_fraction=.2)
        server.device, server.batch_size = 'cpu', 2
        server.global_model = FixedPredictions()
        server.rs_test_acc, server.rs_train_loss, server.hierarchical_metrics = [], [], []
        server._resolve_target_breed = lambda: (0, 'target')
        server._hierarchical_config = lambda: {'family_to_idx': {'cat': 0, 'dog': 1},
                                              'parent_by_breed': {'target': 1}}
        data = TensorDataset(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([1, 0]), torch.tensor([0, 1]))
        server.clients = [SimpleNamespace(_load_hierarchical_data=lambda **kwargs: data)]
        server._evaluate_hierarchical('evaluation')
        before = list(server.rs_test_acc)
        metrics = server._evaluate_hierarchical('after_learning_test', split='test')
        self.assertEqual(metrics['split'], 'test')
        self.assertEqual(server.rs_test_acc, before)
        self.assertEqual(len(server.hierarchical_metrics), 2)

    def test_relearning_leaves_checkpoint_model_untouched(self):
        from evaluate_pet_representations import relearn
        model = HierarchicalCNN(2, 3, image_size=32, channels=(4, 8, 8), hidden_dim=8, pool_grid=1)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        data = {'x': torch.rand(6, 3, 32, 32), 'y': torch.tensor([1, 1, 0, 1, 1, 0]),
                'y_subclass': torch.tensor([0, 1, 2, 0, 1, 2])}
        args = SimpleNamespace(seed=1, relearn_lr=.01, relearn_per_breed=1,
                               relearn_steps=2, relearn_eval_every=1, batch_size=3, device='cpu')
        records = relearn(model, data, data, 0, args)
        self.assertEqual([row['step'] for row in records], [0, 1, 2])
        self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in before.items()))

    def test_retain_server_aggregates_weighted_updates_without_forget(self):
        import copy
        server = object.__new__(FOUL)
        server.args = SimpleNamespace(pet_encoder_unlearning='full', pruning_rate=0,
                                      pet_unlearning_method='retain', min_target_accuracy=.1)
        server.global_model = torch.nn.Linear(1, 1, bias=False)
        server.global_model.weight.data.zero_()
        server.device, server.meta_lr, server.unlearn_round = 'cpu', 1., 1
        server.eval_gap, server.auto_break = 1, False
        server.Budget, server.experiment_times = [], {}
        server._resolve_target_breed = lambda: (0, 'target')
        server._evaluate_hierarchical = lambda *args, **kwargs: {'target_breed_accuracy': .9}
        server._record_model_cost = lambda *args: None
        server._compare_clip_selection = lambda *args: None
        server._sync_hierarchical_clients = lambda **kwargs: None
        server.send_models = lambda: None
        server.save_results = lambda: None
        server._save_pet_treatment_model = lambda *args: None
        server._save_hierarchical_metrics = lambda: None
        clients = []
        for count, delta in [(1, 1.), (3, 3.)]:
            def update(model, target, include_forget, count=count, delta=delta):
                self.assertFalse(include_forget)
                local = copy.deepcopy(model)
                local.weight.data.add_(delta)
                return {'retain_model': local, 'retain_count': count, 'forget_model': None}
            clients.append(SimpleNamespace(unlearning_group_updates=update, re_init=lambda args: None,
                _load_hierarchical_data=lambda: TensorDataset(torch.ones(1, 1), torch.ones(1), torch.zeros(1))))
        server.clients = clients
        server.select_clients = lambda: clients
        server._unlearn_hierarchical()
        self.assertAlmostEqual(server.global_model.weight.item(), 2.5)
        self.assertEqual(server.completed_unlearning_updates, 1)
        self.assertAlmostEqual(server.update_diagnostics[0]['server_vs_retain_cosine'], 1.0)
        self.assertIsNone(server.update_diagnostics[0]['server_vs_forget_cosine'])

    def test_checkpoint_preserves_baseline_and_rejects_changed_split(self):
        server = object.__new__(FOUL)
        server.dataset, server.algorithm, server.goal, server.times = 'OxfordPet', 'FOUL', 'check', 0
        server.device = 'cpu'
        model = HierarchicalCNN(2, 3, image_size=32, channels=(4, 8, 8), hidden_dim=8, pool_grid=1)
        config = {'family_to_idx': {'cat': 0, 'dog': 1},
                  'breed_to_idx': {'a': 0, 'b': 1, 'c': 2}, 'image_size': 32}
        server._hierarchical_config = lambda: config
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'baseline.pt'
            torch.save(model, path)
            before = path.read_bytes()
            server.args = SimpleNamespace(pet_checkpoint=str(path), model=model, pet_validation_fraction=.2)
            with self.assertRaisesRegex(ValueError, 'Legacy'):
                server.load_model()
            server.args.pet_validation_fraction = 0
            server.load_model()
            self.assertEqual(path.read_bytes(), before)
            server.args.model = HierarchicalResNet18(2, 3, pretrained=False)
            with self.assertRaisesRegex(ValueError, 'architecture'):
                server.load_model()

    def test_probe_uses_training_information_and_generalizes(self):
        from evaluate_pet_representations import linear_probe
        train = np.array([[-3], [-2], [2], [3]])
        result = linear_probe(train, np.array([0, 0, 1, 1]),
                              np.array([[-1], [1]]), np.array([0, 1]), 1)
        self.assertEqual(result['auroc'], 1)
        self.assertEqual(result['balanced_accuracy'], 1)

    def test_validation_is_disjoint_reproducible_and_keeps_singletons(self):
        labels = np.array([0] * 10 + [1] * 10 + [2])
        train, val = split_indices(labels, .2, 7)
        self.assertEqual(len(val), 4)
        self.assertIn(20, train)
        self.assertEqual(set(train) & set(val), set())
        self.assertEqual(set(train) | set(val), set(range(21)))
        np.testing.assert_array_equal(val, split_indices(labels, .2, 7)[1])

    def test_validation_reads_training_partition(self):
        data = {'x': torch.arange(20)[:, None], 'y': torch.zeros(20),
                'y_subclass': torch.tensor([0] * 10 + [1] * 10)}
        with patch('utils.pet_protocol.read_hierarchical_client_data', return_value=data) as read:
            train = load_pet_split('pets', 0, 'root', 'train', .2, 7)
            val = load_pet_split('pets', 0, 'root', 'validation', .2, 7)
        self.assertTrue(all(call.args[2] for call in read.call_args_list))
        self.assertEqual(set(train['x'].flatten().tolist()) & set(val['x'].flatten().tolist()), set())

    def test_random_pruning_shrinks_convolutions_and_is_reproducible(self):
        model = HierarchicalCNN(2, 3, image_size=32, channels=(4, 8, 8), hidden_dim=8, pool_grid=1)
        a, counts = prune_model(model, {}, 0, .5, strategy='random', seed=5)
        b, _ = prune_model(model, {}, 0, .5, strategy='random', seed=5)
        self.assertEqual(a.conv1.out_channels, 2)
        self.assertEqual(model.conv1.out_channels, 4)
        self.assertTrue(torch.equal(a.conv1.weight, b.conv1.weight))
        self.assertLess(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in model.parameters()))
        self.assertEqual(a(torch.rand(2, 3, 32, 32))[1].shape, (2, 3))

    def test_full_encoder_receives_gradients_after_pruning(self):
        torch.set_num_threads(2)
        model = HierarchicalResNet18(2, 3, image_size=32, hidden_dim=8, pretrained=False)
        apply_encoder_policy(model, 'full')
        model, _ = prune_model(model, {}, 0, .5, strategy='random')
        model.train()
        before = model.backbone.conv1.weight.detach().clone()
        bn_before = model.backbone.bn1.running_mean.clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        model(torch.rand(2, 3, 32, 32))[1].sum().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.backbone.conv1.weight))
        self.assertTrue(torch.equal(bn_before, model.backbone.bn1.running_mean))
        with self.assertRaises(ValueError):
            model(torch.rand(2, 512))
        apply_encoder_policy(model, 'frozen')
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))

    def test_retain_control_does_not_compute_forget_model(self):
        client = object.__new__(clientFOUL)
        client.args = SimpleNamespace(pet_forget_scope='attribute')
        client._load_hierarchical_data = lambda **kwargs: TensorDataset(
            torch.ones(2, 2), torch.ones(2), torch.tensor([0, 1]))
        def fit(model, target, group):
            self.assertEqual(group, 'retain')
            return 'retained', 2
        client._fit_unlearning_group = fit
        result = client.unlearning_group_updates(None, 0, include_forget=False)
        self.assertIsNone(result['forget_model'])
        self.assertEqual(result['retain_model'], 'retained')
        self.assertEqual(result['retain_count'], 2)


if __name__ == '__main__':
    unittest.main()
