"""Behavior checks for hierarchical forgetting measurements."""

import unittest
import sys
import types
from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

for optional_module in ("torch.utils.tensorboard", "wandb"):
    try:
        __import__(optional_module)
    except ImportError:
        stub = types.ModuleType(optional_module)
        if optional_module == "torch.utils.tensorboard":
            stub.SummaryWriter = object
        sys.modules[optional_module] = stub

from flcore.servers.serverfoul import FOUL
from flcore.clients.clientfoul import clientFOUL


class FixedPredictions(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.encoder_frozen = True

    def encode(self, features):
        return features

    def forward(self, features):
        families = features[:, 0].long()
        breeds = features[:, 1].long()
        family_logits = torch.full((len(features), 2), -10.0)
        breed_logits = torch.full((len(features), 3), -10.0)
        family_logits.scatter_(1, families[:, None], 10.0)
        breed_logits.scatter_(1, breeds[:, None], 10.0)
        return family_logits, breed_logits


class PetExperimentTests(unittest.TestCase):
    def test_reference_keeps_target_family_but_excludes_target_breed_loss(self):
        class TwoHeads(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.family = torch.nn.Linear(2, 2)
                self.breed = torch.nn.Linear(2, 3)

            def forward(self, x):
                return self.family(x), self.breed(x)

        client = object.__new__(clientFOUL)
        client.device = "cpu"
        client.args = SimpleNamespace(breed_loss_weight=2.0)
        client.loss = torch.nn.CrossEntropyLoss()
        model = TwoHeads()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        family_before = model.family.weight.detach().clone()
        breed_before = model.breed.weight.detach().clone()
        batch = (torch.tensor([[1., 2.], [2., 1.]]),
                 torch.tensor([1, 1]), torch.tensor([0, 0]))

        client._hierarchical_step(model, optimizer, batch, target_breed_id=0, group="retain")
        self.assertFalse(torch.equal(model.family.weight, family_before))
        self.assertTrue(torch.equal(model.breed.weight, breed_before))

    def test_evaluation_separates_target_other_dogs_and_cats(self):
        server = object.__new__(FOUL)
        server.args = SimpleNamespace(breed_loss_weight=2.0)
        server.device = "cpu"
        server.batch_size = 3
        server.global_model = FixedPredictions()
        server.rs_test_acc = []
        server.rs_train_loss = []
        server.hierarchical_metrics = []
        server.model_costs = []
        server._resolve_target_breed = lambda: (0, "Chihuahua")
        server._hierarchical_config = lambda: {
            "family_to_idx": {"cat": 0, "dog": 1},
            "parent_by_breed": {"Chihuahua": 1, "Pug": 1, "Siamese": 0},
        }
        features = torch.tensor([[1., 1.], [1., 1.], [0., 2.]])
        labels = TensorDataset(features, torch.tensor([1, 1, 0]), torch.tensor([0, 1, 2]))
        server.clients = [SimpleNamespace(_load_hierarchical_data=lambda is_train: labels)]

        metrics = server._evaluate_hierarchical("check")
        self.assertEqual(metrics["groups"]["target"], {
            "samples": 1, "breed_accuracy": 0.0, "family_accuracy": 1.0,
        })
        self.assertEqual(metrics["groups"]["same_family_other_breeds"]["breed_accuracy"], 1.0)
        self.assertEqual(metrics["groups"]["other_family"]["breed_accuracy"], 1.0)
        server._record_model_cost("check")
        self.assertEqual(server.model_costs[0]["trainable_parameters"], 1)


if __name__ == "__main__":
    unittest.main()
