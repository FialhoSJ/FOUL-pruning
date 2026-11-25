"""Shape and gradient checks for pretrained pet classification and pruning."""

import unittest
import json
import os
import tempfile

import numpy as np
import torch

from flcore.clients.clientfoul import clientFOUL
from flcore.servers.foul_pruning import prune_model
from flcore.trainmodel.hierarchical_resnet import HierarchicalResNet18


class PetResNetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_frozen_features_train_heads_and_finetuning_reaches_last_block(self):
        model = HierarchicalResNet18(2, 37, image_size=64, pretrained=False)
        model.train()
        self.assertFalse(model.backbone.training)
        family, breed = model(torch.randn(2, 512))
        (family.sum() + breed.sum()).backward()
        self.assertIsNotNone(model.fc1.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

        model.zero_grad(set_to_none=True)
        model.unfreeze_last_block()
        family, breed = model(torch.rand(2, 3, 64, 64))
        (family.sum() + breed.sum()).backward()
        self.assertIsNotNone(model.backbone.layer4[1].conv2.weight.grad)
        self.assertFalse(model.backbone.training)
        model.freeze_encoder()
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))

    def test_pruning_shrinks_projection_and_preserves_outputs(self):
        model = HierarchicalResNet18(2, 37, image_size=64, hidden_dim=8, pretrained=False)
        summary = {"fc1": {"sums": {
            0: torch.tensor([100., 90., 80., 70., 1., 1., 1., 1.]),
            1: torch.ones(8),
        }, "counts": {0: 1, 1: 1}}}
        compact, counts = prune_model(model, summary, target_breed_id=0, pruning_rate=0.5)
        self.assertEqual(counts["fc1"], {"before": 8, "kept": 4, "removed": 4})
        self.assertEqual(compact.fc1.out_features, 4)
        self.assertTrue(torch.equal(model.backbone.conv1.weight, compact.backbone.conv1.weight))
        self.assertEqual(tuple(compact(torch.rand(3, 512))[0].shape), (3, 2))
        self.assertEqual(tuple(compact(torch.rand(3, 512))[1].shape), (3, 37))

    def test_client_caches_only_frozen_encoder_features(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as root:
            dataset_dir = os.path.join(root, "OxfordPet")
            os.makedirs(os.path.join(dataset_dir, "train"))
            with open(os.path.join(dataset_dir, "config.json"), "w", encoding="utf-8") as output:
                json.dump({"family_to_idx": {"cat": 0, "dog": 1},
                           "breed_to_idx": {"Chihuahua": 0, "Pug": 1}}, output)
            images = np.random.default_rng(1).integers(0, 256, (2, 64, 64, 3), dtype=np.uint8)
            np.savez_compressed(os.path.join(dataset_dir, "train", "0.npz"), data={
                "x": images, "y": np.array([1, 1]), "y_subclass": np.array([0, 1]),
            })
            client = object.__new__(clientFOUL)
            client.model = HierarchicalResNet18(2, 2, image_size=64, pretrained=False)
            client.args = SimpleNamespace(dataset_root=root)
            client.dataset = "OxfordPet"
            client.id = 0
            client.batch_size = 2
            client.device = "cpu"
            cached = client._load_hierarchical_data()
            self.assertIs(cached, client._load_hierarchical_data())
            self.assertEqual(tuple(cached.tensors[0].shape), (2, 512))
            client.model.unfreeze_last_block()
            raw = client._load_hierarchical_data()
            self.assertEqual(tuple(raw.tensors[0].shape), (2, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
