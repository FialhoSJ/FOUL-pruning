# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
from flcore.clients.clientbase import Client
from utils.hierarchical_data import read_hierarchical_client_data

class clientFOUL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.is_hierarchical = args.dataset in ("OxfordPet", "StanfordDogs")

    def _load_hierarchical_data(self, is_train=True):
        dataset_root = getattr(self.args, "dataset_root", "../dataset")
        data = read_hierarchical_client_data(
            self.dataset, self.id, is_train=is_train, dataset_root=dataset_root
        )
        return TensorDataset(data["x"], data["y"], data["y_subclass"])

    def _max_local_epochs(self):
        if self.train_slow and self.local_epochs > 1:
            return int(np.random.randint(1, max(2, self.local_epochs // 2 + 1)))
        return self.local_epochs

    def _hierarchical_step(self, model, optimizer, batch, target_breed_id=None, group="all"):
        images, families, breeds = (value.to(self.device) for value in batch)
        if group == "forget":
            mask = breeds == int(target_breed_id)
            if not torch.any(mask):
                return 0
            images, breeds = images[mask], breeds[mask]
            _, breed_logits = model(images)
            loss = self.loss(breed_logits, breeds)
        else:
            family_logits, breed_logits = model(images)
            loss = self.loss(family_logits, families)
            if group == "retain":
                mask = breeds != int(target_breed_id)
                if torch.any(mask):
                    loss = loss + self.loss(breed_logits[mask], breeds[mask])
            else:
                loss = loss + self.loss(breed_logits, breeds)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return int(images.shape[0])

    def _train_hierarchical(self):
        loader = DataLoader(
            self._load_hierarchical_data(is_train=True),
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
        )
        self.model.train()
        start_time = time.time()
        for _ in range(self._max_local_epochs()):
            for batch in loader:
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self._hierarchical_step(self.model, self.optimizer, batch)
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def _fit_unlearning_group(self, global_model, target_breed_id, group):
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        loader = DataLoader(
            self._load_hierarchical_data(is_train=True),
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=(group == "retain"),
        )
        sample_count = 0
        for _ in range(self._max_local_epochs()):
            for batch in loader:
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                sample_count += self._hierarchical_step(
                    model, optimizer, batch, target_breed_id=target_breed_id, group=group
                )
        return model, sample_count

    def unlearning_group_updates(self, global_model, target_breed_id):
        """Train independent retain/forget copies from the same global model."""
        data = self._load_hierarchical_data(is_train=True)
        breeds = data.tensors[2]
        target_count = int(torch.sum(breeds == int(target_breed_id)).item())
        retain_count = len(breeds)
        if target_count == 0:
            forget_model = None
        else:
            forget_model, _ = self._fit_unlearning_group(
                global_model, target_breed_id, group="forget"
            )
        if retain_count == 0:
            retain_model = None
        else:
            retain_model, _ = self._fit_unlearning_group(
                global_model, target_breed_id, group="retain"
            )
        return {
            "retain_model": retain_model,
            "forget_model": forget_model,
            "retain_count": retain_count,
            "forget_count": target_count,
        }

    def class_activation_summaries(self):
        """Return per-breed sums/counts of ReLU channel activations only."""
        loader = DataLoader(
            self._load_hierarchical_data(is_train=True),
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )
        if not hasattr(self.model, "feature_layers"):
            raise TypeError("hierarchical FOUL model must expose feature_layers()")

        summaries = {}
        captured = {}
        handles = []
        for name, layer in self.model.feature_layers():
            summaries[name] = {"sums": {}, "counts": {}}

            def capture(_module, _inputs, output, layer_name=name):
                captured[layer_name] = output.detach()

            handles.append(layer.register_forward_hook(capture))

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for images, _families, breeds in loader:
                    captured.clear()
                    self.model(images.to(self.device))
                    for name, activation in captured.items():
                        activation = torch.relu(activation)
                        if activation.ndim == 4:
                            activation = activation.mean(dim=(2, 3))
                        if activation.ndim != 2:
                            raise ValueError(f"feature layer {name} returned shape {tuple(activation.shape)}")
                        activation = activation.cpu()
                        for breed_id in torch.unique(breeds):
                            mask = breeds == breed_id
                            breed_key = int(breed_id.item())
                            class_sum = activation[mask].sum(dim=0)
                            summaries[name]["sums"][breed_key] = (
                                summaries[name]["sums"].get(breed_key, torch.zeros_like(class_sum))
                                + class_sum
                            )
                            summaries[name]["counts"][breed_key] = (
                                summaries[name]["counts"].get(breed_key, 0) + int(mask.sum().item())
                            )
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(was_training)

        return summaries

    def train(self):
        if self.is_hierarchical:
            self._train_hierarchical()
            return
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def unlearn(self):
        if self.is_hierarchical:
            raise RuntimeError(
                "Hierarchical FOUL unlearning is coordinated by the server using "
                "unlearning_group_updates()."
            )
        """ need to define a client unlearn method for the algorithm too
        which will be called in the serveravg train loop and initiate the unlearn method
        therefore we define the unlean method here"""
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
