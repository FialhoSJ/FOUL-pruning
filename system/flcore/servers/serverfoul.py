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

import time
import json
import os
from flcore.clients.clientfoul import clientFOUL
from flcore.servers.serverbase import Server
from utils.model_utils import ParamDict
from threading import Thread
import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import copy
from torch.optim.lr_scheduler import StepLR
import numpy as np
import statistics

from torch.utils.tensorboard import SummaryWriter
import wandb
from utils.hierarchical_data import read_hierarchical_client_data
from utils.clip_breed_compare import compare_prompt_cluster
from flcore.servers.foul_pruning import aggregate_activation_summaries, prune_model

class FOUL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFOUL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        """ Learn-Unlearn Round Split """
        self.learn_round = args.learn_round
        self.unlearn_round = args.global_rounds - self.learn_round
        if self.unlearn_round < 0:
            raise ValueError("The global rounds must be higher than learn round")

        # A standalone unlearning run continues from the model saved by the
        # learning stage. Joint runs keep using the in-memory trained model.
        if args.learn_mode == 'unlearn':
            self.load_model()
        self.Budget = []
        self.update_grads = None
        self.foul_c = args.c_parameter
        self.foul_rounds = args.cagrad_rounds
        self.foul_lr = args.cagrad_learning_rate
        self.momentum = args.momentum
        self.step_size = args.step_size
        self.meta_lr = args.meta_lr
        self.gamma = args.gamma
        self.beta = args.beta_foul
        self.device = args.device
        self.fgrad_balance = args.forget_balance
        self.rgrad_balance = args.retain_balance
        self.hierarchical_metrics = []
        self.pruning_counts = None
        self.clip_comparison = None
        self._target_breed_cache = None

    def FUL_evaluate(self, acc=None, loss=None):
        if self._is_hierarchical():
            return self._evaluate_hierarchical("evaluation")
        return super().FUL_evaluate(acc=acc, loss=loss)

    def set_clients(self, clientObj):
        if not self._is_hierarchical():
            return super().set_clients(clientObj)
        dataset_root = getattr(self.args, "dataset_root", "../dataset")
        for client_id, train_slow, send_slow in zip(
            range(self.num_clients), self.train_slow_clients, self.send_slow_clients
        ):
            train_data = read_hierarchical_client_data(
                self.dataset, client_id, is_train=True, dataset_root=dataset_root
            )
            test_data = read_hierarchical_client_data(
                self.dataset, client_id, is_train=False, dataset_root=dataset_root
            )
            client = clientObj(
                self.args,
                id=client_id,
                train_samples=len(train_data["y"]),
                test_samples=len(test_data["y"]),
                train_slow=train_slow,
                send_slow=send_slow,
            )
            self.clients.append(client)

    def _is_hierarchical(self):
        return self.dataset in ("OxfordPet", "StanfordDogs")

    def _hierarchical_config(self):
        dataset_root = getattr(self.args, "dataset_root", "../dataset")
        config_path = os.path.join(dataset_root, self.dataset, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Missing pet dataset config: {config_path}")
        with open(config_path, "r", encoding="utf-8") as config_file:
            return json.load(config_file)

    def _resolve_target_breed(self):
        if self._target_breed_cache is not None:
            return self._target_breed_cache
        config = self._hierarchical_config()
        breed_to_idx = config.get("breed_to_idx", {})
        requested = getattr(self.args, "forget_breed", None)
        if requested is None:
            requested = getattr(self.args, "forget_class", None)
        if requested is None:
            raise ValueError("Set -forget_breed to an annotated breed name or numeric breed ID")

        name_to_id = {str(name).casefold(): int(index) for name, index in breed_to_idx.items()}
        requested_text = str(requested).strip()
        if requested_text.casefold() in name_to_id:
            target_id = name_to_id[requested_text.casefold()]
            target_name = next(name for name, index in breed_to_idx.items() if int(index) == target_id)
        else:
            try:
                target_id = int(requested_text)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown breed {requested!r}; available breeds: {', '.join(sorted(breed_to_idx))}"
                ) from exc
            target_by_id = {int(index): name for name, index in breed_to_idx.items()}
            if target_id not in target_by_id:
                raise ValueError(
                    f"Unknown breed ID {target_id}; valid IDs are 0 through {len(target_by_id) - 1}"
                )
            target_name = target_by_id[target_id]

        self._target_breed_cache = (target_id, target_name)
        return self._target_breed_cache

    def _evaluate_hierarchical(self, stage):
        target_id, target_name = self._resolve_target_breed()
        dataset_root = getattr(self.args, "dataset_root", "../dataset")
        config = self._hierarchical_config()
        num_families = len(config["family_to_idx"])
        totals = {
            "samples": 0,
            "breed_correct": 0,
            "family_correct": 0,
            "family_samples": 0,
            "target_correct": 0,
            "target_samples": 0,
            "other_correct": 0,
            "other_samples": 0,
            "loss_sum": 0.0,
            "family_loss_sum": 0.0,
            "breed_loss_sum": 0.0,
            "family_predictions": [0] * num_families,
            "family_labels": [0] * num_families,
        }
        self.global_model.eval()
        with torch.no_grad():
            for client in self.clients:
                if hasattr(self.global_model, "encode") and self.global_model.encoder_frozen:
                    images, families, breeds = client._load_hierarchical_data(
                        is_train=False
                    ).tensors
                else:
                    data = read_hierarchical_client_data(
                        self.dataset, client.id, is_train=False, dataset_root=dataset_root
                    )
                    images, families, breeds = data["x"], data["y"], data["y_subclass"]
                for start in range(0, len(images), self.batch_size):
                    end = start + self.batch_size
                    x = images[start:end].to(self.device)
                    family = families[start:end].to(self.device)
                    breed = breeds[start:end].to(self.device)
                    family_logits, breed_logits = self.global_model(x)
                    breed_prediction = breed_logits.argmax(dim=1)
                    breed_correct = breed_prediction == breed
                    totals["samples"] += len(breed)
                    totals["breed_correct"] += int(breed_correct.sum().item())
                    family_loss = float(torch.nn.functional.cross_entropy(family_logits, family).item())
                    breed_loss = float(torch.nn.functional.cross_entropy(breed_logits, breed).item())
                    totals["family_loss_sum"] += family_loss * len(breed)
                    totals["breed_loss_sum"] += breed_loss * len(breed)
                    totals["loss_sum"] += (
                        family_loss + float(getattr(self.args, "breed_loss_weight", 2.0)) * breed_loss
                    ) * len(breed)
                    if num_families > 1:
                        totals["family_samples"] += len(family)
                        family_prediction = family_logits.argmax(dim=1)
                        totals["family_correct"] += int((family_prediction == family).sum().item())
                        for family_id in range(num_families):
                            totals["family_predictions"][family_id] += int(
                                (family_prediction == family_id).sum().item()
                            )
                            totals["family_labels"][family_id] += int(
                                (family == family_id).sum().item()
                            )
                    target_mask = breed == target_id
                    other_mask = ~target_mask
                    totals["target_samples"] += int(target_mask.sum().item())
                    totals["other_samples"] += int(other_mask.sum().item())
                    totals["target_correct"] += int((breed_correct & target_mask).sum().item())
                    totals["other_correct"] += int((breed_correct & other_mask).sum().item())

        if totals["samples"] == 0:
            raise ValueError("The pet dataset has no test examples")
        metrics = {
            "stage": stage,
            "round": int(getattr(self, "current_round", 0)),
            "target_breed": target_name,
            "target_breed_accuracy": (
                totals["target_correct"] / totals["target_samples"] if totals["target_samples"] else None
            ),
            "target_breed_samples": totals["target_samples"],
            "non_target_breed_accuracy": (
                totals["other_correct"] / totals["other_samples"] if totals["other_samples"] else None
            ),
            "non_target_breed_samples": totals["other_samples"],
            "family_accuracy": (
                totals["family_correct"] / totals["family_samples"] if totals["family_samples"] else None
            ),
            "overall_breed_accuracy": totals["breed_correct"] / totals["samples"],
            "test_loss": totals["loss_sum"] / totals["samples"],
            "family_test_loss": totals["family_loss_sum"] / totals["samples"],
            "breed_test_loss": totals["breed_loss_sum"] / totals["samples"],
            "family_prediction_counts": totals["family_predictions"] if num_families > 1 else None,
            "family_label_counts": totals["family_labels"] if num_families > 1 else None,
        }
        self.rs_test_acc.append(metrics["overall_breed_accuracy"])
        self.rs_train_loss.append(metrics["test_loss"])
        self.hierarchical_metrics.append(metrics)

        print(f"Family accuracy: {metrics['family_accuracy'] if metrics['family_accuracy'] is not None else 'N/A'}")
        if num_families > 1:
            print(f"Family predictions / labels: {metrics['family_prediction_counts']} / "
                  f"{metrics['family_label_counts']}")
        print(f"Target breed ({target_name}) accuracy: {metrics['target_breed_accuracy']}")
        print(f"Non-target breed accuracy: {metrics['non_target_breed_accuracy']}")
        print(f"Overall breed accuracy: {metrics['overall_breed_accuracy']:.4f}")
        print(f"Family/Breed test loss: {metrics['family_test_loss']:.4f} / {metrics['breed_test_loss']:.4f}")
        return metrics

    def _save_hierarchical_metrics(self):
        if not self.hierarchical_metrics:
            return
        results_dir = "../results"
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(
            results_dir,
            f"{self.dataset}_{self.algorithm}_{self.goal}_{self.times}_hierarchy.json",
        )
        with open(output_path, "w", encoding="utf-8") as result_file:
            json.dump(
                {
                    "metrics": self.hierarchical_metrics,
                    "pruning_counts": self.pruning_counts,
                    "clip_comparison": self.clip_comparison,
                },
                result_file,
                indent=2,
            )

    def _sync_hierarchical_clients(self, reset_optimizers=False):
        for client in self.clients:
            client.model = copy.deepcopy(self.global_model).to(self.device)
            if reset_optimizers:
                client.optimizer = client.make_optimizer(client.model)
                client.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    client.optimizer, gamma=self.args.learning_rate_decay_gamma
                )
                client.learning_rate_decay = self.args.learning_rate_decay

    @staticmethod
    def _model_delta_vector(local_model, global_model):
        deltas = []
        local_parameters = [p for p in local_model.parameters() if p.requires_grad]
        global_parameters = [p for p in global_model.parameters() if p.requires_grad]
        if len(local_parameters) != len(global_parameters):
            raise ValueError("local and global model parameter counts do not match")
        for local_parameter, global_parameter in zip(local_parameters, global_parameters):
            if local_parameter.shape != global_parameter.shape:
                raise ValueError("local and global model parameter shapes do not match")
            deltas.append((local_parameter.detach() - global_parameter.detach()).reshape(-1))
        return torch.cat(deltas).cpu()

    def _compare_clip_selection(self, target_id, target_name):
        if not getattr(self.args, "clip_compare", False):
            return
        config = self._hierarchical_config()
        family_id = config["parent_by_breed"][str(target_id)]
        dataset_root = getattr(self.args, "dataset_root", "../dataset")
        images_by_breed = {}
        for client in self.clients:
            data = read_hierarchical_client_data(
                self.dataset, client.id, is_train=True, dataset_root=dataset_root
            )
            same_family = data["y"] == family_id
            for image, breed in zip(data["x"][same_family], data["y_subclass"][same_family]):
                images_by_breed.setdefault(int(breed), []).append(image)

        target_images = images_by_breed.get(target_id, [])
        if not target_images:
            raise ValueError(f"CLIP comparison found no examples for target breed {target_name}")
        max_samples = max(int(getattr(self.args, "clip_max_samples", 1000)), 2)
        rng = np.random.default_rng(int(getattr(self.args, "seed", 1)))
        target_cap = max(1, max_samples // 2)
        target_indices = rng.permutation(len(target_images))[:target_cap]
        selected_images = [target_images[int(index)] for index in target_indices]
        selected_labels = [target_id] * len(selected_images)

        other_images = [
            (breed_id, image)
            for breed_id in sorted(images_by_breed)
            if breed_id != target_id
            for image in images_by_breed[breed_id]
        ]
        remaining = max_samples - len(selected_images)
        if remaining > 0 and other_images:
            other_indices = rng.permutation(len(other_images))[:remaining]
            for index in other_indices:
                breed_id, image = other_images[int(index)]
                selected_images.append(image)
                selected_labels.append(breed_id)
        if len(selected_images) < 2:
            raise ValueError("CLIP comparison needs at least two images from the target family")

        prompt = getattr(self.args, "clip_prompt", None) or f"a photo of a {target_name}"
        clusters = min(int(getattr(self.args, "clip_clusters", 5)), len(selected_images))
        self.clip_comparison = compare_prompt_cluster(
            torch.stack(selected_images),
            selected_labels,
            target_id,
            prompt,
            num_clusters=clusters,
            batch_size=self.batch_size,
            device=self.device,
        )
        print(f"CLIP prompt/cluster comparison: {self.clip_comparison}")

    def _unlearn_hierarchical(self):
        target_id, target_name = self._resolve_target_breed()
        if hasattr(self.global_model, "freeze_encoder"):
            self.global_model.freeze_encoder()
            for client in self.clients:
                client._pet_feature_cache = {}
            self._sync_hierarchical_clients(reset_optimizers=True)
        print("\n======================================")
        print("\nFED Hierarchical Unlearning Stage")
        print(f"Forget breed: {target_name} (id={target_id})")

        baseline = self._evaluate_hierarchical("before_pruning")
        minimum_accuracy = float(getattr(self.args, "min_target_accuracy", 0.1))
        if baseline["target_breed_accuracy"] is None or baseline["target_breed_accuracy"] < minimum_accuracy:
            raise ValueError(
                f"Target breed accuracy before unlearning is {baseline['target_breed_accuracy']}; "
                f"at least {minimum_accuracy:.1%} is required. Train longer before forgetting."
            )

        self._compare_clip_selection(target_id, target_name)

        self._sync_hierarchical_clients()
        summaries = [client.class_activation_summaries() for client in self.clients]
        aggregated_summaries = aggregate_activation_summaries(summaries)
        reference_layer = next(iter(aggregated_summaries.values()))
        all_samples = sum(reference_layer["counts"].values())
        target_samples = reference_layer["counts"].get(target_id, 0)
        if all_samples == 0 or target_samples == 0:
            raise ValueError(
                f"Target breed {target_name} has no federated training samples; "
                "the trained checkpoint was not changed"
            )

        rate = float(getattr(self.args, "pruning_rate", 0.5))
        compact_model, self.pruning_counts = prune_model(
            self.global_model, aggregated_summaries, target_id, pruning_rate=rate
        )
        self.global_model = compact_model.to(self.device)
        self._sync_hierarchical_clients(reset_optimizers=True)
        print(f"Pruning rate: {rate:.0%}")
        print(f"Pruning counts: {self.pruning_counts}")
        self._evaluate_hierarchical("after_pruning")

        for client in self.clients:
            client.re_init(self.args)

        successful_updates = 0
        for round_index in range(self.unlearn_round):
            round_start = time.time()
            self.current_round = round_index
            selected_clients = self.select_clients()
            self.selected_clients = selected_clients
            self.send_models()

            if round_index % self.eval_gap == 0:
                print(f"\n-------------Unlearning round: {round_index}-------------")
                self._evaluate_hierarchical("unlearning")

            retain_updates, forget_updates = [], []
            for client in selected_clients:
                groups = client.unlearning_group_updates(self.global_model, target_id)
                if groups["retain_model"] is not None:
                    retain_updates.append(self._model_delta_vector(groups["retain_model"], self.global_model))
                if groups["forget_model"] is not None:
                    forget_updates.append(self._model_delta_vector(groups["forget_model"], self.global_model))

            if not forget_updates:
                print("No selected client has target-breed samples this round; skipping model update.")
                self.Budget.append(time.time() - round_start)
                continue
            if not retain_updates:
                raise ValueError("No retain update was produced; compact checkpoint was not saved")

            retain_tensor = torch.stack(retain_updates)
            forget_tensor = torch.stack(forget_updates)
            update = self.foul_update(
                retain_tensor,
                forget_tensor,
                len(retain_updates) + len(forget_updates),
            )
            trainable_parameters = [
                parameter for parameter in self.global_model.parameters()
                if parameter.requires_grad
            ]
            model_vector = parameters_to_vector(trainable_parameters)
            vector_to_parameters(
                model_vector + update.to(model_vector.device) * self.meta_lr,
                trainable_parameters,
            )
            successful_updates += 1
            self.Budget.append(time.time() - round_start)
            print("Completed sample-group FOUL aggregation.")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        if successful_updates == 0:
            raise ValueError("No FOUL update completed; compact checkpoint was not saved")
        self._sync_hierarchical_clients()
        self._evaluate_hierarchical("after_unlearning")
        self.save_results()
        self.save_global_model()
        self._save_hierarchical_metrics()
        print(f"Compact unlearned model saved; per-layer channels: {self.pruning_counts}")

    def train(self):
        print("\n======================================")
        print("\nFED Learning Stage")
        for i in range(self.learn_round if self._is_hierarchical() else self.learn_round + 1):
            s_t = time.time()
            if (self._is_hierarchical() and hasattr(self.global_model, "unfreeze_last_block")
                    and i == getattr(self.args, "pet_unfreeze_round", -1)):
                self.global_model.unfreeze_last_block()
                for client in self.clients:
                    client._pet_feature_cache = {}
                self._sync_hierarchical_clients(reset_optimizers=True)
                print(f"Fine-tuning ResNet18 layer4 from learning round {i}")
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.FUL_evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            old_model = None if self._is_hierarchical() else copy.deepcopy(self.global_model)
            self.aggregate_parameters()

            if self._is_hierarchical():
                self.Budget.append(time.time() - s_t)
                print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break
                continue

            """
            Metrics of gradient angles here
            - Cosines between (gradients of retain set) vs. (aggregated gradient) 
            """
            r_angle_dict = {}
            f_angle_dict = {}
            # print(self.forget_list)
            for client in self.selected_clients:
                cos = self.cos_sim(old_model, self.global_model, client.model)
                if client.id in self.forget_list:
                    f_angle_dict[f"{client.id}"] = cos
                else:
                    r_angle_dict[f"{client.id}"] = cos
                if self.args.log:
                    self.writer.add_scalar(f"client-charts/client{client.id}_angle", cos, self.current_round)
                    wandb.log({f"client-charts/client{client.id}_angle": cos}, step=self.current_round)

            print(f"======= Client Angle =======")
            f_angle_dict = {k: f_angle_dict[k] for k in sorted(f_angle_dict, reverse=True)}
            r_angle_dict = {k: r_angle_dict[k] for k in sorted(r_angle_dict, reverse=True)}
            print(r_angle_dict)
            print(f_angle_dict)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        if self._is_hierarchical():
            self._evaluate_hierarchical("after_learning")

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        round_costs = self.Budget[1:] if len(self.Budget) > 1 else self.Budget
        print(sum(round_costs) / len(round_costs))

        self.save_results()
        self.save_global_model()
        if self._is_hierarchical():
            self._save_hierarchical_metrics()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFOUL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def unlearn(self):
        """ Need to put the fed unlearning part here which probably has a similar structure to the above learning part
        need to copy that code and configure it

        create a separate function for the unlearning part here

        for client in self.selected_clients:    ## (Question) in the unlearn function we need to set the clients to
                                                ##  unlearn ---> but when it comes to client unlearn what data we
                                                ##  need to specify right ??????
                client.unlearn()
        """
        if self._is_hierarchical():
            self._unlearn_hierarchical()
            return

        ### Init Everything
        for client in self.clients:
            client.re_init(self.args)

        ### Fed Client wise forgetting stage
        print("\n======================================")
        print("\nFED Unlearning Stage")
        for i in range(self.unlearn_round + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.FUL_evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            # self.receive_grads()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            old_model = copy.deepcopy(self.global_model)

            # self.aggregate_parameters(
            # self.aggregate_foul()
            meta_weights = self.aggregate_foul(
                meta_weights=self.global_model,
                selected_clients=self.uploaded_models,
                selected_id=self.uploaded_ids,
                lr_meta=self.meta_lr
            )
            self.global_model.load_state_dict(copy.deepcopy(meta_weights))
            # self.network.reset_weights(meta_weights)

            """
            Metrics of gradient angles here
            - Cosines between (gradients of retain set) vs. (aggregated gradient) 
            """
            r_angle_dict = {}
            f_angle_dict = {}
            # print(self.forget_list)
            for client in self.selected_clients:
                cos = self.cos_sim(old_model, self.global_model, client.model)
                norm = self.diff_weight(old_model, client.model)
                if client.id in self.forget_list:
                    f_angle_dict[f"{client.id}"] = cos
                else:
                    r_angle_dict[f"{client.id}"] = cos
                if self.args.log:
                    self.writer.add_scalar(f"client-charts/client{client.id}_angle", cos, self.current_round)
                    wandb.log({f"client-charts/client{client.id}_angle": cos}, step=self.current_round)
                    self.writer.add_scalar(f"client-charts/client{client.id}_norm", norm, self.current_round)
                    wandb.log({f"client-charts/client{client.id}_norm": norm}, step=self.current_round)

            print(f"======= Client Angle =======")
            print(r_angle_dict)
            print(f_angle_dict)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(self.foul_c = args.c_parameter
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFOUL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.FUL_evaluate()

    def aggregate_foul(self, meta_weights, selected_clients, selected_id, lr_meta):
        # Lấy tất cả parameter names
        param_names = [name for name, _ in meta_weights.named_parameters()]

        """
        Hyper-gradients calculation & merge
        - In this section, the all_domains_grad_tensor is arranged according to the selected_id
        """
        all_client_grads = []
        retain_grads = []
        forget_grads = []

        for i_client, client in zip(selected_id, selected_clients):
            client_grad = [torch.flatten(inner_param - meta_param) for inner_param, meta_param in
                                 zip(client.parameters(), meta_weights.parameters())]
            client_grad_vector = torch.cat(client_grad).cpu()
            # all_client_grads.append(client_grad_vector)
            if i_client in self.forget_list: # This part should be verified in the future to make it flexible to the data.
                forget_grads.append(client_grad_vector)
            else:
                retain_grads.append(client_grad_vector)

        """
        - Retain Grads normalization.
        """
        if self.rgrad_balance:
            # Apply balancing
            # Step 1: Compute norms for each gradient vector
            retain_grad_norms = [torch.norm(grad) for grad in retain_grads]

            # Step 2: Determine scaling factors to balance the norms
            # Example: Scale all norms to a target value (e.g., the average norm)
            target_norm = torch.mean(torch.tensor(retain_grad_norms))
            scaling_factors = [target_norm / norm if norm > 0 else 1.0 for norm in retain_grad_norms]

            # Step 3: Scale gradient vectors
            balanced_retain_grads = [grad * scale for grad, scale in zip(retain_grads, scaling_factors)]

            if self.args.log:
                for i, client_grad in enumerate(balanced_retain_grads):
                    diff_norms = torch.norm(client_grad)
                    wandb.log({f"client-charts/client{i}_norm_balance":diff_norms}, step=self.current_round)

            # Step 4: Stack the balanced gradients into a tensor
            retain_grad_tensor = torch.stack(balanced_retain_grads).cpu()
        else:
            retain_grad_tensor = torch.stack(retain_grads).cpu()



        """
        - Forget Grads normalization.
        """
        if self.fgrad_balance:
            # Apply balancing
            # Step 1: Compute norms for each gradient vector
            forget_grad_norms = [torch.norm(grad) for grad in forget_grads]

            # Step 2: Determine scaling factors to balance the norms
            # Example: Scale all norms to a target value (e.g., the average norm)
            target_norm = torch.mean(torch.tensor(forget_grad_norms))
            scaling_factors = [target_norm / norm if norm > 0 else 1.0 for norm in forget_grad_norms]

            # Step 3: Scale gradient vectors
            balanced_forget_grads = [grad * scale for grad, scale in zip(forget_grads, scaling_factors)]

            if self.args.log:
                for i, client_id in enumerate(self.forget_list):
                    diff_norms = torch.norm(balanced_forget_grads[i])
                    wandb.log({f"client-charts/client{client_id}_norm_balance": diff_norms}, step=self.current_round)

            # Step 4: Stack the balanced gradients into a tensor
            forget_grad_tensor = torch.stack(balanced_forget_grads).cpu()
        else:
            forget_grad_tensor = torch.stack(forget_grads).cpu()

        # all_domains_grad_tensor = torch.stack(all_client_grads)
        foul_grad = self.foul_update(retain_grad_tensor, forget_grad_tensor, len(selected_clients))
        # foul_grad = self.foul_update(all_domains_grad_tensor, len(selected_clients))

        # """ Update Grad to Model Parameters """
        # print(f"foul_grad: {foul_grad.size()}")

        meta_weights_vector = parameters_to_vector(meta_weights.parameters())
        vector_to_parameters(meta_weights_vector + foul_grad * lr_meta, meta_weights.parameters())

        # Tạo và in ra ParamDict mới từ trạng thái cập nhật của meta_weights
        updated_meta_weights = ParamDict(meta_weights.state_dict())

        return updated_meta_weights

    def foul_update(self, retain_grad_vec, forget_grad_vec, num_clients):
        """
        retain_grad_vec <torch tensor>: [num_retain_clients, dim] // CUDA:0
        forget_grad_vec <torch tensor>: [num_forget_clients, dim] // CUDA:0
        === === === === === === === ===
        The optimization is applied on CPU to save the GPU utilization.            print(f"current learn: {self.current_learn}")

        """

        r_grads = retain_grad_vec
        f_grads = forget_grad_vec

        """ Get dim for averaging """
        r_dim = r_grads.size()[0]
        f_dim = f_grads.size()[0]

        """ Retain mean grads """
        GGr = r_grads.mm(r_grads.t()).cpu()
        scale_r = (torch.diag(GGr)+1e-4).sqrt().mean()
        GGr = GGr / scale_r.pow(2)
        Ggr = GGr.mean(1, keepdims=True)
        ggr = Ggr.mean(0, keepdims=True)

        """ Forget mean grads """
        GGf = f_grads.mm(f_grads.t()).cpu()
        scale_f = (torch.diag(GGf)+1e-4).sqrt().mean()
        GGf = GGf / scale_f.pow(2)
        Ggf = GGf.mean(1, keepdims=True)
        ggf = Ggf.mean(0, keepdims=True)

        """ Get mean all """
        gg = (ggr * r_dim + ggf * f_dim)/(r_dim + f_dim)

        """ Define optimization variables w """
        w = torch.zeros(num_clients, 1, requires_grad=True)
        if num_clients == 50:
            w_opt = torch.optim.SGD([w], lr=self.foul_lr, momentum=0.5)
        else:
            w_opt = torch.optim.SGD([w], lr=self.foul_lr, momentum=0.5)

        c = (gg+1e-4).sqrt() * self.foul_c

        w_best = None
        obj_best = np.inf
        for i in range(21):
            w_opt.zero_grad()
            ww = torch.softmax(w, 0)
            """ Minimization objective function """

            obj = (
                   (ww[0:r_dim].t().mm(Ggr)
                    - self.beta * ww[r_dim:r_dim+f_dim].t().mm(Ggf))
                   + c * (torch.abs(
                            ww[0:r_dim].t().mm(GGr).mm(ww[0:r_dim])
                          - self.beta * ww[r_dim:r_dim+f_dim].t().mm(GGf).mm(ww[r_dim:r_dim+f_dim])
                          )+ 1e-4).sqrt()
                  )

            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < 20:
                obj.backward(retain_graph=True)
                w_opt.step()

        ww = torch.softmax(w_best, 0)
        gw_norm = (torch.abs(ww[0:r_dim].t().mm(GGr).mm(ww[0:r_dim])
                          - self.beta * ww[r_dim:r_dim+f_dim].t().mm(GGf).mm(ww[r_dim:r_dim+f_dim])) + 1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm+1e-4)

        g = (torch.cat((r_grads.cpu(),f_grads.cpu()), dim=0).mean(0)).view(-1, 1)
        g += lmbda * ((r_grads.cpu() * ww[0:r_dim]).sum(0)
                      - self.beta * (f_grads.cpu() * ww[r_dim:r_dim+f_dim]).sum(0)
                      ).view(-1, 1)
        g /= (1 + self.foul_c ** 2)

        return g.squeeze().to(self.device)

#     def aggregate_foul(self):
#         grad_ez = sum(p.numel() for p in self.global_model.parameters())
#         grads = torch.Tensor(grad_ez, self.num_clients)
#
#         for index, model in enumerate(self.grads):
#             grad2vec2(model, grads, index)
#
#         # Split to grads retain & grads forget
#         g = self.cagrad(grads, self.num_clients)
#
#         model_origin = copy.deepcopy(self.global_model)
#         self.overwrite_grad2(self.global_model, g)
#         for param in self.global_model.parameters():
#             param.data += param.grad
#
#         angle = [self.cos_sim(model_origin, self.global_model, models) for models in self.grads]
#         self.angle_value = statistics.mean(angle)
#
#         angle_value = []
#         for i in self.grads:
#             for j in self.grads:
#                 angle_value = [self.cosine_similarity(i, j)]
#
#         self.grads_angle_value = statistics.mean(angle_value)
#
#     def cagrad(self, grad_vec, num_clients):
#
#         grads = grad_vec.to(self.device)
#
#         GG = grads.t().mm(grads)
#         # to(device)
#         scale = (torch.diag(GG) + 1e-4).sqrt().mean()
#         GG = GG / scale.pow(2)
#         Gg = GG.mean(1, keepdims=True)
#         gg = Gg.mean(0, keepdims=True)
#
#         w = torch.zeros(num_clients, 1, requires_grad=True, device=self.device)
#         #         w = torch.zeros(num_clients, 1, requires_grad=True).to(self.device)
#
#         if num_clients == 50:
#             w_opt = torch.optim.SGD([w], lr=self.foul_lr * 2, momentum=self.momentum)
#         else:
#             w_opt = torch.optim.SGD([w], lr=self.foul_lr, momentum=self.momentum)
#
#         scheduler = StepLR(w_opt, step_size=self.step_size, gamma=self.gamma)
#
#         c = (gg + 1e-4).sqrt() * self.foul_c
#         w_best = None
#         obj_best = np.inf
#         for i in range(self.foul_rounds + 1):
#             w_opt.zero_grad()
#             ww = torch.softmax(w, dim=0)
#             obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
#             if obj.item() < obj_best:
#                 obj_best = obj.item()
#                 w_best = w.clone()
#             if i < self.foul_rounds:
#                 obj.backward()
#                 w_opt.step()
#                 scheduler.step()
#
#                 # Check this scheduler. step()
#
#         ww = torch.softmax(w_best, dim=0)
#         gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
#
#         lmbda = c.view(-1) / (gw_norm + 1e-4)
#         g = ((1 / num_clients + ww * lmbda).view(
#             -1, 1).to(grads.device) * grads.t()).sum(0) / (1 + self.foul_c ** 2)
#         return g
#
#         # def overwrite_grad(self, m, newgrad, grad_dims):
#         #     newgrad = newgrad * self.num_clients  # to match the sum loss
#         #     cnt = 0
#         #     for mm in m.shared_modules():
#         #         for param in mm.parameters():
#         #             beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
#         #             en = sum(grad_dims[:cnt + 1])
#         #             this_grad = newgrad[beg: en].contiguous().view(param.data.size())
#         #             param.grad = this_grad.data.clone()
#         #             cnt += 1
#
#     def overwrite_grad2(self, m, newgrad):
#         newgrad = newgrad * self.num_clients
#         for param in m.parameters():
#             # Get the number of elements in the current parameter
#             num_elements = param.numel()
#
#             # Extract a slice of new_params with the same number of elements
#             param_slice = newgrad[:num_elements]
#
#             # Reshape the slice to match the shape of the current parameter
#             param.grad = param_slice.view(param.data.size())
#
#             # Move to the next slice in new_params
#             newgrad = newgrad[num_elements:]
#
#     # def grad2vec(m, grads, grad_dims, task):
#     #     grads[:, task].fill_(0.0)
#     #     cnt = 0
#     #     for mm in m.shared_modules():
#     #         for p in mm.parameters():
#     #             grad_cur = p.data.detach().clone()
#     #             beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
#     #             en = sum(grad_dims[:cnt + 1])
#     #             grads[beg:en, task].copy_(grad_cur.data.view(-1))
#     #             cnt += 1
#
# def grad2vec2(m, grads, task):
#     grads[:, task].fill_(0.0)
#     all_params = torch.cat([param.detach().view(-1) for param in m.parameters()])
#     # print(all_params.size())
#     grads[:, task].copy_(all_params)
