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

# !/usr/bin/env python
import copy
import json
import torch
import argparse
import os
import time
import random
import warnings
import numpy as np
import torchvision
import logging

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverfoul import FOUL
from flcore.servers.serverufoul import UFOUL
from flcore.servers.serverretrain import Retrain
from flcore.servers.serverconda import CONDA
from flcore.servers.servermode import MODE


from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.uresnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *
from flcore.trainmodel.hierarchical_cnn import HierarchicalCNN
from flcore.trainmodel.hierarchical_resnet import HierarchicalResNet18
from utils.pet_protocol import apply_encoder_policy

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        args.run_seed = args.seed + i
        random.seed(args.run_seed)
        np.random.seed(args.run_seed)
        torch.manual_seed(args.run_seed)
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Pet datasets use aligned family/breed labels and a two-head model.
        is_hierarchical = args.dataset in ("OxfordPet", "StanfordDogs")
        hierarchy_config = None
        if is_hierarchical:
            if args.algorithm != "FOUL":
                raise ValueError("Pet protocol currently requires -algo FOUL, including its reference mode")
            if args.pet_checkpoint and args.learn_mode != "unlearn":
                raise ValueError("--pet-checkpoint is only used with -lm unlearn")
            if not 0 <= args.pet_validation_fraction < 1:
                raise ValueError("--pet-validation-fraction must be in [0, 1)")
            if args.pet_backbone == "cnn" and args.pet_encoder_unlearning not in ("preserve", "full"):
                raise ValueError("CNN unlearning updates the full network")
            if args.pet_unfreeze_round >= 0 and args.pet_encoder_training != "frozen":
                raise ValueError("Use either --pet-unfreeze-round or --pet-encoder-training")
            if min(args.pet_learning_rate, args.pet_backbone_lr, args.breed_loss_weight) <= 0:
                raise ValueError("Pet learning rates and breed loss weight must be positive")
            if args.pet_pool_grid < 1:
                raise ValueError("--pet-pool-grid must be positive")
            if not 0 <= args.min_target_accuracy <= 1:
                raise ValueError("--min-target-accuracy must be between 0 and 1")
            if not 0 <= args.pruning_rate < 1:
                raise ValueError("--pruning-rate must be between 0 (no pruning) and 1")
            if args.pet_pruning_only and args.pruning_rate == 0:
                raise ValueError("--pet-pruning-only requires --pruning-rate greater than 0")
            if args.pet_retrain_reference and args.learn_mode != "learn":
                raise ValueError("--pet-retrain-reference requires -lm learn")
            if args.pet_pruning_only and args.learn_mode == "learn":
                raise ValueError("--pet-pruning-only requires -lm unlearn or joint")
            if args.pet_unfreeze_round < -1 or args.pet_unfreeze_round >= args.learn_mode_rounds:
                raise ValueError("--pet-unfreeze-round must be -1 or a learning round")
            if args.pet_backbone == "cnn" and args.pet_unfreeze_round >= 0:
                raise ValueError("--pet-unfreeze-round requires --pet-backbone resnet18")
            if args.learn_mode == "joint" and not 0 < args.learn_mode_rounds < args.global_rounds:
                raise ValueError("Joint pet runs need learning rounds between 1 and global_rounds - 1")
            if args.learn_mode == "learn" and args.learn_mode_rounds < 1:
                raise ValueError("Pet learning needs at least one learning round")
            if args.learn_mode == "unlearn" and args.global_rounds <= args.learn_mode_rounds:
                raise ValueError("Pet unlearning needs at least one unlearning round")
            config_path = os.path.join(args.dataset_root, args.dataset, "config.json")
            if not os.path.isfile(config_path):
                raise FileNotFoundError(
                    f"Dataset config not found: {config_path}. Generate the federated pet dataset first."
                )
            with open(config_path, "r", encoding="utf-8") as config_file:
                hierarchy_config = json.load(config_file)
            if args.num_clients != int(hierarchy_config["num_clients"]):
                raise ValueError(
                    f"-nc {args.num_clients} does not match the generated dataset's "
                    f"{hierarchy_config['num_clients']} clients"
                )
            configured_image_size = int(hierarchy_config["image_size"])
            if args.image_size is not None and args.image_size != configured_image_size:
                raise ValueError(
                    f"--image-size {args.image_size} does not match generated data size "
                    f"{configured_image_size}; regenerate the data to change it"
                )
            args.num_classes = len(hierarchy_config["family_to_idx"])
            args.num_breeds = len(hierarchy_config["breed_to_idx"])
            if args.pet_backbone == "resnet18":
                args.model = HierarchicalResNet18(
                    num_families=args.num_classes,
                    num_breeds=args.num_breeds,
                    image_size=configured_image_size,
                    pretrained=args.pet_initialization == "imagenet" and args.learn_mode != "unlearn",
                ).to(args.device)
                if args.learn_mode != "unlearn":
                    if args.pet_initialization == "random" and args.pet_encoder_training != "full":
                        raise ValueError("Randomly initialized ResNet requires --pet-encoder-training full")
                    apply_encoder_policy(args.model, args.pet_encoder_training)
            else:
                args.model = HierarchicalCNN(
                    num_families=args.num_classes,
                    num_breeds=args.num_breeds,
                    image_size=configured_image_size,
                    pool_grid=args.pet_pool_grid,
                ).to(args.device)
            print(f"Pet training: {args.pet_backbone}, AdamW, head lr={args.pet_learning_rate}, "
                  f"breed weight={args.breed_loss_weight}")
        # Generate args.model
        elif model_str == "mlr":  # convex
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "CNN":  # non-convex
            if "MNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "PACS" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "DNN":  # non-convex
            if "MNIST" in args.dataset:
                args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

        elif model_str == "ResNet18":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)

        elif model_str == "ResNet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)

        elif model_str == "ResNet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "AlexNet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "GoogleNet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                      num_classes=args.num_classes).to(args.device)

            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "MobileNet":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)


        elif model_str == "LSTM":
            args.model = LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size,
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "BiLSTM":
            args.model = BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim,
                                                   output_size=args.num_classes, num_layers=1,
                                                   embedding_dropout=0, lstm_dropout=0, attention_dropout=0,
                                                   embedding_length=args.feature_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size,
                                  num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size,
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2,
                                          num_classes=args.num_classes, max_len=args.max_len).to(args.device)

        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "HARCNN":
            if args.dataset == 'HAR':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                    pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'PAMAP2':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                    pool_kernel_size=(1, 2)).to(args.device)



        ## uresnet foul code starts here
        elif model_str == "uresnet":
            if "MNIST" in args.dataset:
                args.model = UResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=args.num_classes).to(args.device)
            else :
                args.model = UResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=args.num_classes).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if is_hierarchical and args.algorithm != "FOUL":
            raise ValueError("OxfordPet and StanfordDogs currently require -algo FOUL")
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)
        elif args.algorithm == 'FOUL':
            if is_hierarchical:
                requested = args.forget_breed
                if requested is None:
                    raise ValueError("Set --forget-breed to an Oxford/Stanford breed")
                breed_map = hierarchy_config["breed_to_idx"]
                valid_names = {name.casefold(): int(index) for name, index in breed_map.items()}
                request_text = str(requested).strip()
                if request_text.casefold() in valid_names:
                    args.forget_breed = str(requested).strip()
                else:
                    try:
                        breed_id = int(request_text)
                    except ValueError as exc:
                        raise ValueError(
                            f"Unknown breed {requested!r}; choose a breed name or numeric ID"
                        ) from exc
                    if breed_id not in set(valid_names.values()):
                        raise ValueError(f"Unknown breed ID {breed_id}")
                    args.forget_breed = request_text
                server = FOUL(args, i)
            else:
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = FOUL(args, i)
        elif args.algorithm == 'UFOUL':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = UFOUL(args, i)
        elif args.algorithm == 'CONDA':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = CONDA(args, i)
        elif args.algorithm == 'Retrain':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = Retrain(args, i)
        elif args.algorithm == 'MODE':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MODE(args, i)
        elif args.algorithm == 'FUSED':
            from flcore.servers.serverFUSED import FUSED
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FUSED(args, i)
        else:
            raise NotImplementedErrorFOUL

        """ Start of the learn/unlearn process"""
        if args.learn_mode == 'learn':
            print("Learn")
            server.train()
        elif args.learn_mode == 'unlearn':
            print("Unlearn")
            server.unlearn()
        elif args.learn_mode == 'joint':
            print("Joint Learn & Unlearn")
            server.train()
            server.unlearn()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times,
                 results_dir=args.results_dir)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="PACS")
    parser.add_argument('-nb', "--num_classes", type=int, default=7)
    parser.add_argument('-m', "--model", type=str, default="uresnet")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=500)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('--results-dir', type=str, default='../results',
                        help='Directory for experiment JSON, H5 and pet checkpoints')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=32000,
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0,
                        help="lambda/sqrt(GLOABL-ITetain_grad_vecRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    # MOON / FedCAC / FedLC
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE    self.selected_clients = self.selected_clients()

    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-grained than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedDBE
    # parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)
    # FedCAGrad
    parser.add_argument('-car', "--cagrad_rounds", type=int, default=100)
    parser.add_argument('-calr', "--cagrad_learning_rate", type=float, default=25)
    parser.add_argument('-mmt', "--momentum", type=float, default=0.5)
    parser.add_argument('-ss', "--step_size", type=int, default=30)
    parser.add_argument('-gam', "--gamma", type=float, default=0.5)
    parser.add_argument('-c', "--c_parameter", type=float, default=0.5)
    parser.add_argument('-foulmlr', "--meta_lr", type=float, default=0.5)

    ### unlearning arguments to be mentioned here
    parser.add_argument('-lm', "--learn_mode", type=str, default='learn',
                    choices=['learn', 'unlearn', 'joint'],
                    help="Learn or Unlearn mode (default: learn)")
    parser.add_argument('-lmr', "--learn_mode_rounds", type=int, default=50,
                    help="Number of rounds for learn mode before unlearning")
    # parser.add_argument('-learn_percentage', "--learn_percentage", type=int, default=50,help="Number of clients we want to do learn")
    parser.add_argument('-f_index', "--f_index", type=int, nargs='+', default=3, help="List of client indices to forget")
    parser.add_argument('-log', "--log", action='store_true')

    ### FOUL
    parser.add_argument('-beta_foul', "--beta_foul", type=float, default=2.0)
    parser.add_argument('-forget_balance', "--forget_balance", type=bool, default=False, help="Balance f gradients")
    parser.add_argument('-retain_balance', "--retain_balance", type=bool, default=False, help="Balance r gradients")
    parser.add_argument('-re_init', "--re_init", type=bool, default=False)

    ### CONDA
    parser.add_argument('-dampening_constant', "--dampening_constant", type=float, default=10.0, help='lambda 10 for MNIST, 1 for CIFAR-10 and CIFAR100')
    parser.add_argument('-dampening_upper_bound', "--dampening_upper_bound", type=float, default=1.0,help='U 10 for MNIST, 1 for CIFAR-10 and CIFAR100' )
    parser.add_argument('-cutoff_alpha', "--cutoff_alpha", type=float, default=1.0)

    ### FUSED
    # Add these arguments after existing argument definitions
    parser.add_argument('-ncl', "--num_critical_layers", type=int, default=4,
                help="Number of critical layers to identify for unlearning")
    parser.add_argument('-sr', "--sparsity_ratio", type=float, default=0.9,
                        help="Sparsity ratio for adapters (0.9 = 90%% sparse)")
    parser.add_argument('-uc', "--unlearning_clients", type=int, nargs='+', default=[0],
                        help="List of client IDs to unlearn (e.g., -uc 0 1 2)")
    parser.add_argument('-ut', "--unlearning_type", type=str, default='client',
                        choices=['client', 'domain', 'class', 'sample'],
                        help="Type of unlearning: client-level, domain-level, class-level, or sample-level")
    parser.add_argument('-ud', "--unlearning_domain", type=int, default=0,
                        help="Domain ID to unlearn (0-3 for PACS: 0=art_painting, 1=cartoon, 2=photo, 3=sketch)")
    parser.add_argument('-fc', "--forget_class", type=int, default=None,
                        help="Class label to forget (for class-level unlearning)")
    parser.add_argument('--dataset-root', type=str, default='../dataset',
                        help='Root containing OxfordPet/StanfordDogs generated client data')
    parser.add_argument('--image-size', type=int, default=None,
                        help='Validate image size against the generated pet dataset config')
    parser.add_argument('-fb', '--forget-breed', dest='forget_breed', type=str, default=None,
                        help='Annotated pet breed name or breed ID to forget')
    parser.add_argument('--pruning-rate', type=float, default=0.5,
                        help='Fraction of target-discriminative channels removed before FOUL; 0 disables pruning')
    parser.add_argument('--pet-pruning-only', action='store_true',
                        help='Evaluate and save the pruned pet model without unlearning updates')
    parser.add_argument('--pet-retrain-reference', action='store_true',
                        help='Train a reference with the target excluded from breed loss but retained in family loss')
    parser.add_argument('--pet-checkpoint', type=str, default=None,
                        help='Explicit trained checkpoint for -lm unlearn; validate architecture and labels')
    parser.add_argument('--pet-pruning-strategy', choices=('magnitude', 'discriminative', 'random'), default='magnitude',
                        help='Magnitude compresses without using target labels; discriminative is the legacy targeted ablation')
    parser.add_argument('--pet-forget-scope', choices=('examples', 'attribute'), default='examples',
                        help='Examples removes target images from all retained tasks; attribute preserves their family supervision')
    parser.add_argument('--pet-skip-final-test', action='store_true',
                        help='Do not evaluate the official test during validation-only tuning runs')
    parser.add_argument('--pet-unlearning-method', choices=('foul', 'retain'), default='foul',
                        help='FOUL aggregation or sample-weighted retain-only adaptation')
    parser.add_argument('--pet-encoder-training', choices=('frozen', 'last-block', 'full'), default='frozen')
    parser.add_argument('--pet-encoder-unlearning', choices=('preserve', 'frozen', 'last-block', 'full'), default='preserve',
                        help='Preserve the checkpoint policy by default; full updates the entire encoder')
    parser.add_argument('--pet-initialization', choices=('imagenet', 'random'), default='imagenet',
                        help='ResNet initialization; the CNN always starts randomly')
    parser.add_argument('--pet-validation-fraction', type=float, default=0.0,
                        help='Stratified fraction of each client training set reserved for validation; 0 is legacy test monitoring')
    parser.add_argument('--pet-split-seed', type=int, default=1,
                        help='Fixed train/validation split seed, independent of optimization seed')
    parser.add_argument('--pet-augment', action='store_true',
                        help='Flip and brightness augmentation of raw training images')
    parser.add_argument('--pet-learning-rate', type=float, default=0.001,
                        help='AdamW learning rate for OxfordPet and StanfordDogs')
    parser.add_argument('--pet-backbone', choices=('resnet18', 'cnn'), default='resnet18',
                        help='Pet image encoder; resnet18 uses ImageNet pretrained weights')
    parser.add_argument('--pet-backbone-lr', type=float, default=1e-5,
                        help='Learning rate for trainable ResNet18 encoder parameters')
    parser.add_argument('--pet-unfreeze-round', type=int, default=-1,
                        help='Learning round to fine-tune ResNet18 layer4; -1 keeps it frozen')
    parser.add_argument('--breed-loss-weight', type=float, default=2.0,
                        help='Weight of breed classification loss during pet training')
    parser.add_argument('--pet-pool-grid', type=int, default=4,
                        help='Spatial grid retained before the pet classification heads')
    parser.add_argument('--min-target-accuracy', type=float, default=0.3,
                        help='Minimum pre-unlearning target accuracy required to start pruning')
    parser.add_argument('--clip-compare', action='store_true',
                        help='Compare an optional CLIP-selected visual cluster with breed labels')
    parser.add_argument('--clip-prompt', type=str, default=None,
                        help='Text prompt used for the optional CLIP comparison')
    parser.add_argument('--clip-clusters', type=int, default=5,
                        help='Number of visual clusters for the optional CLIP comparison')
    parser.add_argument('--clip-max-samples', type=int, default=1000,
                        help='Maximum family-matched images encoded by optional CLIP comparison')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for training, client selection and CLIP sampling')

    
    parser.add_argument('-wb', "--use_wandb", action='store_true',
                    help="Enable Weights & Biases logging")

                    




    args = parser.parse_args()


    # ============ COMPATIBILITY LAYER ============
    # Map new argument names to old FOUL/serverbase expected names
    if not hasattr(args, 'learn') or args.learn is None:
        args.learn = args.learn_mode
    
    # Ensure learn_round exists (already defined, but double-check)
    if not hasattr(args, 'learn_round'):
        args.learn_round = args.learn_mode_rounds

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    # args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=', getattr(args, arg))
    print("=" * 50)

    run(args)
