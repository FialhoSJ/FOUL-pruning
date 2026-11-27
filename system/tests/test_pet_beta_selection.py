"""Ensure beta selection uses validation metrics only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import select_pet_beta


class BetaSelectionTests(unittest.TestCase):
    def test_selects_closest_to_retraining_reference_without_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = {
                'experiment': {'forget_scope': 'examples', 'seed': 1,
                               'split_seed': 1, 'validation_fraction': .2},
                'metrics': [self.metric('after_learning', 0, .01, .8)],
            }
            self.write(root / 'OxfordPet_FOUL_resnet_examples_reference_s1_0_hierarchy.json', reference)
            for beta, target_confidence in [(1, .2), (2, .03), (4, .15)]:
                goal = f'resnet_examples_magnitude_foul_beta{beta}_s1'
                candidate = {
                    'experiment': {'forget_scope': 'examples', 'pruning_strategy': 'magnitude',
                                   'forget_weight': beta, 'pruning_rate': .25,
                                   'unlearning_method': 'foul', 'seed': 1, 'split_seed': 1,
                                   'validation_fraction': .2, 'unlearn_rounds': 30,
                                   'encoder_policy': 'full', 'source_checkpoint_sha256': 'same'},
                    'metrics': [self.metric('after_unlearning', 0, target_confidence, .8)],
                }
                self.write(root / f'OxfordPet_FOUL_{goal}_0_hierarchy.json', candidate)
                checkpoint = root / 'models' / 'OxfordPet' / f'FOUL_{goal}_0_pruning_unlearning.pt'
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.touch()
            with patch.object(sys, 'argv', ['select_pet_beta.py', '--results-dir', str(root)]):
                select_pet_beta.main()
            selected = json.loads((root / 'comparisons' / 'beta_selection_s1.json').read_text())
            self.assertEqual(selected['selected']['beta'], 2)

            candidate_path = root / 'OxfordPet_FOUL_resnet_examples_magnitude_foul_beta2_s1_0_hierarchy.json'
            candidate = json.loads(candidate_path.read_text())
            candidate['metrics'].append(self.metric('after_unlearning_test', .1, .1, .8, split='test'))
            self.write(candidate_path, candidate)
            with patch.object(sys, 'argv', ['select_pet_beta.py', '--results-dir', str(root)]):
                with self.assertRaisesRegex(ValueError, 'Test metrics found'):
                    select_pet_beta.main()

    @staticmethod
    def metric(stage, target_accuracy, target_confidence, retained_accuracy, split='validation'):
        return {'stage': stage, 'split': split, 'target_breed_accuracy': target_accuracy,
                'target_mean_confidence': target_confidence,
                'non_target_breed_accuracy': retained_accuracy, 'family_accuracy': .95,
                'groups': {'target': {'family_accuracy': .9}}}

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')


if __name__ == '__main__':
    unittest.main()
