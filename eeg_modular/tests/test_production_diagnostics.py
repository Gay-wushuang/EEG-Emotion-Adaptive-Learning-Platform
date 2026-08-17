import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg_emotion.models.production_baseline import load_production_package


class ProductionDiagnosticsTest(unittest.TestCase):
    def test_shapley_contributions_reconstruct_fusion_logits(self):
        model, _, _ = load_production_package(ROOT / "production_baseline_v1")
        inputs = {
            "filtered": torch.randn(3, 10, 4),
            "bandpower": torch.randn(3, 10, 4),
        }
        with torch.no_grad():
            output = model.diagnostic_outputs(inputs)
        reconstructed = (
            output["logits_none"]
            + output["logit_contribution_filtered"]
            + output["logit_contribution_bandpower"]
        )
        torch.testing.assert_close(reconstructed, output["logits_fusion"])
        torch.testing.assert_close(model(inputs), output["logits_fusion"])


if __name__ == "__main__":
    unittest.main()
