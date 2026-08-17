import unittest

import numpy as np

from eeg_emotion.features.personal_calibration import PersonalBaselineCalibrator


class PersonalCalibrationTest(unittest.TestCase):
    def test_requires_quality_and_removes_median_offset(self):
        values = np.ones((16, 10, 4), dtype=np.float32) * np.array(
            [1.0, 2.0, 4.0, -4.0], dtype=np.float32
        )
        calibrator = PersonalBaselineCalibrator().fit(
            {"filtered": values, "bandpower": values.copy()},
            ["trusted"] * 16,
        )
        transformed = calibrator.transform(
            {"filtered": values, "bandpower": values.copy()}
        )
        np.testing.assert_allclose(
            np.median(transformed["filtered"], axis=(0, 1)),
            [0.0, 0.0, 1.0, -1.0],
        )

    def test_rejects_low_quality_baseline(self):
        values = np.zeros((16, 10, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Trusted baseline fraction"):
            PersonalBaselineCalibrator().fit(
                {"filtered": values}, ["low_ood"] * 16
            )


if __name__ == "__main__":
    unittest.main()
