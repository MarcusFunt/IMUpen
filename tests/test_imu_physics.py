import math
from pathlib import Path
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import imu_physics as physics
import measure
import simulate_imu


def state(index, *, orientation=(1, 0, 0, 0), acceleration=(0, 0, 0),
          omega=(0, 0, 0), alpha=(0, 0, 0), position=(0, 0, 0)):
    return {"seq": index, "time_s": index / 208.0,
            "orientation_body_to_world": orientation,
            "acceleration_world": acceleration,
            "angular_velocity_body_rad_s": omega,
            "angular_acceleration_body_rad_s2": alpha,
            "position_world_m": position, "contact": index % 2 == 0}


class ImuPhysicsTests(unittest.TestCase):
    def test_static_six_face_orientations_recover_gravity_axes(self):
        faces = [
            ((1, 0, 0, 0), (0, 0, 1)),
            (physics.quat_from_axis_angle((1, 0, 0), math.pi), (0, 0, -1)),
            (physics.quat_from_axis_angle((1, 0, 0), math.pi / 2), (0, 1, 0)),
            (physics.quat_from_axis_angle((1, 0, 0), -math.pi / 2), (0, -1, 0)),
            (physics.quat_from_axis_angle((0, 1, 0), math.pi / 2), (-1, 0, 0)),
            (physics.quat_from_axis_angle((0, 1, 0), -math.pi / 2), (1, 0, 0)),
        ]
        for orientation, expected in faces:
            sample = physics.rigid_body_sample(orientation_body_to_world=orientation)
            for got, want in zip(sample["specific_force_g"], expected):
                self.assertAlmostEqual(got, want, delta=1e-9)
            self.assertEqual(sample["gyro_rad_s"], (0.0, 0.0, 0.0))

    def test_constant_angular_velocity_preserves_sign_and_scale(self):
        omega = math.pi / 2
        dt = 1 / 208
        states = [state(i, orientation=physics.quat_from_axis_angle((0, 0, 1), omega * i * dt),
                        omega=(0, 0, omega)) for i in range(832)]
        result = physics.simulate_sensor(states)
        gyro = [row["raw"]["gz"] * physics.GYRO_DPS_PER_LSB for row in result["samples"]]
        self.assertTrue(all(value > 89.9 for value in gyro))
        integrated = sum(gyro[1:]) * dt
        self.assertAlmostEqual(integrated, omega * 831 * dt * 180 / math.pi, delta=0.03)

    def test_specific_force_includes_translation_and_sensor_lever_arm(self):
        translated = physics.rigid_body_sample(acceleration_world=(2, 0, 0))
        self.assertAlmostEqual(translated["specific_force_m_s2"][0], 2.0)
        self.assertAlmostEqual(translated["specific_force_m_s2"][2], physics.G0)
        rotated = physics.rigid_body_sample(angular_velocity_body_rad_s=(0, 0, 2),
                                            sensor_offset_body_m=(1, 0, 0))
        self.assertAlmostEqual(rotated["specific_force_m_s2"][0], -4.0)
        self.assertAlmostEqual(rotated["specific_force_m_s2"][2], physics.G0)

    def test_mount_transform_bias_noise_clipping_and_sensor_delay_are_traceable(self):
        mount = physics.quat_from_axis_angle((1, 0, 0), math.pi / 2)
        mounted = physics.rigid_body_sample(sensor_mount_orientation_body=mount)
        self.assertAlmostEqual(mounted["specific_force_g"][1], 1.0)
        states = [state(i, omega=(0, 0, (1 if i >= 4 else 0))) for i in range(12)]
        result = physics.simulate_sensor(states, seed=77, gyro_bias_dps=(0, 0, 600),
                                         gyro_noise_std_dps=0.1, gyro_delay_samples=2)
        self.assertGreater(result["clipping_by_axis"]["gz"], 0)
        self.assertEqual(result["samples"][5]["gyro_dps_truth"], result["samples"][3]["gyro_dps_truth"])
        self.assertTrue(result["synthetic"])
        self.assertIn("deg/s", result["units"]["gyroscope"])

    def test_four_clock_truth_and_robust_fit_recover_injected_rate(self):
        truth = physics.generate_clock_truth(frames=832, nominal_hz=208,
                                             odr_ppm=250, mcu_clock_ppm=-100,
                                             irq_jitter_std_us=35,
                                             irq_outlier_probability=0.02,
                                             irq_outlier_us=30000,
                                             host_latency_mean_us=1500,
                                             host_latency_jitter_us=600, seed=19)
        points = [(anchor["sample_index"], anchor["irq_time_us"])
                  for anchor in truth["anchors"]]
        model = measure.fit_robust_line(points, nominal_hz=208)
        self.assertAlmostEqual(model["estimated_hz"], truth["apparent_mcu_hz"],
                               delta=truth["apparent_mcu_hz"] * 0.001)
        reconstructed = [model["ybar"] + model["slope_us_per_sample"] *
                         (row["seq"] - model["xbar"]) for row in truth["rows"]]
        errors = [abs(estimate - row["mcu_time_us"])
                  for estimate, row in zip(reconstructed, truth["rows"])]
        self.assertLess(measure.percentiles(errors)["p99"], 3000)
        self.assertGreater(truth["rows"][0]["host_receive_time_us"], truth["rows"][0]["mcu_time_us"])
        self.assertEqual(truth["clock_names"], ["sensor_time_us", "mcu_time_us",
                                                 "irq_time_us", "host_receive_time_us"])

    def test_zero_jitter_rate_fit_is_within_100_ppm_of_clock_model(self):
        truth = physics.generate_clock_truth(frames=832, odr_ppm=220, mcu_clock_ppm=80,
                                             seed=2)
        model = measure.fit_robust_line([(a["sample_index"], a["irq_time_us"])
                                         for a in truth["anchors"]])
        self.assertLess(abs(model["estimated_hz"] / truth["apparent_mcu_hz"] - 1) * 1e6, 1)
        self.assertLess(abs(model["nominal_drift_ppm"] -
                            (truth["apparent_mcu_hz"] / truth["nominal_hz"] - 1) * 1e6), 1)

    def test_machine_readable_simulation_records_seed_four_clocks_and_raw_truth(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = root / "first.json", root / "second.json"
            simulate_imu.generate(first, frames=32, seed=22, irq_outlier_probability=0)
            simulate_imu.generate(second, frames=32, seed=22, irq_outlier_probability=0)
            document = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(document["source_type"], "synthetic")
        self.assertEqual(document["configuration"]["seed"], 22)
        self.assertEqual(len(document["samples"]), 32)
        sample = document["samples"][0]
        self.assertEqual(set(sample["raw"]), {"gx", "gy", "gz", "ax", "ay", "az"})
        self.assertTrue(all(-32768 <= value <= 32767 for value in sample["raw"].values()))
        self.assertIn("sensor_time_us", document["clock_truth"]["clock_names"])
        self.assertIn("host_receive_time_us", document["clock_truth"]["clock_names"])
        self.assertGreater(document["free_integration_drift_metric_m"], 0)
        self.assertIn("truth orientation", document["free_integration_note"])


if __name__ == "__main__":
    unittest.main()
