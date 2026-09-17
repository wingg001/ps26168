import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.filters.coordinates import LocalTangentPlane
from src.filters.ukf import ScaledUKF, state_mean
from src.filters.constraints import (
    hx_nhc,
    hx_speed,
    hx_gnss,
    chi_square_gate,
    match_cnn_speed,
)
from src.eval.run_phase3 import (
    load_config,
    calculate_metrics,
    resolve_repo_path,
    resolve_manifest_path,
    estimate_time_alignment,
    parse_phone_gnss_speed,
    estimate_initial_course,
    compute_initialization,
)
from src.phase1.time_alignment import interpolate_vehicle_to_phone_time


class TestPhase3(unittest.TestCase):
    def test_coordinates_enu(self):
        ltp = LocalTangentPlane(51.0, 0.0, 100.0)
        enu = ltp.to_enu(51.001, 0.001, 100.0)
        self.assertAlmostEqual(enu[0], 70.197, delta=0.1)
        self.assertAlmostEqual(enu[1], 111.25, delta=0.1)

    def test_ukf_circular_angles(self):
        sigmas = np.zeros((3, 15))
        sigmas[0, 6] = np.pi - 0.1
        sigmas[1, 6] = -np.pi + 0.1
        sigmas[2, 6] = np.pi
        Wm = np.full(3, 1 / 3)
        mean_x = state_mean(sigmas, Wm)
        self.assertAlmostEqual(abs(mean_x[6]), np.pi, delta=0.05)

    def test_ukf_cov_psd_repair(self):
        ukf = ScaledUKF(15, 1e-3, 2.0, 0.0)
        ukf.P = np.eye(15)
        ukf.P[0, 0] = -1e-6
        ukf._symmetrize_and_ensure_psd()
        self.assertGreaterEqual(ukf.regularization_count, 1)
        vals, _ = np.linalg.eigh(ukf.P)
        self.assertTrue(np.all(vals >= 0.0))

    def test_cnn_timestamp_matching(self):
        cnn_t = np.array([10.0, 10.5, 11.0])
        cnn_v = np.array([5.0, 6.0, 7.0])
        self.assertEqual(match_cnn_speed(10.5, cnn_t, cnn_v, tol=0.1), 6.0)
        self.assertEqual(match_cnn_speed(10.55, cnn_t, cnn_v, tol=0.1), 6.0)
        self.assertTrue(np.isnan(match_cnn_speed(10.7, cnn_t, cnn_v, tol=0.1)))

    def test_load_config(self):
        cfg = load_config()
        self.assertIn("ukf", cfg)
        self.assertIn("data", cfg)
        self.assertEqual(cfg["ukf"]["alpha"], 1.0)

    def test_repo_relative_path_resolution(self):
        resolved = resolve_repo_path("configs/phase3_navigation.yaml")
        self.assertTrue(resolved.is_absolute())
        self.assertTrue(resolved.exists())

    def test_phase2_manifest_path_resolution(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phase2_root = root / "phase2_speed_estimation"
            manifest_path = phase2_root / "reports" / "phase2" / "dataset_manifest.json"
            data_file = phase2_root / "data" / "Synchronised V abd S datasets" / "Vw12" / "S-Vw12.csv"
            data_file.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            data_file.write_text("x", encoding="utf-8")
            manifest_path.write_text("{}", encoding="utf-8")

            resolved = resolve_manifest_path(
                "data/Synchronised V abd S datasets/Vw12/S-Vw12.csv",
                manifest_path,
            )
            self.assertEqual(resolved, data_file.resolve())
            self.assertNotIn(str(Path("reports") / "phase2" / "data"), str(resolved))

    def test_time_alignment_dict_result(self):
        t_p = np.arange(0.0, 100.0, 0.1)
        t_v = t_p + 12.5
        speed = 5.0 + 3.0 * np.sin(0.1 * t_p) + 2.0 * np.cos(0.03 * t_p)
        offset_s, quality = estimate_time_alignment(t_p, speed, t_v, speed)
        self.assertAlmostEqual(offset_s, 12.5, delta=1.0)
        self.assertIn(quality, ("high", "moderate", "low"))

    def test_time_alignment_flat_phone_speed_fallback(self):
        t_p = np.arange(0.0, 100.0, 0.1)
        t_v = t_p + 7.0
        flat_speed = np.full_like(t_p, 10.0)
        offset_s, quality = estimate_time_alignment(t_p, flat_speed, t_v, flat_speed)
        self.assertEqual(quality, "fallback")
        self.assertAlmostEqual(offset_s, 7.0, delta=0.1)

    def test_interpolation_four_arguments(self):
        t_p = np.arange(0.0, 100.0, 0.1)
        t_v = t_p + 12.5
        values = 2.0 * t_v
        out = interpolate_vehicle_to_phone_time(t_p, t_v, values, 12.5)
        self.assertEqual(len(out), len(t_p))
        self.assertTrue(np.allclose(out, 2.0 * (t_p + 12.5)))

    def test_metrics(self):
        mae, rmse, max_err = calculate_metrics([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mae, 2.0)
        self.assertAlmostEqual(rmse, np.sqrt(14 / 3))
        self.assertAlmostEqual(max_err, 3.0)

    def test_measurement_models(self):
        x = np.zeros(15)
        x[0:2] = [4.0, 5.0]
        self.assertTrue(np.allclose(hx_gnss(x), [4.0, 5.0]))
        self.assertTrue(np.allclose(hx_nhc(x), [0.0, 0.0]))
        self.assertTrue(np.allclose(hx_speed(x), [0.0]))

    def test_gnss_gate(self):
        self.assertTrue(chi_square_gate(2.5, 2, 0.95))
        self.assertFalse(chi_square_gate(10.0, 2, 0.95))

    def test_gnss_noise_calibration_formula(self):
        from src.calibration.gnss_noise import calibrate_gnss_noise, MAD_TO_SIGMA

        rng = np.random.default_rng(0)
        n = 200
        phone_e = rng.normal(loc=5.0, scale=10.0, size=n)
        phone_n = rng.normal(loc=-4.0, scale=8.0, size=n)
        ref_e = np.zeros(n)
        ref_n = np.zeros(n)
        valid = np.ones(n, dtype=bool)
        cal = calibrate_gnss_noise(phone_e, phone_n, ref_e, ref_n, valid)
        self.assertAlmostEqual(cal.sigma_e_m, MAD_TO_SIGMA * np.median(np.abs(phone_e)), delta=0.3)
        self.assertAlmostEqual(cal.sigma_n_m, MAD_TO_SIGMA * np.median(np.abs(phone_n)), delta=0.3)
        expected_iso = np.sqrt(0.5 * (cal.sigma_e_m**2 + cal.sigma_n_m**2))
        self.assertAlmostEqual(cal.sigma_iso_m, expected_iso, delta=1e-9)
        self.assertGreater(cal.sigma_used_m, 0)

    def test_gnss_noise_calibration_floor(self):
        from src.calibration.gnss_noise import calibrate_gnss_noise

        n = 50
        rng = np.random.default_rng(1)
        phone_e = rng.normal(0.0, 0.5, n)
        phone_n = rng.normal(0.0, 0.5, n)
        cal = calibrate_gnss_noise(phone_e, phone_n, np.zeros(n), np.zeros(n), np.ones(n, dtype=bool), sigma_floor_m=3.0)
        self.assertEqual(cal.sigma_used_m, 3.0)
        self.assertTrue(np.allclose(cal.r_covariance, np.eye(2) * 9.0))

    def test_gnss_noise_calibration_window(self):
        from src.calibration.gnss_noise import calibrate_gnss_noise

        n = 200
        rng = np.random.default_rng(2)
        phone_e = rng.normal(0.0, 20.0, n)
        phone_n = rng.normal(0.0, 20.0, n)
        t_rel = np.arange(n, dtype=float)
        valid = np.ones(n, dtype=bool)
        cal = calibrate_gnss_noise(
            phone_e, phone_n, np.zeros(n), np.zeros(n), valid,
            window_duration_s=50.0, t_rel=t_rel,
        )
        self.assertEqual(cal.n_samples, 50)
        self.assertAlmostEqual(cal.res_e_median_m, np.median(phone_e[:50]), delta=0.5)

    def test_gnss_noise_calibration_distinct_fixes(self):
        from src.calibration.gnss_noise import calibrate_gnss_noise

        n = 300
        rng = np.random.default_rng(3)
        phone_e = rng.normal(0.0, 20.0, n)
        phone_n = rng.normal(0.0, 20.0, n)
        ref_e = np.zeros(n)
        ref_n = np.zeros(n)
        valid = np.ones(n, dtype=bool)
        # Only 10 distinct fixes: each fix repeated across a 30-row block.
        distinct = np.zeros(n, dtype=bool)
        distinct[0::30] = True
        cal = calibrate_gnss_noise(phone_e, phone_n, ref_e, ref_n, valid,
                                   distinct=distinct, window_duration_s=300.0)
        self.assertEqual(cal.n_distinct_fixes, 10)
        self.assertEqual(cal.n_repeated_rows_skipped, n - 10)
        self.assertEqual(cal.n_samples, 10)
        # Each distinct fix contributes exactly one sample -> MAD over the 10 first values.
        self.assertAlmostEqual(
            cal.res_e_mad_m,
            np.median(np.abs(phone_e[0::30])),
            delta=1e-9,
        )

    def test_load_config_gnss_noise(self):
        cfg = load_config()
        self.assertIn("gnss_noise", cfg["ukf"])
        self.assertEqual(cfg["ukf"]["gnss_noise"]["mode"], "smartphone_accuracy")
        self.assertIn(cfg["ukf"]["gnss_noise"]["mode"], ("smartphone_accuracy", "empirical", "per_sample"))
        self.assertGreater(cfg["ukf"]["gnss_noise"]["sigma_floor_m"], 0)
        self.assertGreater(cfg["ukf"]["gnss_noise"]["calibration_window_s"], 0)
        self.assertLessEqual(cfg["ukf"]["gnss_noise"]["calibration_window_s"],
                             float(cfg["blackout_start_s"]))


class TestSmartInit(unittest.TestCase):
    """Runtime initialization uses SMARTPHONE data only (no V-file/reference) and
    falls back to zero bias / zero velocity / documented yaw when legitimate
    smartphone signals are missing."""

    def _phone_fixes(self):
        # 5 distinct fixes travelling 6 m/s toward the north-east over ~40 s.
        t = np.arange(0, 40.0, 0.1)
        v = 6.0
        d = v * (t - t[0])
        e = 0.7 * d
        n = 0.7 * d
        return e, n

    @staticmethod
    def _distinct_mask(len_p):
        mask = np.zeros(len_p, dtype=bool)
        mask[0] = True
        mask[100] = True   # ~10 s
        mask[300] = True   # ~30 s
        return mask

    def test_parse_phone_gnss_speed_units(self):
        import pandas as pd
        df = pd.DataFrame({"GPS SPEED (Kmh)": [24.09, np.nan, 25.5, 0.0, 26.2]})
        out = parse_phone_gnss_speed(df, "GPS SPEED (Kmh)")
        # Stored values are in m/s (verified against reference + displacement);
        # NOT divided by 3.6.
        self.assertAlmostEqual(out[0], 24.09)
        self.assertAlmostEqual(out[4], 26.2)
        self.assertTrue(np.isnan(out[1]))
        self.assertTrue(np.isfinite(out[3]))

    def test_estimate_initial_course_sufficient_motion(self):
        e, n = self._phone_fixes()
        mask = self._distinct_mask(len(e))
        est = estimate_initial_course(e, n, mask, min_fixes=3, min_span_m=50.0)
        self.assertIsNotNone(est)
        # Motion is NE -> course ~45 deg clockwise from north; span ~34 m.
        self.assertAlmostEqual(est["course_deg"] % 360.0, 45.0, delta=0.5)
        self.assertGreater(est["span_m"], 50.0)
        self.assertEqual(est["n_fixes"], 3)
        # Yaw follows filter convention: yaw = 90 - course.
        self.assertAlmostEqual(np.degrees(est["yaw_rad"]), 90.0 - 45.0, delta=0.5)

    def test_estimate_initial_course_insufficient_fixes(self):
        e, n = self._phone_fixes()
        mask = np.zeros(len(e), dtype=bool)
        # Only 2 distinct fixes, need 3.
        est = estimate_initial_course(e, n, mask, min_fixes=3, min_span_m=50.0)
        self.assertIsNone(est)

    def test_estimate_initial_course_insufficient_motion(self):
        e, n = self._phone_fixes()
        mask = self._distinct_mask(len(e))
        # Require an impossible span -> insufficient motion -> None (documented fallback).
        est = estimate_initial_course(e, n, mask, min_fixes=3, min_span_m=5000.0)
        self.assertIsNone(est)

    def test_zero_bias_fallback_when_static_invalid(self):
        e, n = self._phone_fixes()
        mask = self._distinct_mask(len(e))
        speed = np.full(len(e), 6.0)
        R = np.eye(3)

        # Invalid static calibration -> zero bias regardless of measured values.
        init = compute_initialization(
            e, n, mask, speed, zupt_enabled=False,
            calib_accel_bias=np.array([9.9, 9.9, 0.0]),
            calib_gyro_bias=np.array([0.5, 0.5, 0.5]),
            R_veh=R,
        )
        self.assertTrue(np.allclose(init["accel_bias_veh"], 0.0))
        self.assertTrue(np.allclose(init["gyro_bias_veh"], 0.0))
        self.assertIn("zero", init["bias_source"].lower())

    def test_valid_static_calibration_keeps_measured_bias(self):
        e, n = self._phone_fixes()
        mask = self._distinct_mask(len(e))
        speed = np.full(len(e), 6.0)
        init = compute_initialization(
            e, n, mask, speed, zupt_enabled=True,
            calib_accel_bias=np.array([0.4, 0.2, 0.0]),
            calib_gyro_bias=np.array([-0.02, -0.01, -0.002]),
            R_veh=np.eye(3),
        )
        self.assertTrue(np.allclose(init["accel_bias_veh"], [0.4, 0.2, 0.0]))
        self.assertTrue(np.allclose(init["gyro_bias_veh"], [-0.02, -0.01, -0.002]))

    def test_gnss_velocity_initialization(self):
        e, n = self._phone_fixes()
        mask = self._distinct_mask(len(e))
        speed = np.full(len(e), 6.0)
        init = compute_initialization(
            e, n, mask, speed, zupt_enabled=False,
            calib_accel_bias=np.zeros(3), calib_gyro_bias=np.zeros(3), R_veh=np.eye(3),
        )
        # speed 6 m/s, course ~45 deg -> vel (E,N) ~ (4.24, 4.24).
        self.assertAlmostEqual(init["init_vel"][0], 6.0 * np.sin(np.radians(45)), delta=0.1)
        self.assertAlmostEqual(init["init_vel"][1], 6.0 * np.cos(np.radians(45)), delta=0.1)
        self.assertAlmostEqual(np.linalg.norm(init["init_vel"]), 6.0, delta=0.1)

    def test_gnss_course_heading_initialization(self):
        e, n = self._phone_fixes()
        mask = self._distinct_mask(len(e))
        speed = np.full(len(e), 6.0)
        init = compute_initialization(
            e, n, mask, speed, zupt_enabled=False,
            calib_accel_bias=np.zeros(3), calib_gyro_bias=np.zeros(3), R_veh=np.eye(3),
        )
        self.assertIn("phone GNSS course", init["heading_source"])
        self.assertAlmostEqual(np.degrees(init["init_yaw"]), 90.0 - 45.0, delta=0.5)

    def test_insufficient_motion_heading_fallback(self):
        e, n = self._phone_fixes()
        mask = self._distinct_mask(len(e))
        speed = np.full(len(e), 6.0)
        init = compute_initialization(
            e, n, mask, speed, zupt_enabled=False,
            calib_accel_bias=np.zeros(3), calib_gyro_bias=np.zeros(3), R_veh=np.eye(3),
            min_span_m=5000.0,
        )
        self.assertIn("fallback yaw=0", init["heading_source"])
        self.assertEqual(init["init_yaw"], 0.0)
        # No reliable course -> velocity stays zero (logged reason), no ground truth.
        self.assertTrue(np.allclose(init["init_vel"], 0.0))
        self.assertIn("no reliable course", init["velocity_source"])

    def test_no_valid_speed_velocity_fallback(self):
        e, n = self._phone_fixes()
        mask = self._distinct_mask(len(e))
        speed = np.zeros(len(e))
        init = compute_initialization(
            e, n, mask, speed, zupt_enabled=False,
            calib_accel_bias=np.zeros(3), calib_gyro_bias=np.zeros(3), R_veh=np.eye(3),
        )
        self.assertTrue(np.allclose(init["init_vel"], 0.0))
        self.assertIn("no valid phone GNSS speed", init["velocity_source"])

    def test_no_vfile_data_in_runtime_initialization(self):
        # compute_initialization accepts ONLY phone-derived inputs; a reference
        # heading/velocity cannot be passed in (no df_v / v_map argument exists).
        import inspect
        sig = inspect.signature(compute_initialization)
        params = list(sig.parameters)
        self.assertNotIn("df_v", params)
        self.assertNotIn("v_map", params)
        self.assertNotIn("ref", params)
        self.assertNotIn("ground", params)
        # And the implementation must not import/use a vehicle heading field.
        src = inspect.getsource(compute_initialization)
        self.assertNotIn("heading_deg", src)
        self.assertNotIn("v_map", src)
        self.assertNotIn("df_v", src)
        self.assertNotIn("get_initial_yaw", src)


class TestSmartphoneAccuracy(unittest.TestCase):
    """R3: smartphone-only GNSS uncertainty from the phone's GPS ACCURACY field.

    No vehicle/reference data enters the sigma computation. The uncertainty
    scale is the 75th percentile of valid distinct-fix GPS ACCURACY values in
    the calibration window, floored at sigma_floor_m (a policy minimum that
    prevents overconfidence).
    """

    @staticmethod
    def _cal(acc, valid, distinct=None, floor=3.0, duration=60.0, cap=None):
        from src.calibration.gnss_noise import calibrate_gnss_noise_from_accuracy

        if distinct is None:
            distinct = np.ones(len(acc), dtype=bool)
        t_rel = np.arange(len(acc), dtype=float)
        return calibrate_gnss_noise_from_accuracy(
            np.asarray(acc, dtype=float),
            np.asarray(valid, dtype=bool),
            sigma_floor_m=floor,
            window_start_s=0.0,
            window_duration_s=duration,
            t_rel=t_rel,
            distinct=distinct,
            accuracy_cap_m=cap,
        )

    def test_75th_percentile_calculation(self):
        acc = [2.0, 3.0, 3.0, 4.0, 5.0]
        cal = self._cal(acc, [True] * 5, floor=0.0)
        expected = np.percentile(acc, 75)
        self.assertAlmostEqual(cal.uncertainty_scale_m, expected, delta=1e-9)
        self.assertAlmostEqual(cal.runtime_scale_m, expected, delta=1e-9)
        self.assertTrue(np.allclose(cal.r_covariance, np.eye(2) * (expected**2)))

    def test_floor_binding(self):
        acc = [0.5, 1.0, 0.1, 2.0]
        cal = self._cal(acc, [True] * 4, floor=3.0)
        self.assertLess(cal.uncertainty_scale_m, 3.0)
        self.assertEqual(cal.runtime_scale_m, 3.0)

    def test_all_invalid_falls_back_to_floor(self):
        acc = [np.nan, np.nan, np.nan]
        cal = self._cal(acc, [True] * 3, floor=3.0)
        self.assertEqual(cal.n_samples, 0)
        self.assertEqual(cal.runtime_scale_m, 3.0)

    def test_invalid_values_filtered(self):
        acc = [2.0, np.nan, 0.0, -1.0, 4.0, 3.0]
        cal = self._cal(acc, [True] * 6, floor=0.0)
        self.assertEqual(cal.n_samples, 3)
        self.assertEqual(cal.n_invalid_values_skipped, 3)
        self.assertAlmostEqual(
            cal.uncertainty_scale_m, np.percentile([2.0, 4.0, 3.0], 75), delta=1e-9
        )

    def test_distinct_fix_filtering(self):
        acc = [2.0, 9.0, 3.0, 40.0, 4.0, 5.0]
        distinct = [True, False, True, False, True, False]
        cal = self._cal(acc, [True] * 6, distinct=distinct, floor=0.0)
        self.assertEqual(cal.n_samples, 3)
        self.assertEqual(cal.n_repeated_rows_skipped, 3)
        self.assertAlmostEqual(
            cal.uncertainty_scale_m, np.percentile([2.0, 3.0, 4.0], 75), delta=1e-9
        )

    def test_optional_quality_cap(self):
        acc = [2.0, 3.0, 50.0, 200.0]
        cal = self._cal(acc, [True] * 4, floor=0.0, cap=100.0)
        self.assertEqual(cal.n_samples, 3)
        self.assertEqual(cal.n_invalid_values_skipped, 1)
        self.assertEqual(cal.quality_cap_m, 100.0)

    def test_cap_disabled_when_none(self):
        acc = [2.0, 3.0, 50.0, 200.0]
        cal = self._cal(acc, [True] * 4, floor=0.0, cap=None)
        self.assertEqual(cal.n_samples, 4)
        self.assertIsNone(cal.quality_cap_m)

    def test_no_reference_data_in_signature(self):
        import inspect

        from src.calibration.gnss_noise import calibrate_gnss_noise_from_accuracy

        params = set(inspect.signature(calibrate_gnss_noise_from_accuracy).parameters)
        for forbidden in ("ref_e", "ref_n", "phone_e", "phone_n", "df_v", "v_map", "ref_enu"):
            self.assertNotIn(forbidden, params)
        src = inspect.getsource(calibrate_gnss_noise_from_accuracy)
        self.assertNotIn("ref_enu_p", src)
        self.assertNotIn("ref_enu", src)


if __name__ == "__main__":
    unittest.main()
