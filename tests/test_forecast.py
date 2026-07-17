from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from experiments import configs, model
from pipeline import forecast


class CalendarFeatureTests(unittest.TestCase):
    def test_german_holiday_starts_at_local_midnight_not_utc_midnight(self):
        # Karfreitag 2026 beginnt in Deutschland am 03.04. um 00:00 CEST,
        # also bereits am 02.04. um 22:00 UTC.
        index = pd.DatetimeIndex(
            ["2026-04-02T21:45:00Z", "2026-04-02T22:00:00Z"]
        )
        np.testing.assert_array_equal(
            model._calendar_feature(index, "holiday"),
            np.array([0.0, 1.0], dtype=np.float32),
        )

    def test_weekend_uses_local_day_across_spring_dst(self):
        index = pd.DatetimeIndex(
            ["2026-03-28T22:30:00Z", "2026-03-29T22:30:00Z"]
        )
        # Lokal: Samstag 23:30 CET, danach Montag 00:30 CEST.
        np.testing.assert_array_equal(
            model._calendar_feature(index, "weekend"),
            np.array([1.0, 0.0], dtype=np.float32),
        )

    def test_holidays_are_available_beyond_original_training_years(self):
        self.assertIn(pd.Timestamp("2027-03-26").date(), model._de_holidays(2027))


class DeliveryWindowTests(unittest.TestCase):
    def _assert_geometry(self, target_start: str, expected_cutoff: str) -> pd.DatetimeIndex:
        cutoff, future, delivery = forecast._delivery_window(target_start)
        self.assertEqual(cutoff, pd.Timestamp(expected_cutoff))
        self.assertEqual(len(future), 155)
        self.assertEqual(int(delivery.sum()), 96)
        delivery_index = future[delivery]
        target_utc = pd.Timestamp(target_start).tz_convert("UTC")
        self.assertEqual(delivery_index[0], target_utc)
        self.assertEqual(delivery_index[-1], target_utc + pd.Timedelta(minutes=15 * 95))
        deltas = delivery_index.to_series().diff().dropna()
        self.assertTrue((deltas == pd.Timedelta("15min")).all())
        return delivery_index

    def test_summer_target_uses_local_previous_day_gate(self):
        self._assert_geometry(
            "2026-07-18T00:00:00+02:00",
            "2026-07-17T07:00:00+00:00",
        )

    def test_winter_target_uses_local_previous_day_gate(self):
        self._assert_geometry(
            "2026-01-18T00:00:00+01:00",
            "2026-01-17T08:00:00+00:00",
        )

    def test_spring_dst_keeps_96_real_arena_steps(self):
        delivery = self._assert_geometry(
            "2026-03-29T00:00:00+01:00",
            "2026-03-28T08:00:00+00:00",
        )
        # 96 reale Schritte reichen wegen der fehlenden lokalen Stunde bis 00:45.
        self.assertEqual(
            delivery[-1].tz_convert(forecast.ARENA_TZ),
            pd.Timestamp("2026-03-30T00:45:00+02:00"),
        )

    def test_autumn_dst_keeps_96_real_arena_steps(self):
        delivery = self._assert_geometry(
            "2026-10-25T00:00:00+02:00",
            "2026-10-24T07:00:00+00:00",
        )
        # Die wiederholte lokale Stunde laesst den Horizont um 22:45 enden.
        self.assertEqual(
            delivery[-1].tz_convert(forecast.ARENA_TZ),
            pd.Timestamp("2026-10-25T22:45:00+01:00"),
        )

    def test_naive_or_non_midnight_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Zeitzone"):
            forecast._delivery_window("2026-07-18T00:00:00")
        with self.assertRaisesRegex(ValueError, "Tagesbeginn"):
            forecast._delivery_window("2026-07-18T01:00:00+02:00")


class QuantileConversionTests(unittest.TestCase):
    @staticmethod
    def _linear_curves(base: float, slope: float, horizon: int = 3) -> np.ndarray:
        levels = forecast.CHRONOS_QUANTILE_LEVELS[:, None]
        time = np.arange(horizon, dtype=float)[None, :]
        return base + slope * levels + time

    def test_arena_quantiles_are_interpolated_not_requantiled(self):
        q1 = self._linear_curves(100.0, 10.0)
        q2 = self._linear_curves(200.0, 20.0)
        result = forecast.delu_quantiles({"a": q1, "b": q2})
        expected = (
            300.0
            + 30.0 * forecast.ARENA_QUANTILE_LEVELS[:, None]
            + 2.0 * np.arange(3, dtype=float)[None, :]
        )
        np.testing.assert_allclose(result, expected)

    def test_point_is_native_median(self):
        q1 = self._linear_curves(100.0, 10.0)
        q2 = self._linear_curves(200.0, 20.0)
        expected = 300.0 + 30.0 * 0.5 + 2.0 * np.arange(3, dtype=float)
        np.testing.assert_allclose(forecast.delu_point({"a": q1, "b": q2}), expected)

    def test_ensemble_is_deterministic_inverse_cdf_with_shared_u(self):
        q1 = self._linear_curves(100.0, 10.0)
        q2 = self._linear_curves(200.0, 20.0)
        first = forecast.delu_ensemble({"a": q1, "b": q2}, n=100)
        second = forecast.delu_ensemble({"a": q1, "b": q2}, n=100)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (100, 3))

        # Dasselbe u wird in beiden Regionen und ueber alle Zeitpunkte genutzt.
        u = np.clip((np.arange(100) + 0.5) / 100, 0.01, 0.99)
        expected = (
            300.0
            + 30.0 * u[:, None]
            + 2.0 * np.arange(3, dtype=float)[None, :]
        )
        np.testing.assert_allclose(first, expected)

    def test_invalid_quantile_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Shape"):
            forecast.delu_quantiles({"a": np.ones((20, 96))})
        with self.assertRaisesRegex(ValueError, "positive Ganzzahl"):
            forecast.delu_ensemble({"a": np.ones((21, 96))}, n=0)

    def test_crossing_is_rearranged_without_discarding_predicted_values(self):
        curves = np.repeat(
            forecast.CHRONOS_QUANTILE_LEVELS[:, None], repeats=2, axis=1
        )
        curves[[0, 1]] = curves[[1, 0]]
        repaired = forecast._as_monotone_quantiles(curves)
        self.assertTrue((np.diff(repaired, axis=0) >= 0).all())
        np.testing.assert_array_equal(np.sort(curves, axis=0), repaired)


class LiveInputTests(unittest.TestCase):
    def test_each_live_forecast_clears_file_caches_before_reading(self):
        clear = mock.Mock()

        def area_frame(*_args, **_kwargs):
            self.assertTrue(clear.called)
            return pd.DataFrame(), ["target"], [], []

        prediction = np.zeros((1, 21, 155), dtype=np.float32)
        with (
            mock.patch.object(forecast.data, "clear_caches", clear),
            mock.patch.object(forecast, "_validate_live_inputs"),
            mock.patch.object(forecast.model, "area_frame", side_effect=area_frame),
            mock.patch.object(forecast.model, "build_task", return_value={"target": []}),
            mock.patch.object(forecast.model, "predict", return_value=[prediction]),
        ):
            result, timestamps = forecast.zone_quantiles(
                "G1_C1", "2026-07-18T00:00:00+02:00"
            )
        clear.assert_called_once_with()
        self.assertEqual(result["de_lu"].shape, (21, 96))
        self.assertEqual(len(timestamps), 96)

    def test_g3_c5_requires_fresh_german_aux(self):
        cfg = configs.get("G3_C5")
        cutoff = pd.Timestamp("2026-07-17T07:00:00Z")
        future = pd.date_range(cutoff + pd.Timedelta("15min"), periods=155, freq="15min")
        target = pd.Series([1.0], index=pd.DatetimeIndex([cutoff]))
        stale = pd.Series(
            [1.0], index=pd.DatetimeIndex([cutoff.floor("h") - pd.Timedelta("1h")])
        )
        with (
            mock.patch.object(forecast.data, "target", return_value=target),
            mock.patch.object(forecast.data, "aux", return_value=stale),
        ):
            with self.assertRaisesRegex(ValueError, "veraltet"):
                forecast._validate_live_inputs(cfg, "50hertz", cutoff, future)

    def test_g1_c1_needs_no_aux_or_weather(self):
        cfg = configs.get("G1_C1")
        cutoff = pd.Timestamp("2026-07-17T07:00:00Z")
        future = pd.date_range(cutoff + pd.Timedelta("15min"), periods=155, freq="15min")
        target = pd.Series([1.0], index=pd.DatetimeIndex([cutoff]))
        with (
            mock.patch.object(forecast.data, "target", return_value=target),
            mock.patch.object(forecast.data, "aux") as aux,
            mock.patch.object(forecast.data, "weather") as weather,
        ):
            forecast._validate_live_inputs(cfg, "de_lu", cutoff, future)
        aux.assert_not_called()
        weather.assert_not_called()

    def test_luxembourg_c5_requires_price_but_not_smard_generation(self):
        cfg = configs.get("G3_C5")
        cutoff = pd.Timestamp("2026-07-17T07:00:00Z")
        future = pd.date_range(cutoff + pd.Timedelta("15min"), periods=155, freq="15min")
        target = pd.Series([1.0], index=pd.DatetimeIndex([cutoff]))
        price = pd.Series([1.0], index=pd.DatetimeIndex([cutoff.floor("h")]))
        weather_frame = pd.DataFrame(
            {"temperature_2m": np.ones(len(future))}, index=future
        )
        requested_aux: list[str] = []

        def aux(_area, name):
            requested_aux.append(name)
            return price

        with (
            mock.patch.object(forecast.data, "target", return_value=target),
            mock.patch.object(forecast.data, "aux", side_effect=aux),
            mock.patch.object(forecast.data, "weather", return_value=weather_frame),
        ):
            forecast._validate_live_inputs(cfg, "lu", cutoff, future)
        self.assertEqual(requested_aux, ["price"])


class ModelOutputTests(unittest.TestCase):
    class _Tensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class _Pipe:
        def __init__(self, output):
            self.output = output

        def predict(self, tasks, prediction_length, batch_size):
            return [ModelOutputTests._Tensor(self.output) for _ in tasks]

    def test_model_predict_accepts_explicit_native_quantile_axis(self):
        output = np.zeros((1, len(model.CHRONOS_QUANTILE_LEVELS), 7), dtype=np.float32)
        result = model.predict([{"target": np.ones(96)}], 7, model_pipe=self._Pipe(output))
        self.assertEqual(result[0].shape, (1, 21, 7))

    def test_model_predict_rejects_sample_like_axis(self):
        output = np.zeros((1, 100, 7), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "nativen Quantilen"):
            model.predict([{"target": np.ones(96)}], 7, model_pipe=self._Pipe(output))


if __name__ == "__main__":
    unittest.main()
