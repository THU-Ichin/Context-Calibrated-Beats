from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import compare_bpm as bpm


class CsvPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fps = 50.0
        frame_count = 600
        logits = np.full(frame_count, -6.0)
        self.beat_times = np.arange(0.0, 10.01, 0.7)
        for time_seconds in self.beat_times:
            index = min(round(time_seconds * self.fps), frame_count - 1)
            logits[index] = 6.0
        self.frames = bpm.FramePredictions(
            fps=self.fps,
            raw_beat_logits=logits.copy(),
            raw_downbeat_logits=logits.copy(),
            fused_beat_logits=logits.copy(),
            fused_downbeat_logits=logits.copy(),
            window_seconds=30.0,
            hop_seconds=10.0,
            overlap_windows=1,
        )
        self.result = bpm.BeatResult(
            method="beat-this-fused",
            beat_times=self.beat_times,
            downbeat_times=self.beat_times[::4],
        )

    def test_no_beat_ranges_split_grid_without_bridging(self) -> None:
        segments = [
            bpm.ActivitySegment(0, 3.0, 6.0, True, "user", "test gap")
        ]
        ranges = bpm.active_ranges(segments, 10.0)
        normalized, decoding = bpm.build_normalized_grid(
            self.result, self.frames, ranges
        )

        self.assertEqual(ranges, [(0.0, 3.0), (6.0, 10.0)])
        self.assertFalse(
            np.any(
                (normalized.beat_times >= 3.0)
                & (normalized.beat_times < 6.0)
            )
        )
        self.assertFalse(
            np.any(
                (decoding.interval_midpoints >= 3.0)
                & (decoding.interval_midpoints < 6.0)
            )
        )
        for start, end in ranges:
            times = normalized.beat_times[
                (normalized.beat_times >= start)
                & (normalized.beat_times <= end)
            ]
            local_bpm = 60.0 / np.diff(times)
            self.assertTrue(np.all(local_bpm >= bpm.NORMALIZED_BPM_MIN))
            self.assertTrue(np.all(local_bpm < bpm.NORMALIZED_BPM_MAX))

    def test_csv_round_trip_preserves_inference_and_beats(self) -> None:
        ranges = [(0.0, 10.0)]
        normalized, decoding = bpm.build_normalized_grid(
            self.result, self.frames, ranges
        )
        metadata = bpm.InferenceMetadata(
            fps=self.fps,
            window_seconds=30.0,
            hop_seconds=10.0,
            overlap_windows=1,
            sample_rate=22050,
            duration_seconds=10.0,
            audio_path="test.wav",
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            bpm.write_inference_metadata_csv(root / "metadata.csv", metadata)
            bpm.write_frame_predictions_csv(root / "frames.csv", self.frames)
            loaded_metadata = bpm.read_inference_metadata_csv(
                root / "metadata.csv"
            )
            loaded_frames = bpm.read_frame_predictions_csv(
                root / "frames.csv", loaded_metadata
            )

            bpm.write_grid_decisions_csv(root / "grid.csv", decoding)
            loaded_decoding = bpm.read_grid_decisions_csv(root / "grid.csv")

            midpoints, raw_bpm, smooth_bpm = bpm.local_tempo(
                normalized.beat_times
            )
            bpm.write_beats_csv(
                root / "beats.csv",
                normalized,
                midpoints,
                raw_bpm,
                smooth_bpm,
                sample_rate=22050,
                ranges=ranges,
            )
            loaded_beats = bpm.read_beats_csv(
                root / "beats.csv", "beat-this-normalized"
            )

        self.assertEqual(
            len(loaded_frames.fused_beat_logits),
            len(self.frames.fused_beat_logits),
        )
        np.testing.assert_allclose(
            loaded_decoding.base_bpm, decoding.base_bpm, atol=1e-4
        )
        np.testing.assert_allclose(
            loaded_beats.beat_times, normalized.beat_times, atol=1e-9
        )

    def test_p3a_diagnostics_do_not_modify_fused_beats(self) -> None:
        beat_times = np.asarray([0.0, 0.7, 1.4, 2.8, 3.5, 4.2, 4.9, 5.6])
        result = bpm.BeatResult("beat-this-fused", beat_times)
        diagnostics, candidates = bpm.diagnose_fused_beats(
            result, self.frames, [(0.0, 6.0)]
        )

        self.assertEqual(len(diagnostics), len(beat_times) - 1)
        self.assertTrue(
            any(item.candidate_type == "missing_beat" for item in candidates)
        )
        np.testing.assert_array_equal(result.beat_times, beat_times)

        off_result, _, off_decisions = bpm.apply_conservative_repairs(
            result, candidates, [(0.0, 6.0)], "off"
        )
        np.testing.assert_array_equal(off_result.beat_times, beat_times)
        self.assertTrue(any(item.status == "disabled" for item in off_decisions))

    def test_p3b_preview_applies_only_to_repaired_interface(self) -> None:
        beat_times = np.asarray([0.0, 0.7, 1.4, 2.8, 3.5, 4.2, 4.9, 5.6])
        result = bpm.BeatResult("beat-this-fused", beat_times)
        _, candidates = bpm.diagnose_fused_beats(
            result, self.frames, [(0.0, 6.0)]
        )
        repaired_result, records, decisions = bpm.apply_conservative_repairs(
            result, candidates, [(0.0, 6.0)], "preview"
        )

        np.testing.assert_array_equal(result.beat_times, beat_times)
        self.assertIn(2.1, repaired_result.beat_times)
        self.assertTrue(
            any(item.status == "preview_applied" for item in decisions)
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            repaired = root / "repaired.csv"
            decisions_path = root / "decisions.csv"
            bpm.write_repaired_beats_csv(
                repaired,
                repaired_result,
                records,
                22050,
                [(0.0, 6.0)],
            )
            bpm.write_repair_decisions_csv(decisions_path, decisions)
            loaded = bpm.read_beats_csv(repaired, "beat-this-repaired")
            loaded_decisions = bpm.read_repair_decisions_csv(decisions_path)

        np.testing.assert_array_equal(loaded.beat_times, repaired_result.beat_times)
        self.assertEqual(len(loaded_decisions), len(decisions))

        repeated_result, _, repeated_decisions = bpm.apply_conservative_repairs(
            result, candidates, [(0.0, 6.0)], "preview"
        )
        np.testing.assert_array_equal(
            repeated_result.beat_times, repaired_result.beat_times
        )
        self.assertEqual(
            [item.status for item in repeated_decisions],
            [item.status for item in decisions],
        )

    def test_p3a_detects_extra_beat_and_protects_smooth_motion(self) -> None:
        extra_times = np.asarray([0.0, 0.7, 1.05, 1.4, 2.1, 2.8, 3.5, 4.2])
        _, extra_candidates = bpm.diagnose_fused_beats(
            bpm.BeatResult("beat-this-fused", extra_times),
            self.frames,
            [(0.0, 5.0)],
        )
        self.assertTrue(
            any(item.candidate_type == "extra_beat" for item in extra_candidates)
        )
        extra_repaired, _, extra_decisions = bpm.apply_conservative_repairs(
            bpm.BeatResult("beat-this-fused", extra_times),
            extra_candidates,
            [(0.0, 5.0)],
            "preview",
        )
        self.assertNotIn(1.05, extra_repaired.beat_times)
        self.assertTrue(
            any(
                item.candidate_type == "extra_beat"
                and item.status == "preview_applied"
                for item in extra_decisions
            )
        )

        periods = np.asarray([0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.70])
        motion_times = np.concatenate(([0.0], np.cumsum(periods)))
        diagnostics, motion_candidates = bpm.diagnose_fused_beats(
            bpm.BeatResult("beat-this-fused", motion_times),
            self.frames,
            [(0.0, 5.0)],
        )
        self.assertTrue(
            any(item.classification == "protected_tempo_motion" for item in diagnostics)
        )
        self.assertTrue(
            any(item.candidate_type == "tempo_motion" for item in motion_candidates)
        )
        motion_repaired, _, motion_decisions = bpm.apply_conservative_repairs(
            bpm.BeatResult("beat-this-fused", motion_times),
            motion_candidates,
            [(0.0, 5.0)],
            "preview",
        )
        np.testing.assert_array_equal(motion_repaired.beat_times, motion_times)
        self.assertFalse(
            any(item.status == "preview_applied" for item in motion_decisions)
        )

    def test_p3c_phase_grid_keeps_transition_intervals_continuous(self) -> None:
        beat_times = np.asarray(
            [
                0.00,
                0.34,
                0.68,
                1.02,
                1.36,
                1.70,
                2.04,
                2.72,
                2.88,
                3.06,
                3.22,
                3.40,
                3.56,
                3.92,
                4.24,
                4.60,
                4.94,
            ]
        )
        result, events, transitions = bpm.build_phase_aware_grid(
            bpm.BeatResult("beat-this-repaired", beat_times),
            self.frames,
            [(0.0, 5.0)],
        )
        intervals = np.diff(result.beat_times)

        self.assertTrue(np.all(intervals > 60.0 / bpm.NORMALIZED_BPM_MAX))
        self.assertTrue(np.all(intervals <= 60.0 / bpm.NORMALIZED_BPM_MIN))
        self.assertLess(float(np.max(intervals)), 0.45)
        self.assertTrue(
            {item.selected_scale for item in events}.issubset(set(bpm.GRID_SCALES))
        )
        self.assertGreater(len(events), 10)
        self.assertIsInstance(transitions, list)

    def test_p3c_phase_grid_does_not_bridge_no_beat_ranges(self) -> None:
        ranges = [(0.0, 3.0), (6.0, 10.0)]
        result, events, _ = bpm.build_phase_aware_grid(
            self.result,
            self.frames,
            ranges,
        )

        self.assertFalse(
            np.any(
                (result.beat_times >= 3.0)
                & (result.beat_times < 6.0)
            )
        )
        self.assertEqual({item.activity_segment_id for item in events}, {0, 1})

    def test_p3c_bidirectional_refinement_uses_both_fixed_anchors(self) -> None:
        times = np.asarray([0.00, 0.36, 0.76, 1.16, 1.52])
        logits = np.full(600, -6.0)
        for time_seconds in (0.36, 0.74, 1.16):
            logits[round(time_seconds * self.fps)] = 6.0
        frames = bpm.FramePredictions(
            fps=self.fps,
            raw_beat_logits=logits.copy(),
            raw_downbeat_logits=logits.copy(),
            fused_beat_logits=logits.copy(),
            fused_downbeat_logits=logits.copy(),
            window_seconds=30.0,
            hop_seconds=10.0,
            overlap_windows=1,
        )
        events = []
        for index, time_seconds in enumerate(times):
            bridge = index == 2
            events.append(
                bpm.PhaseGridEvent(
                    grid_beat_index=index + 1,
                    activity_segment_id=0,
                    beat_time_seconds=float(time_seconds),
                    selected_scale=1.0,
                    target_period_seconds=0.40,
                    predicted_time_seconds=float(time_seconds),
                    phase_residual_seconds=0.0,
                    beat_probability=0.01,
                    downbeat_probability=0.0,
                    event_source="phase_bridge" if bridge else "acoustic_peak",
                    transition_type="phase_bridge" if bridge else "stable",
                    path_cost=float(index),
                )
            )
        result, refined, transitions = bpm.refine_phase_grid_bidirectionally(
            bpm.BeatResult("beat-this-phase-aware", times),
            events,
            [],
            frames,
        )

        self.assertEqual(result.beat_times[1], times[1])
        self.assertEqual(result.beat_times[3], times[3])
        self.assertTrue(np.any(np.abs(result.beat_times - times) > 1e-6))
        self.assertTrue(
            any(item.transition_type == "bidirectional_bridge" for item in transitions)
        )
        intervals = np.diff(result.beat_times)
        self.assertTrue(np.all(intervals > 60.0 / bpm.NORMALIZED_BPM_MAX))
        self.assertTrue(np.all(intervals <= 60.0 / bpm.NORMALIZED_BPM_MIN))

    def test_p3c_future_window_backtracks_and_restores_missing_beat(self) -> None:
        period = 0.34
        correct_grid = np.arange(0.0, 12.01, period)
        past = correct_grid[correct_grid <= 3.40 + 1e-9]
        wrong_phase = np.linspace(3.90, 6.62, 8)
        bridges = np.asarray([6.96, 7.22])
        future = correct_grid[correct_grid >= 7.48 - 1e-9]
        times = np.concatenate((past, wrong_phase, bridges, future))

        logits = np.full(round(12.5 * self.fps), -6.0)
        for time_seconds in correct_grid:
            logits[round(time_seconds * self.fps)] = 6.0
        frames = bpm.FramePredictions(
            fps=self.fps,
            raw_beat_logits=logits.copy(),
            raw_downbeat_logits=np.full_like(logits, -6.0),
            fused_beat_logits=logits.copy(),
            fused_downbeat_logits=np.full_like(logits, -6.0),
            window_seconds=30.0,
            hop_seconds=10.0,
            overlap_windows=1,
        )
        events = []
        for index, time_seconds in enumerate(times):
            is_bridge = time_seconds in bridges
            events.append(
                bpm.PhaseGridEvent(
                    grid_beat_index=index + 1,
                    activity_segment_id=0,
                    beat_time_seconds=float(time_seconds),
                    selected_scale=1.0,
                    target_period_seconds=period,
                    predicted_time_seconds=float(time_seconds),
                    phase_residual_seconds=0.0,
                    beat_probability=(
                        0.01 if is_bridge else float(time_seconds >= 7.48)
                    ),
                    downbeat_probability=0.0,
                    event_source="phase_bridge" if is_bridge else "acoustic_peak",
                    transition_type="phase_bridge" if is_bridge else "stable",
                    path_cost=float(index),
                )
            )

        result, refined, transitions = bpm.refine_phase_grid_bidirectionally(
            bpm.BeatResult("beat-this-phase-aware", times),
            events,
            [],
            frames,
        )
        self.assertEqual(len(result.beat_times), len(times) + 1)
        corrected = result.beat_times[
            (result.beat_times > 3.40) & (result.beat_times < 7.48)
        ]
        residuals = np.abs(
            corrected / period - np.round(corrected / period)
        ) * period
        self.assertLess(float(np.max(residuals)), 0.02)
        self.assertTrue(
            any(
                item.transition_type == "future_confirmed_backtrack"
                for item in transitions
            )
        )
        self.assertTrue(
            any(item.event_source == "future_confirmed_backtrack" for item in refined)
        )

    def test_p3c_long_backtrack_accepts_half_rate_future_evidence(self) -> None:
        period = 0.30
        correct_grid = np.arange(0.0, 20.11, period)
        past = correct_grid[correct_grid <= 5.10 + 1e-9]
        entrance = np.asarray([5.40, 5.70])
        wrong_middle = np.arange(6.10, 14.51, 0.40)
        future = correct_grid[correct_grid >= 15.00 - 1e-9]
        times = np.concatenate((past, entrance, wrong_middle, future))

        logits = np.full(round(20.5 * self.fps), -6.0)
        for time_seconds in correct_grid:
            logits[round(time_seconds * self.fps)] = 6.0
        frames = bpm.FramePredictions(
            fps=self.fps,
            raw_beat_logits=logits.copy(),
            raw_downbeat_logits=np.full_like(logits, -6.0),
            fused_beat_logits=logits.copy(),
            fused_downbeat_logits=np.full_like(logits, -6.0),
            window_seconds=30.0,
            hop_seconds=10.0,
            overlap_windows=1,
        )
        events = []
        middle_start = len(past) + len(entrance)
        future_start = len(past) + len(entrance) + len(wrong_middle)
        for index, time_seconds in enumerate(times):
            is_middle = middle_start <= index < future_start
            is_transition = len(past) <= index < middle_start + 1
            in_supported_window = index < len(past) or index >= future_start
            supported_peak = in_supported_window and index % 2 == 0
            events.append(
                bpm.PhaseGridEvent(
                    grid_beat_index=index + 1,
                    activity_segment_id=0,
                    beat_time_seconds=float(time_seconds),
                    selected_scale=(0.5 if is_middle and index % 2 else 1.0),
                    target_period_seconds=period,
                    predicted_time_seconds=float(time_seconds),
                    phase_residual_seconds=0.0,
                    beat_probability=1.0 if supported_peak or index in {
                        len(past),
                        len(past) + 1,
                    } else 0.01,
                    downbeat_probability=0.0,
                    event_source=(
                        "phase_bridge"
                        if is_transition
                        else "acoustic_peak"
                        if supported_peak
                        else "theoretical_grid"
                    ),
                    transition_type=(
                        "phase_bridge" if is_transition else "stable"
                    ),
                    path_cost=float(index),
                )
            )

        result, refined, transitions = bpm.refine_phase_grid_bidirectionally(
            bpm.BeatResult("beat-this-phase-aware", times),
            events,
            [],
            frames,
        )
        self.assertGreater(len(result.beat_times), len(times))
        rebuilt = result.beat_times[
            (result.beat_times >= 5.40) & (result.beat_times <= 15.00)
        ]
        np.testing.assert_allclose(np.diff(rebuilt), period, atol=1e-6)
        self.assertTrue(
            any(
                item.transition_type == "future_long_backtrack"
                for item in transitions
            )
        )
        self.assertTrue(
            any(item.event_source == "future_long_backtrack" for item in refined)
        )


if __name__ == "__main__":
    unittest.main()
