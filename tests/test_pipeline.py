from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import CCB as ccb


class CsvPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fps = 50.0
        frame_count = 600
        logits = np.full(frame_count, -6.0)
        self.beat_times = np.arange(0.0, 10.01, 0.7)
        for time_seconds in self.beat_times:
            index = min(round(time_seconds * self.fps), frame_count - 1)
            logits[index] = 6.0
        self.frames = ccb.FramePredictions(
            fps=self.fps,
            fused_beat_logits=logits.copy(),
            fused_downbeat_logits=logits.copy(),
            window_seconds=30.0,
            hop_seconds=10.0,
            overlap_windows=1,
        )
        self.result = ccb.BeatResult(
            method="beat-this-fused",
            beat_times=self.beat_times,
            downbeat_times=self.beat_times[::4],
        )

    def test_cli_exposes_only_the_production_output_contract(self) -> None:
        parser = ccb.build_parser()
        args = parser.parse_args(["song.mp3"])
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }

        self.assertEqual(args.output, Path("results"))
        self.assertFalse(args.refresh_cache)
        self.assertFalse(args.no_click)
        self.assertIsNone(args.cache_dir)
        self.assertEqual(args.music_gain, 0.1)
        self.assertEqual(args.click_gain, 0.9)
        self.assertIn("--refresh-cache", option_strings)
        self.assertIn("--no-click", option_strings)
        self.assertIn("--cache-dir", option_strings)
        self.assertIn("--version", option_strings)
        self.assertEqual(ccb.CCB_VERSION, "1.0.0")
        self.assertNotIn("--output-profile", option_strings)
        self.assertNotIn("--repair-mode", option_strings)
        self.assertNotIn("--grid-mode", option_strings)

        cache = ccb.inference_cache_paths(
            Path("song.mp3"), args.output, Path("cache")
        )
        self.assertEqual(
            cache["root"],
            Path("cache") / ccb.audio_storage_key(Path("song.mp3")),
        )
        self.assertEqual(cache["metadata"].name, "inference.csv")
        self.assertEqual(cache["frames"].name, "frames.csv")
        self.assertEqual(cache["fused"].name, "fused_beats.csv")

    def test_click_mix_uses_gains_and_prevents_clipping(self) -> None:
        result = ccb.BeatResult(
            method="ccb-final",
            beat_times=np.asarray([0.0]),
            downbeat_times=np.asarray([], dtype=float),
        )
        music = np.asarray([1.0, -1.0])
        clicks = np.asarray([1.0, 1.0])

        with (
            patch.object(ccb.librosa, "clicks", return_value=clicks),
            patch.object(ccb.librosa.util, "normalize", side_effect=lambda x: x),
            patch.object(ccb.sf, "write") as write,
        ):
            ccb.write_click_track(
                Path("click.wav"),
                music,
                22050,
                result,
                music_gain=1.0,
                click_gain=1.0,
            )

        written = write.call_args.args[1]
        np.testing.assert_allclose(written, np.asarray([1.0, 0.0]))
        self.assertLessEqual(float(np.max(np.abs(written))), 1.0)

    def test_no_beat_ranges_split_grid_without_bridging(self) -> None:
        segments = [
            ccb.ActivitySegment(0, 3.0, 6.0, True, "user", "test gap")
        ]
        ranges = ccb.active_ranges(segments, 10.0)
        normalized, events, _ = ccb.build_phase_aware_grid(
            self.result, self.frames, ranges
        )
        decoding = ccb.phase_events_to_grid_decoding(events)

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
            self.assertTrue(np.all(local_bpm >= ccb.NORMALIZED_BPM_MIN))
            self.assertTrue(np.all(local_bpm < ccb.NORMALIZED_BPM_MAX))

    def test_csv_round_trip_preserves_inference_and_beats(self) -> None:
        ranges = [(0.0, 10.0)]
        normalized, events, _ = ccb.build_phase_aware_grid(
            self.result, self.frames, ranges
        )
        metadata = ccb.InferenceMetadata(
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
            ccb.write_inference_metadata_csv(root / "metadata.csv", metadata)
            ccb.write_frame_predictions_csv(root / "frames.csv", self.frames)
            loaded_metadata = ccb.read_inference_metadata_csv(
                root / "metadata.csv"
            )
            loaded_frames = ccb.read_frame_predictions_csv(
                root / "frames.csv", loaded_metadata
            )

            midpoints, raw_bpm, smooth_bpm = ccb.local_tempo(
                normalized.beat_times
            )
            ccb.write_beats_csv(
                root / "beats.csv",
                normalized,
                midpoints,
                raw_bpm,
                smooth_bpm,
                sample_rate=22050,
                ranges=ranges,
            )
            loaded_beats = ccb.read_beats_csv(
                root / "beats.csv", "beat-this-normalized"
            )

        self.assertEqual(
            len(loaded_frames.fused_beat_logits),
            len(self.frames.fused_beat_logits),
        )
        np.testing.assert_allclose(
            loaded_beats.beat_times, normalized.beat_times, atol=1e-9
        )

    def test_cache_fingerprint_rejects_changed_audio_or_settings(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "song.wav"
            audio.write_bytes(b"original-audio")
            cache_paths = ccb.inference_cache_paths(
                audio, root / "results", root / "cache"
            )
            cache_paths["root"].mkdir(parents=True)
            audio_stat = audio.stat()
            ccb.write_inference_metadata_csv(
                cache_paths["metadata"],
                ccb.InferenceMetadata(
                    fps=50.0,
                    window_seconds=30.0,
                    hop_seconds=10.0,
                    overlap_windows=1,
                    sample_rate=22050,
                    duration_seconds=1.0,
                    audio_path=str(audio.resolve()),
                    cache_schema_version=ccb.CACHE_SCHEMA_VERSION,
                    audio_size_bytes=audio_stat.st_size,
                    audio_mtime_ns=audio_stat.st_mtime_ns,
                    checkpoint="final0",
                    beat_this_version=ccb.beat_this_version(),
                    ccb_version=ccb.CCB_VERSION,
                ),
            )
            cache_paths["frames"].write_text("frames", encoding="utf-8")
            cache_paths["fused"].write_text("beats", encoding="utf-8")

            self.assertTrue(
                ccb.inference_cache_available(
                    audio,
                    root / "results",
                    root / "cache",
                    sample_rate=22050,
                    checkpoint="final0",
                    hop_seconds=10.0,
                )
            )
            self.assertFalse(
                ccb.inference_cache_available(
                    audio,
                    root / "results",
                    root / "cache",
                    sample_rate=22050,
                    checkpoint="other",
                    hop_seconds=10.0,
                )
            )

            audio.write_bytes(b"changed-audio-with-another-size")
            self.assertFalse(
                ccb.inference_cache_available(
                    audio,
                    root / "results",
                    root / "cache",
                    sample_rate=22050,
                    checkpoint="final0",
                    hop_seconds=10.0,
                )
            )

    def test_p3a_diagnostics_do_not_modify_fused_beats(self) -> None:
        beat_times = np.asarray([0.0, 0.7, 1.4, 2.8, 3.5, 4.2, 4.9, 5.6])
        result = ccb.BeatResult("beat-this-fused", beat_times)
        diagnostics = ccb.diagnose_fused_beats(
            result, self.frames, [(0.0, 6.0)]
        )

        self.assertEqual(len(diagnostics), len(beat_times) - 1)
        self.assertTrue(
            any(item.classification == "strong_outlier" for item in diagnostics)
        )
        np.testing.assert_array_equal(result.beat_times, beat_times)

    def test_diagnostics_protect_smooth_tempo_motion(self) -> None:
        periods = np.asarray([0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.70])
        motion_times = np.concatenate(([0.0], np.cumsum(periods)))
        diagnostics = ccb.diagnose_fused_beats(
            ccb.BeatResult("beat-this-fused", motion_times),
            self.frames,
            [(0.0, 5.0)],
        )
        self.assertTrue(
            any(item.classification == "protected_tempo_motion" for item in diagnostics)
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
        result, events, transitions = ccb.build_phase_aware_grid(
            ccb.BeatResult("beat-this-repaired", beat_times),
            self.frames,
            [(0.0, 5.0)],
        )
        intervals = np.diff(result.beat_times)

        self.assertTrue(np.all(intervals > 60.0 / ccb.NORMALIZED_BPM_MAX))
        self.assertTrue(np.all(intervals <= 60.0 / ccb.NORMALIZED_BPM_MIN))
        self.assertLess(float(np.max(intervals)), 0.45)
        self.assertTrue(
            {item.selected_scale for item in events}.issubset(set(ccb.GRID_SCALES))
        )
        self.assertGreater(len(events), 10)
        self.assertIsInstance(transitions, list)

    def test_p3c_phase_grid_does_not_bridge_no_beat_ranges(self) -> None:
        ranges = [(0.0, 3.0), (6.0, 10.0)]
        result, events, _ = ccb.build_phase_aware_grid(
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

    def test_p3c_segment_start_uses_stable_future_phase(self) -> None:
        source_times = np.asarray(
            [0.52, 1.22, 2.08, 2.76, 3.46, 4.16, 4.86, 5.54]
        )
        logits = np.full(600, -6.0)
        for time_seconds in source_times:
            logits[round(time_seconds * self.fps)] = 6.0
        frames = ccb.FramePredictions(
            fps=self.fps,
            fused_beat_logits=logits.copy(),
            fused_downbeat_logits=np.full_like(logits, -6.0),
            window_seconds=30.0,
            hop_seconds=10.0,
            overlap_windows=1,
        )

        result, events, transitions = ccb.build_phase_aware_grid(
            ccb.BeatResult("beat-this-fused", source_times),
            frames,
            [(0.0, 6.0)],
        )

        self.assertAlmostEqual(result.beat_times[0], 0.68, delta=0.03)
        self.assertEqual(events[0].event_source, "future_phase_start")
        self.assertTrue(
            any(item.transition_type == "future_phase_start" for item in transitions)
        )
        future_reference = np.arange(0.68, 3.49, 0.35)
        np.testing.assert_allclose(
            result.beat_times[: len(future_reference)],
            future_reference,
            atol=0.04,
        )

    def test_p3c_segment_start_keeps_an_already_aligned_anchor(self) -> None:
        source_times = np.arange(0.20, 5.81, 0.70)
        logits = np.full(600, -6.0)
        for time_seconds in source_times:
            logits[round(time_seconds * self.fps)] = 6.0
        frames = ccb.FramePredictions(
            fps=self.fps,
            fused_beat_logits=logits.copy(),
            fused_downbeat_logits=np.full_like(logits, -6.0),
            window_seconds=30.0,
            hop_seconds=10.0,
            overlap_windows=1,
        )

        result, events, transitions = ccb.build_phase_aware_grid(
            ccb.BeatResult("beat-this-fused", source_times),
            frames,
            [(0.0, 6.0)],
        )

        self.assertAlmostEqual(result.beat_times[0], 0.20, places=6)
        self.assertEqual(events[0].event_source, "acoustic_anchor")
        self.assertFalse(
            any(item.transition_type == "future_phase_start" for item in transitions)
        )

    def test_p3c_bidirectional_refinement_uses_both_fixed_anchors(self) -> None:
        times = np.asarray([0.00, 0.36, 0.76, 1.16, 1.52])
        logits = np.full(600, -6.0)
        for time_seconds in (0.36, 0.74, 1.16):
            logits[round(time_seconds * self.fps)] = 6.0
        frames = ccb.FramePredictions(
            fps=self.fps,
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
                ccb.PhaseGridEvent(
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
        result, refined, transitions = ccb.refine_phase_grid_bidirectionally(
            ccb.BeatResult("beat-this-phase-aware", times),
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
        self.assertTrue(np.all(intervals > 60.0 / ccb.NORMALIZED_BPM_MAX))
        self.assertTrue(np.all(intervals <= 60.0 / ccb.NORMALIZED_BPM_MIN))

    def test_p3c4_invalid_refinement_falls_back_to_greedy_phase_grid(self) -> None:
        greedy, events, transitions = ccb.build_phase_aware_grid(
            self.result,
            self.frames,
            [(0.0, 10.0)],
        )
        invalid_times = greedy.beat_times.copy()
        invalid_times[2] = invalid_times[1] + 0.10
        invalid = ccb.BeatResult(
            "beat-this-phase-aware",
            invalid_times,
            downbeat_times=greedy.downbeat_times,
        )

        with patch.object(
            ccb,
            "refine_phase_grid_bidirectionally",
            return_value=(invalid, events, transitions),
        ):
            result, final_events, _, applied, fallback, reason = (
                ccb.finalize_phase_grid(
                    greedy,
                    events,
                    transitions,
                    self.frames,
                    [(0.0, 10.0)],
                    "bidirectional",
                )
            )

        np.testing.assert_array_equal(result.beat_times, greedy.beat_times)
        self.assertEqual(len(final_events), len(events))
        self.assertFalse(applied)
        self.assertTrue(fallback)
        self.assertIn("result timestamps differ", reason)

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
        frames = ccb.FramePredictions(
            fps=self.fps,
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
                ccb.PhaseGridEvent(
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

        result, refined, transitions = ccb.refine_phase_grid_bidirectionally(
            ccb.BeatResult("beat-this-phase-aware", times),
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

    def test_p3c_reconciles_one_extra_beat_between_stable_anchors(self) -> None:
        left_times = np.arange(0.0, 2.2801, 0.38)
        left_anchor = float(left_times[-1])
        right_anchor = 6.84
        crowded_middle = np.linspace(left_anchor, right_anchor, 14)[1:-1]
        right_times = right_anchor + np.arange(0, 7, dtype=float) * 0.38
        times = np.concatenate((left_times, crowded_middle, right_times))
        events = []
        for index, time_seconds in enumerate(times):
            in_middle = left_anchor < time_seconds < right_anchor
            events.append(
                ccb.PhaseGridEvent(
                    grid_beat_index=index + 1,
                    activity_segment_id=0,
                    beat_time_seconds=float(time_seconds),
                    selected_scale=(2.0 if time_seconds < 5.0 else 1.0),
                    target_period_seconds=0.38,
                    predicted_time_seconds=float(time_seconds),
                    phase_residual_seconds=0.0,
                    beat_probability=0.95 if not in_middle else 0.10,
                    downbeat_probability=0.0,
                    event_source="phase_bridge" if in_middle else "acoustic_peak",
                    transition_type="phase_bridge" if in_middle else "stable",
                    path_cost=float(index),
                )
            )
        transition_specs = (
            (2.28, 3.10, "phase_bridge", 2.0, 2.0),
            (3.10, 3.90, "scale_switch", 2.0, 0.5),
            (3.90, 4.70, "scale_switch", 0.5, 2.0),
            (4.70, 5.50, "bidirectional_bridge", 2.0, 1.0),
            (5.50, 6.50, "phase_bridge", 1.0, 1.0),
        )
        transitions = [
            ccb.GridTransition(
                transition_id=index + 1,
                activity_segment_id=0,
                start_seconds=start,
                end_seconds=end,
                previous_scale=previous_scale,
                next_scale=next_scale,
                previous_period_seconds=0.38,
                next_period_seconds=0.35,
                phase_adjustment_seconds=0.04,
                transition_type=transition_type,
                diagnostic_note="synthetic count error",
            )
            for index, (
                start,
                end,
                transition_type,
                previous_scale,
                next_scale,
            ) in enumerate(transition_specs)
        ]

        repaired, repaired_transitions = ccb.reconcile_anchor_beat_counts(
            events,
            transitions,
            self.frames,
        )
        repaired_times = np.asarray(
            [item.beat_time_seconds for item in repaired], dtype=float
        )
        left_index = int(np.flatnonzero(np.isclose(repaired_times, left_anchor))[0])
        right_index = int(np.flatnonzero(np.isclose(repaired_times, right_anchor))[0])

        self.assertEqual(right_index - left_index, 12)
        np.testing.assert_allclose(
            np.diff(repaired_times[left_index : right_index + 1]),
            0.38,
            atol=1e-9,
        )
        self.assertTrue(
            any(
                item.transition_type == "anchor_count_reconciled"
                for item in repaired_transitions
            )
        )

    def test_p3c_reconciles_one_missing_beat_between_stable_anchors(self) -> None:
        left_times = np.arange(0.0, 2.2801, 0.38)
        left_anchor = float(left_times[-1])
        right_anchor = 6.84
        sparse_middle = np.linspace(left_anchor, right_anchor, 12)[1:-1]
        right_times = right_anchor + np.arange(0, 7, dtype=float) * 0.38
        times = np.concatenate((left_times, sparse_middle, right_times))
        events = [
            ccb.PhaseGridEvent(
                grid_beat_index=index + 1,
                activity_segment_id=0,
                beat_time_seconds=float(time_seconds),
                selected_scale=1.0,
                target_period_seconds=0.38,
                predicted_time_seconds=float(time_seconds),
                phase_residual_seconds=0.0,
                beat_probability=(
                    0.10 if left_anchor < time_seconds < right_anchor else 0.95
                ),
                downbeat_probability=0.0,
                event_source=(
                    "phase_bridge"
                    if left_anchor < time_seconds < right_anchor
                    else "acoustic_peak"
                ),
                transition_type=(
                    "phase_bridge"
                    if left_anchor < time_seconds < right_anchor
                    else "stable"
                ),
                path_cost=float(index),
            )
            for index, time_seconds in enumerate(times)
        ]
        transitions = [
            ccb.GridTransition(
                transition_id=index + 1,
                activity_segment_id=0,
                start_seconds=start,
                end_seconds=end,
                previous_scale=1.0,
                next_scale=1.0,
                previous_period_seconds=0.38,
                next_period_seconds=0.41,
                phase_adjustment_seconds=0.04,
                transition_type="phase_bridge",
                diagnostic_note="synthetic missing beat",
            )
            for index, (start, end) in enumerate(
                ((2.28, 3.10), (3.10, 4.10), (4.10, 5.10), (5.10, 6.50))
            )
        ]

        repaired, repaired_transitions = ccb.reconcile_anchor_beat_counts(
            events,
            transitions,
            self.frames,
        )
        repaired_times = np.asarray(
            [item.beat_time_seconds for item in repaired], dtype=float
        )
        left_index = int(np.flatnonzero(np.isclose(repaired_times, left_anchor))[0])
        right_index = int(np.flatnonzero(np.isclose(repaired_times, right_anchor))[0])

        self.assertEqual(right_index - left_index, 12)
        np.testing.assert_allclose(
            np.diff(repaired_times[left_index : right_index + 1]),
            0.38,
            atol=1e-9,
        )
        self.assertTrue(
            any(
                item.transition_type == "anchor_count_reconciled"
                for item in repaired_transitions
            )
        )

    def test_p3c_reconciles_a_4_5_4_downbeat_bar_pattern(self) -> None:
        period = 0.34
        downbeats = np.asarray([0.0, 1.36, 2.72, 4.08])
        times = np.concatenate(
            (
                np.linspace(downbeats[0], downbeats[1], 5),
                np.linspace(downbeats[1], downbeats[2], 6)[1:],
                np.linspace(downbeats[2], downbeats[3], 5)[1:],
            )
        )
        events = []
        for index, time_seconds in enumerate(times):
            in_bad_bar = downbeats[1] < time_seconds < downbeats[2]
            is_downbeat = bool(np.any(np.isclose(time_seconds, downbeats)))
            events.append(
                ccb.PhaseGridEvent(
                    grid_beat_index=index + 1,
                    activity_segment_id=0,
                    beat_time_seconds=float(time_seconds),
                    selected_scale=1.0,
                    target_period_seconds=period,
                    predicted_time_seconds=float(time_seconds),
                    phase_residual_seconds=0.0,
                    beat_probability=0.95 if not in_bad_bar else 0.20,
                    downbeat_probability=0.95 if is_downbeat else 0.0,
                    event_source="phase_bridge" if in_bad_bar else "acoustic_peak",
                    transition_type="phase_bridge" if in_bad_bar else "stable",
                    path_cost=float(index),
                )
            )
        transitions = [
            ccb.GridTransition(
                transition_id=index + 1,
                activity_segment_id=0,
                start_seconds=float(start),
                end_seconds=float(end),
                previous_scale=1.0,
                next_scale=1.0,
                previous_period_seconds=period,
                next_period_seconds=period * 0.8,
                phase_adjustment_seconds=0.04,
                transition_type="phase_bridge",
                diagnostic_note="synthetic five-beat middle bar",
            )
            for index, (start, end) in enumerate(
                ((1.36, 1.85), (1.85, 2.25), (2.25, 2.72))
            )
        ]

        repaired, repaired_transitions = ccb.reconcile_anchor_beat_counts(
            events,
            transitions,
            self.frames,
        )
        repaired_times = np.asarray(
            [item.beat_time_seconds for item in repaired], dtype=float
        )
        left = int(np.flatnonzero(np.isclose(repaired_times, downbeats[1]))[0])
        right = int(np.flatnonzero(np.isclose(repaired_times, downbeats[2]))[0])

        self.assertEqual(right - left, 4)
        np.testing.assert_allclose(
            np.diff(repaired_times[left : right + 1]),
            period,
            atol=1e-9,
        )
        self.assertTrue(
            any(
                item.transition_type == "anchor_count_reconciled"
                and "source=downbeat_bars" in item.diagnostic_note
                for item in repaired_transitions
            )
        )

    def test_p3c_collapses_only_a_sustained_close_doublet_run(self) -> None:
        sustained = np.asarray(
            [2.00, 2.08, 2.80, 2.88, 3.60, 3.68, 4.40, 4.48]
        )
        isolated = np.asarray([0.00, 0.08, 0.50, 1.00])
        times = np.concatenate((isolated, sustained))

        def probabilities(time_seconds: float) -> tuple[float, float]:
            is_right_peak = any(
                np.isclose(time_seconds, value)
                for value in (0.08, 2.08, 2.88, 3.68, 4.48)
            )
            return (0.90 if is_right_peak else 0.60, 0.0)

        collapsed, runs = ccb._collapse_repeated_close_doublets(
            times,
            probabilities,
        )

        np.testing.assert_allclose(
            collapsed,
            np.asarray([0.00, 0.08, 0.50, 1.00, 2.08, 2.88, 3.68, 4.48]),
        )
        self.assertEqual(runs, [(2.0, 4.48, 4)])

    def test_p3c_downbeat_bar_uses_nearby_original_peak_evidence(self) -> None:
        downbeats = np.asarray([0.00, 1.56, 3.10, 4.62])
        times = np.concatenate(
            (
                np.linspace(downbeats[0], downbeats[1], 5),
                np.linspace(downbeats[1], downbeats[2], 6)[1:],
                np.linspace(downbeats[2], downbeats[3], 5)[1:],
            )
        )
        frame_count = 300
        beat_logits = np.full(frame_count, -6.0)
        downbeat_logits = np.full(frame_count, -6.0)
        evidence_times = np.asarray([0.00, 1.60, 3.16, 4.70])
        for time_seconds in evidence_times:
            index = min(round(time_seconds * self.fps), frame_count - 1)
            beat_logits[index] = 6.0
            downbeat_logits[index] = 6.0
        frames = ccb.FramePredictions(
            fps=self.fps,
            fused_beat_logits=beat_logits.copy(),
            fused_downbeat_logits=downbeat_logits.copy(),
            window_seconds=30.0,
            hop_seconds=10.0,
            overlap_windows=1,
        )
        events = []
        for index, time_seconds in enumerate(times):
            is_downbeat = bool(np.any(np.isclose(time_seconds, downbeats)))
            in_bad_bar = downbeats[1] < time_seconds < downbeats[2]
            events.append(
                ccb.PhaseGridEvent(
                    grid_beat_index=index + 1,
                    activity_segment_id=0,
                    beat_time_seconds=float(time_seconds),
                    selected_scale=1.0,
                    target_period_seconds=0.39,
                    predicted_time_seconds=float(time_seconds),
                    phase_residual_seconds=0.0,
                    beat_probability=0.60 if is_downbeat else 0.20,
                    downbeat_probability=0.60 if is_downbeat else 0.0,
                    event_source="phase_bridge" if in_bad_bar else "acoustic_peak",
                    transition_type="phase_bridge" if in_bad_bar else "stable",
                    path_cost=float(index),
                )
            )
        transitions = [
            ccb.GridTransition(
                transition_id=index + 1,
                activity_segment_id=0,
                start_seconds=float(start),
                end_seconds=float(end),
                previous_scale=1.0,
                next_scale=2.0,
                previous_period_seconds=0.39,
                next_period_seconds=0.30,
                phase_adjustment_seconds=0.05,
                transition_type="phase_bridge",
                diagnostic_note="synthetic shifted downbeat evidence",
            )
            for index, (start, end) in enumerate(
                ((1.56, 2.00), (2.00, 2.50), (2.50, 3.10))
            )
        ]

        repaired, repaired_transitions = ccb.reconcile_anchor_beat_counts(
            events,
            transitions,
            frames,
        )
        repaired_times = np.asarray(
            [item.beat_time_seconds for item in repaired], dtype=float
        )
        left = int(np.flatnonzero(np.isclose(repaired_times, downbeats[1]))[0])
        right = int(np.flatnonzero(np.isclose(repaired_times, downbeats[2]))[0])

        self.assertEqual(right - left, 4)
        self.assertTrue(
            any(
                item.transition_type == "anchor_count_reconciled"
                and "source=downbeat_bars" in item.diagnostic_note
                for item in repaired_transitions
            )
        )

    def test_p4_marks_scale_churn_as_unreliable_without_changing_beats(self) -> None:
        times = np.arange(0.0, 12.01, 0.30)
        events = [
            ccb.PhaseGridEvent(
                grid_beat_index=index + 1,
                activity_segment_id=0,
                beat_time_seconds=float(time_seconds),
                selected_scale=1.0,
                target_period_seconds=0.30,
                predicted_time_seconds=float(time_seconds),
                phase_residual_seconds=0.0,
                beat_probability=0.20 if 4.0 <= time_seconds < 8.0 else 0.90,
                downbeat_probability=0.0,
                event_source="theoretical_grid" if 4.0 <= time_seconds < 8.0 else "acoustic_peak",
                transition_type="stable",
                path_cost=float(index),
            )
            for index, time_seconds in enumerate(times)
        ]
        transitions = [
            ccb.GridTransition(
                transition_id=index + 1,
                activity_segment_id=0,
                start_seconds=time_seconds - 0.2,
                end_seconds=time_seconds,
                previous_scale=1.0,
                next_scale=0.5 if index % 2 == 0 else 1.0,
                previous_period_seconds=0.30,
                next_period_seconds=0.30,
                phase_adjustment_seconds=0.0,
                transition_type="scale_switch",
                diagnostic_note="synthetic scale churn",
            )
            for index, time_seconds in enumerate((4.4, 4.8, 6.4, 6.8))
        ]
        original_times = np.asarray([item.beat_time_seconds for item in events])
        reliability = ccb.build_reliability_segments(
            events,
            transitions,
            [],
            [(0.0, 12.0)],
            [],
        )

        self.assertTrue(
            any(
                item.classification == "BEAT_THIS_UNRELIABLE"
                and item.start_seconds <= 4.0
                and item.end_seconds >= 8.0
                for item in reliability
            )
        )
        np.testing.assert_array_equal(
            np.asarray([item.beat_time_seconds for item in events]), original_times
        )

    def test_p4_preserves_no_beat_as_a_separate_classification(self) -> None:
        events = []
        for activity_id, times in enumerate(
            (np.arange(0.0, 3.0, 0.3), np.arange(5.0, 8.01, 0.3))
        ):
            events.extend(
                ccb.PhaseGridEvent(
                    grid_beat_index=len(events) + 1,
                    activity_segment_id=activity_id,
                    beat_time_seconds=float(time_seconds),
                    selected_scale=1.0,
                    target_period_seconds=0.30,
                    predicted_time_seconds=float(time_seconds),
                    phase_residual_seconds=0.0,
                    beat_probability=0.90,
                    downbeat_probability=0.0,
                    event_source="acoustic_peak",
                    transition_type="stable",
                    path_cost=float(len(events)),
                )
                for time_seconds in times
            )
        reliability = ccb.build_reliability_segments(
            events,
            [],
            [],
            [(0.0, 3.0), (5.0, 8.0)],
            [(3.0, 5.0)],
            [ccb.ActivitySegment(1, 3.0, 5.0, True, "user", "silent bridge")],
        )
        no_beat = [item for item in reliability if item.classification == "NO_BEAT"]

        self.assertEqual(len(no_beat), 1)
        self.assertEqual((no_beat[0].start_seconds, no_beat[0].end_seconds), (3.0, 5.0))
        self.assertIn("silent bridge", no_beat[0].reason)


if __name__ == "__main__":
    unittest.main()
