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

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fused.csv"
            repaired = root / "repaired.csv"
            mids, raw, smooth = bpm.local_tempo(result.beat_times)
            bpm.write_beats_csv(
                source, result, mids, raw, smooth, 22050, [(0.0, 6.0)]
            )
            bpm.write_diagnostic_repaired_beats_csv(source, repaired)
            loaded = bpm.read_beats_csv(repaired, "beat-this-repaired")

        np.testing.assert_array_equal(loaded.beat_times, beat_times)

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


if __name__ == "__main__":
    unittest.main()
