import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import API as api


class ApiTest(unittest.TestCase):
    def test_public_errors_keep_standard_exception_compatibility(self) -> None:
        self.assertTrue(issubclass(api.InvalidArgumentError, ValueError))
        self.assertTrue(issubclass(api.ResourceNotFoundError, FileNotFoundError))
        self.assertTrue(issubclass(api.EditConflictError, api.CCBError))
        self.assertTrue(issubclass(api.ItemNotFoundError, KeyError))
        with self.assertRaises(api.ResourceNotFoundError):
            api.run("definitely-missing-audio.wav")

    def setUp(self) -> None:
        api.set_music_gain(api.ccb.DEFAULT_MUSIC_GAIN)
        api.set_click_gain(api.ccb.DEFAULT_CLICK_GAIN)

    def tearDown(self) -> None:
        api.set_music_gain(api.ccb.DEFAULT_MUSIC_GAIN)
        api.set_click_gain(api.ccb.DEFAULT_CLICK_GAIN)

    def make_no_beat_fixture(self, root: Path) -> tuple[Path, Path]:
        audio = root / "song.mp3"
        audio.touch()
        output = root / "results"
        result_dir = output / "song"
        result_dir.mkdir(parents=True)
        (result_dir / "report.json").write_text(
            json.dumps(
                {
                    "audio": str(audio.resolve()),
                    "duration_seconds": 100.0,
                }
            ),
            encoding="utf-8",
        )
        (result_dir / "segments.csv").write_text(
            "segment_id,start_seconds,end_seconds,no_beat,source,note\n"
            "0,0.000000000,100.000000000,0,default,\n",
            encoding="utf-8-sig",
        )
        return audio, output

    def make_beat_fixture(self, root: Path) -> tuple[Path, Path]:
        audio, output = self.make_no_beat_fixture(root)
        result_dir = output / "song"
        report = json.loads((result_dir / "report.json").read_text(encoding="utf-8"))
        report["result"] = {"beat_count": 3, "downbeat_count": 1}
        report["manual_edits"] = {"added": [], "adjusted": [], "deleted": []}
        (result_dir / "report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        (result_dir / "overview.png").touch()
        header = (
            "beat_index,sample_index,beat_time_seconds,is_downbeat,"
            "activity_segment_id,is_no_beat,interval_midpoint_seconds,"
            "raw_local_bpm,smoothed_local_bpm,reliability_class,"
            "reliability_score,reliability_reason\n"
        )
        rows = [
            "1,22050,1.000000000,1,0,0,,,,RELIABLE,1.000000,automatic\n",
            "2,44100,2.000000000,0,0,0,1.500000000,60.0000,60.0000,RELIABLE,1.000000,automatic\n",
            "3,66150,3.000000000,0,0,0,2.500000000,60.0000,60.0000,RELIABLE,1.000000,automatic\n",
        ]
        (result_dir / "beats.csv").write_text(
            header + "".join(rows), encoding="utf-8-sig"
        )
        return audio, output

    def make_cache_fixture(
        self, root: Path, name: str, timestamp: float
    ) -> tuple[Path, Path]:
        audio = root / f"{name}.mp3"
        audio.touch()
        cache_root = root / "cache"
        cache = cache_root / name
        cache.mkdir(parents=True)
        api.ccb.write_inference_metadata_csv(
            cache / "inference.csv",
            api.ccb.InferenceMetadata(
                fps=50.0,
                window_seconds=30.0,
                hop_seconds=10.0,
                overlap_windows=1,
                sample_rate=22050,
                duration_seconds=10.0,
                audio_path=str(audio.resolve()),
            ),
        )
        (cache / "frames.csv").write_text("frames", encoding="utf-8")
        (cache / "fused_beats.csv").write_text("beats", encoding="utf-8")
        os.utime(cache, (timestamp, timestamp))
        return audio, cache_root

    def test_run_reuses_cache_and_returns_final_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "song.mp3"
            audio.touch()
            output = root / "results"

            def fake_analyse(audio_path, output_root, args, estimator):
                self.assertEqual(audio_path, audio)
                self.assertIsNone(estimator)
                self.assertTrue(args.no_click)
                self.assertEqual(args.music_gain, 0.1)
                self.assertEqual(args.click_gain, 0.9)
                self.assertTrue(args.preserve_manual_edits)
                result_dir = api.ccb.result_directory(audio_path, output_root)
                result_dir.mkdir(parents=True)
                for name in ("beats.csv", "overview.png", "segments.csv"):
                    (result_dir / name).touch()
                (result_dir / "report.json").write_text(
                    json.dumps(
                        {
                            "audio": str(audio_path.resolve()),
                            "result": {"dominant_bpm": 160.0},
                        }
                    ),
                    encoding="utf-8",
                )
                return []

            with (
                patch.object(
                    api.ccb, "inference_cache_available", return_value=True
                ),
                patch.object(api.ccb, "BeatThisEstimator") as estimator_class,
                patch.object(
                    api.ccb, "analyse_file", side_effect=fake_analyse
                ) as analyse,
            ):
                result = api.run(audio, output, no_click=True)

            estimator_class.assert_not_called()
            analyse.assert_called_once()
            self.assertEqual(result.audio, audio.resolve())
            result_dir = output / api.ccb.audio_storage_key(audio)
            self.assertEqual(result.output_directory, result_dir.resolve())
            self.assertEqual(result.beats_csv, (result_dir / "beats.csv").resolve())
            self.assertIsNone(result.click_wav)
            self.assertEqual(result.report["result"]["dominant_bpm"], 160.0)

    def test_gain_setters_validate_and_update_run_defaults(self) -> None:
        api.set_music_gain(0.25)
        api.set_click_gain(0.75)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "song.mp3"
            audio.touch()
            output = root / "results"

            def fake_analyse(audio_path, output_root, args, estimator):
                self.assertEqual(args.music_gain, 0.25)
                self.assertEqual(args.click_gain, 0.75)
                result_dir = api.ccb.result_directory(audio_path, output_root)
                result_dir.mkdir(parents=True)
                (result_dir / "report.json").write_text(
                    json.dumps(
                        {
                            "audio": str(audio_path.resolve()),
                            "result": {"dominant_bpm": 160.0},
                        }
                    ),
                    encoding="utf-8",
                )
                return []

            with (
                patch.object(
                    api.ccb, "inference_cache_available", return_value=True
                ),
                patch.object(api.ccb, "analyse_file", side_effect=fake_analyse),
            ):
                api.run(audio, output, no_click=True)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            api.set_music_gain(-0.1)
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            api.set_click_gain(float("nan"))
        api.set_music_gain(0.0)
        with self.assertRaisesRegex(ValueError, "cannot both be zero"):
            api.set_click_gain(0.0)

    def test_no_beat_crud_renumbers_ranges_by_time(self) -> None:
        with TemporaryDirectory() as directory:
            audio, output = self.make_no_beat_fixture(Path(directory))

            later = api.create_no_beat_range(
                audio, 30.0, 40.0, output_dir=output, note="later"
            )
            earlier = api.create_no_beat_range(
                audio, 10.0, 20.0, output_dir=output, note="earlier"
            )
            self.assertEqual(later.segment_id, 1)
            self.assertEqual(earlier.segment_id, 1)
            self.assertEqual(
                [(item.segment_id, item.start_seconds) for item in api.list_no_beat_ranges(audio, output_dir=output)],
                [(1, 10.0), (2, 30.0)],
            )

            moved = api.update_no_beat_range(
                audio,
                2,
                start_seconds=5.0,
                end_seconds=8.0,
                output_dir=output,
            )
            self.assertEqual((moved.segment_id, moved.start_seconds), (1, 5.0))
            self.assertEqual(
                [(item.segment_id, item.start_seconds) for item in api.list_no_beat_ranges(audio, output_dir=output)],
                [(1, 5.0), (2, 10.0)],
            )

            api.delete_no_beat_range(audio, 2, output_dir=output)
            remaining = api.list_no_beat_ranges(audio, output_dir=output)
            self.assertEqual([(item.segment_id, item.start_seconds) for item in remaining], [(1, 5.0)])
            self.assertEqual(api.clear_no_beat_ranges(audio, output_dir=output), 1)
            self.assertEqual(api.list_no_beat_ranges(audio, output_dir=output), [])

            rows = (output / "song" / "segments.csv").read_text(
                encoding="utf-8-sig"
            ).splitlines()
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[1].startswith("0,0.000000000,100.000000000,0"))

    def test_no_beat_management_rejects_invalid_or_protected_changes(self) -> None:
        with TemporaryDirectory() as directory:
            audio, output = self.make_no_beat_fixture(Path(directory))
            api.create_no_beat_range(audio, 10.0, 20.0, output_dir=output)

            with self.assertRaisesRegex(ValueError, "overlaps segment"):
                api.create_no_beat_range(audio, 15.0, 25.0, output_dir=output)
            with self.assertRaisesRegex(ValueError, "0 <= start < end"):
                api.update_no_beat_range(
                    audio, 1, end_seconds=101.0, output_dir=output
                )
            with self.assertRaisesRegex(KeyError, "does not exist"):
                api.delete_no_beat_range(audio, 99, output_dir=output)

            segments_path = output / "song" / "segments.csv"
            with segments_path.open("a", encoding="utf-8") as handle:
                handle.write("2,30.000000000,40.000000000,1,model,protected\n")
            with self.assertRaisesRegex(PermissionError, "not user-managed"):
                api.delete_no_beat_range(audio, 2, output_dir=output)

    def test_run_rejects_a_missing_audio_file(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Audio file does not exist"):
            api.run("missing-song.mp3")

    def test_beat_crud_renumbers_and_persists_operations(self) -> None:
        with TemporaryDirectory() as directory:
            audio, output = self.make_beat_fixture(Path(directory))
            added = api.create_beat(
                audio, 1.5, is_downbeat=False, output_dir=output
            )
            self.assertEqual((added.beat_id, added.reliability_class), (2, "MANUAL_EDIT"))

            moved = api.update_beat(
                audio, 4, time_seconds=0.5, is_downbeat=True, output_dir=output
            )
            self.assertEqual((moved.beat_id, moved.time_seconds), (1, 0.5))
            api.delete_beat(audio, 4, output_dir=output)

            beats = api.list_beats(audio, output_dir=output)
            self.assertEqual(
                [(item.beat_id, item.time_seconds) for item in beats],
                [(1, 0.5), (2, 1.0), (3, 1.5)],
            )
            report = json.loads(
                (output / "song" / "report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(report["manual_edits"]["added"]), 1)
            self.assertEqual(len(report["manual_edits"]["adjusted"]), 1)
            self.assertEqual(len(report["manual_edits"]["deleted"]), 1)

    def test_beat_create_and_update_reject_time_collisions(self) -> None:
        with TemporaryDirectory() as directory:
            audio, output = self.make_beat_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "overlaps existing beat"):
                api.create_beat(audio, 2.0, output_dir=output)
            with self.assertRaisesRegex(ValueError, "overlaps existing beat"):
                api.update_beat(audio, 1, time_seconds=2.0, output_dir=output)

    def test_manual_edits_apply_to_a_fresh_automatic_grid(self) -> None:
        automatic = api.ccb.BeatResult(
            "automatic",
            api.ccb.np.asarray([1.0, 2.0, 3.0]),
            downbeat_times=api.ccb.np.asarray([1.0]),
        )
        edits = {
            "added": [{"time_seconds": 1.5, "is_downbeat": False}],
            "adjusted": [{
                "original_time_seconds": 3.0,
                "new_time_seconds": 3.25,
                "is_downbeat": True,
            }],
            "deleted": [{"original_time_seconds": 2.0}],
        }
        result, reasons, warnings = api.ccb.apply_manual_beat_edits(
            automatic, edits, [(0.0, 4.0)]
        )
        self.assertEqual(result.beat_times.tolist(), [1.0, 1.5, 3.25])
        self.assertEqual(result.downbeat_times.tolist(), [1.0, 3.25])
        self.assertEqual(reasons, {1.5: "manual_added", 3.25: "manual_adjusted"})
        self.assertEqual(warnings, [])

    def test_read_only_result_inspection_and_validation(self) -> None:
        with TemporaryDirectory() as directory:
            audio, output = self.make_beat_fixture(Path(directory))
            report_path = output / "song" / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["reliability"] = {
                "recommended_review_ranges": [
                    {
                        "start_seconds": 1.0,
                        "end_seconds": 2.0,
                        "classification": "BEAT_THIS_UNRELIABLE",
                        "reliability_score": 0.25,
                        "reason": "test range",
                    }
                ]
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")

            result = api.get_result(audio, output_dir=output)
            self.assertEqual(result.report_json, report_path.resolve())
            info = api.inspect_song(audio, output_dir=output)
            self.assertEqual(info.result_status, "READY")
            self.assertEqual(info.cache_status, "MISSING")
            self.assertEqual((info.beat_count, info.downbeat_count), (3, 1))

            ranges = api.get_review_ranges(
                audio,
                output_dir=output,
                classifications=["BEAT_THIS_UNRELIABLE"],
            )
            self.assertEqual(len(ranges), 1)
            self.assertEqual(ranges[0].reason, "test range")
            validation = api.validate_result(audio, output_dir=output)
            self.assertTrue(validation.valid, validation.errors)
            self.assertIn("inference cache is missing", validation.warnings)

    def test_read_only_beat_filters_and_manual_edit_details(self) -> None:
        with TemporaryDirectory() as directory:
            audio, output = self.make_beat_fixture(Path(directory))
            api.create_beat(audio, 1.5, output_dir=output, note="inserted")
            api.update_beat(
                audio, 4, time_seconds=3.5, output_dir=output, note="moved"
            )
            manual = api.list_beats(
                audio,
                output_dir=output,
                start_seconds=1.25,
                end_seconds=3.6,
                manual_only=True,
            )
            self.assertEqual([item.time_seconds for item in manual], [1.5, 3.5])
            adjusted = api.list_beats(
                audio,
                output_dir=output,
                reliability_class="MANUAL_EDIT",
            )
            self.assertEqual(len(adjusted), 2)
            edits = api.get_manual_beat_edits(audio, output_dir=output)
            self.assertEqual(edits.added[0].note, "inserted")
            self.assertEqual(edits.adjusted[0].note, "moved")

    def test_cache_listing_and_pruning_support_latest_and_file_keep(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            older_audio, cache_root = self.make_cache_fixture(
                root, "older", 1_700_000_000.0
            )
            newer_audio, _ = self.make_cache_fixture(
                root, "newer", 1_700_000_100.0
            )

            entries = api.list_caches(cache_dir=cache_root)
            self.assertEqual([item.song_name for item in entries], ["newer", "older"])
            self.assertTrue(all(item.status == "READY" for item in entries))

            preview = api.prune_caches(
                keep=1, cache_dir=cache_root, dry_run=True
            )
            self.assertEqual(preview.deleted_count, 1)
            self.assertEqual(preview.kept_count, 1)
            self.assertTrue((cache_root / "older").is_dir())

            result = api.prune_caches(
                keep=[older_audio], cache_dir=cache_root
            )
            self.assertEqual([item.song_name for item in result.deleted], ["newer"])
            self.assertTrue((cache_root / "older").is_dir())
            self.assertFalse((cache_root / "newer").exists())
            self.assertEqual(
                api.list_caches(cache_dir=cache_root)[0].audio_path,
                older_audio.resolve(),
            )
            cleared = api.prune_caches(cache_dir=cache_root)
            self.assertEqual(cleared.deleted_count, 1)
            self.assertEqual(api.list_caches(cache_dir=cache_root), [])

    def test_same_named_audio_uses_distinct_result_and_cache_keys(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first" / "song.mp3"
            second = root / "second" / "song.mp3"
            first.parent.mkdir()
            second.parent.mkdir()
            first.touch()
            second.touch()
            output = root / "results"
            cache = root / "cache"

            self.assertNotEqual(
                api.ccb.audio_storage_key(first),
                api.ccb.audio_storage_key(second),
            )
            self.assertNotEqual(
                api.ccb.result_directory(first, output),
                api.ccb.result_directory(second, output),
            )
            self.assertNotEqual(
                api.ccb.inference_cache_paths(first, output, cache)["root"],
                api.ccb.inference_cache_paths(second, output, cache)["root"],
            )

            legacy_result = output / "song"
            legacy_result.mkdir(parents=True)
            (legacy_result / "report.json").write_text(
                json.dumps({"audio": str(first.resolve())}), encoding="utf-8"
            )
            self.assertEqual(
                api.ccb.result_directory(first, output), legacy_result
            )
            self.assertNotEqual(
                api.ccb.result_directory(second, output), legacy_result
            )

            legacy_cache = cache / "song"
            legacy_cache.mkdir(parents=True)
            api.ccb.write_inference_metadata_csv(
                legacy_cache / "inference.csv",
                api.ccb.InferenceMetadata(
                    fps=50.0,
                    window_seconds=30.0,
                    hop_seconds=10.0,
                    overlap_windows=1,
                    sample_rate=22050,
                    duration_seconds=1.0,
                    audio_path=str(first.resolve()),
                ),
            )
            self.assertEqual(
                api.ccb.inference_cache_paths(first, output, cache)["root"],
                legacy_cache,
            )
            self.assertNotEqual(
                api.ccb.inference_cache_paths(second, output, cache)["root"],
                legacy_cache,
            )


if __name__ == "__main__":
    unittest.main()
