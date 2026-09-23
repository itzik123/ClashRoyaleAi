"""Capture layer, against synthetic media generated on the fly. Ordering,
timestamps, decimation and the variable-frame-rate guard are properties of the
reader, and a synthetic file's ground truth is exact.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
import cv2  # noqa: E402

from capture.frames import FrameDirSource  # noqa: E402
from capture.source import FrameSource  # noqa: E402
from capture.video import VideoSource, VideoTimingError  # noqa: E402


def _write_video(path, frames=90, fps=30, size=(320, 180)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened(), "no mp4v encoder available in this OpenCV build"
    for i in range(frames):
        image = np.full((size[1], size[0], 3), i % 256, np.uint8)
        # A moving marker, so frame identity is checkable, not just frame
        # count.
        cv2.rectangle(image, (i % size[0], 10), (i % size[0] + 6, 40), (0, 0, 255), -1)
        writer.write(image)
    writer.release()
    return path


def test_video_reads_frames_in_order_with_timestamps(tmp_path):
    path = _write_video(tmp_path / "clip.mp4", frames=60, fps=30)
    with VideoSource(path) as source:
        assert source.size == (320, 180)
        frames = list(source)

    assert len(frames) == 60
    assert [f.index for f in frames] == list(range(60))
    stamps = [f.wall_time_ms for f in frames]
    assert stamps == sorted(stamps)
    # ~33.3 ms apart at 30 fps.
    assert np.allclose(np.diff(stamps), 1000 / 30, atol=2.0)


def test_cfr_probe_accepts_a_constant_rate_file(tmp_path):
    path = _write_video(tmp_path / "cfr.mp4", frames=90, fps=30)
    with VideoSource(path) as source:
        timing = source.probe_timing()
        assert timing["is_cfr"], timing["reason"]
        assert timing["measured_fps"] == pytest.approx(30.0, abs=0.5)
        assert source.fps == pytest.approx(30.0, abs=0.5)


def test_probe_does_not_disturb_the_read_position(tmp_path):
    path = _write_video(tmp_path / "clip.mp4", frames=40, fps=30)
    with VideoSource(path) as source:
        source.probe_timing()
        assert len(list(source)) == 40


def test_vfr_source_refuses_to_report_an_fps(tmp_path):
    """The guard that matters most for phone recordings: a variable-rate file
    reports a nominal fps and decodes fine, while every index-derived tick is
    wrong by an amount that grows when the screen is busy, i.e. when placements
    happen.
    """
    path = _write_video(tmp_path / "clip.mp4", frames=30, fps=30)
    source = VideoSource(path)
    # Force the probe's verdict: no cross-platform OpenCV writer reliably
    # produces a genuinely VFR file.
    source._timing = {
        "nominal_fps": 30.0, "sampled_frames": 30, "is_cfr": False,
        "measured_fps": float("nan"), "jitter": 0.42,
        "reason": "inter-frame deltas vary by 42.0% of the median",
    }
    with pytest.raises(VideoTimingError, match="variable"):
        _ = source.fps
    source.close()


def test_assume_fps_is_an_explicit_opt_out(tmp_path):
    path = _write_video(tmp_path / "clip.mp4", frames=30, fps=30)
    source = VideoSource(path, assume_fps=29.97)
    assert source.fps == pytest.approx(29.97)
    assert source.timing_override == 29.97
    source.close()


def test_sample_every_decimates_by_time(tmp_path):
    path = _write_video(tmp_path / "clip.mp4", frames=90, fps=30)
    with VideoSource(path) as source:
        sampled = list(source.sample_every(5.0))

    assert 14 <= len(sampled) <= 17, f"expected ~15 frames at 5fps from 3s, got {len(sampled)}"
    gaps = np.diff([f.wall_time_ms for f in sampled])
    assert np.all(gaps >= 180), "decimation must not emit bursts"


def test_frame_dir_source_reads_sorted(tmp_path):
    directory = tmp_path / "frames"
    directory.mkdir()
    for i in range(5):
        image = np.full((40, 60, 3), i * 40, np.uint8)
        cv2.imwrite(str(directory / f"frame_{i:04d}.png"), image)

    with FrameDirSource(directory, fps=10.0) as source:
        frames = list(source)
    assert [f.index for f in frames] == [0, 1, 2, 3, 4]
    assert frames[1].wall_time_ms == pytest.approx(100.0)
    assert frames[0].size == (60, 40)


def test_frame_dir_rejects_mixed_sizes(tmp_path):
    """One profile cannot cover two resolutions, so this must not be silent."""
    directory = tmp_path / "frames"
    directory.mkdir()
    cv2.imwrite(str(directory / "a.png"), np.zeros((40, 60, 3), np.uint8))
    cv2.imwrite(str(directory / "b.png"), np.zeros((50, 70, 3), np.uint8))

    with FrameDirSource(directory) as source:
        with pytest.raises(ValueError, match="calibration profile"):
            list(source)


def test_missing_inputs_raise_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoSource(tmp_path / "nope.mp4")
    with pytest.raises(NotADirectoryError):
        FrameDirSource(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no images"):
        FrameDirSource(tmp_path / "empty")


def test_window_capture_is_not_implemented_yet():
    """The `capture` package does not re-export WindowSource; it lives in
    capture.window, which needs the Windows capture API.
    """
    import capture
    assert not hasattr(capture, "WindowSource")
    assert issubclass(VideoSource, FrameSource)
