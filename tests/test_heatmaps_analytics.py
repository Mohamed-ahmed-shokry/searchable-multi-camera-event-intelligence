from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from generate_heatmaps_analytics import (
    LINE_COUNTS_CSV_HEADER,
    ZONE_COUNTS_CSV_HEADER,
    AnalyticsConfig,
    LineCountRecord,
    TrackDurationStats,
    TrajectoryPoint,
    ZoneCountRecord,
    accumulate_trajectory_grid,
    load_analytics_config,
    read_line_counts_csv,
    read_zone_counts_csv,
    save_analytics_config,
    smooth_and_normalize_heatmap,
    write_line_counts_csv,
    write_zone_counts_csv,
)


def test_analytics_config_defaults_and_validation() -> None:
    cfg = AnalyticsConfig()
    assert cfg.grid_bin_size_px == 16
    assert cfg.gaussian_sigma_px == pytest.approx(24.0)
    assert cfg.normalization_mode == "log"
    assert cfg.percentile_clip == pytest.approx(99.0)
    assert cfg.max_points_per_track_cell == 6
    assert cfg.heatmap_alpha == pytest.approx(0.55)
    assert cfg.colormap == "jet"
    assert cfg.overlay_scene_geometry is True
    assert cfg.reference_frame_index == 35

    with pytest.raises(ValueError, match="grid_bin_size_px must be >= 1"):
        AnalyticsConfig(grid_bin_size_px=0)

    with pytest.raises(ValueError, match="gaussian_sigma_px must be >= 0"):
        AnalyticsConfig(gaussian_sigma_px=-1.0)

    with pytest.raises(ValueError, match="normalization_mode must be one of"):
        AnalyticsConfig(normalization_mode="invalid")

    with pytest.raises(ValueError, match="percentile_clip must be in"):
        AnalyticsConfig(percentile_clip=0.0)

    with pytest.raises(ValueError, match="max_points_per_track_cell must be >= 0"):
        AnalyticsConfig(max_points_per_track_cell=-1)

    with pytest.raises(ValueError, match="heatmap_alpha must be in"):
        AnalyticsConfig(heatmap_alpha=1.5)

    with pytest.raises(ValueError, match="colormap must be one of"):
        AnalyticsConfig(colormap="rainbow")

    with pytest.raises(ValueError, match="reference_frame_index must be >= 0"):
        AnalyticsConfig(reference_frame_index=-2)

    with pytest.raises(
        ValueError, match="duration_bucket_edges_seconds must have at least 2 edges"
    ):
        AnalyticsConfig(duration_bucket_edges_seconds=(0.0,))


def test_zone_and_line_count_record_validation() -> None:
    assert ZONE_COUNTS_CSV_HEADER == [
        "zone_id",
        "entry_count",
        "exit_count",
        "total_dwell_seconds",
        "peak_occupancy",
    ]
    assert LINE_COUNTS_CSV_HEADER == [
        "line_id",
        "crossings_ab",
        "crossings_ba",
        "total",
    ]

    zrec = ZoneCountRecord(
        zone_id="entrance_area",
        entry_count=5,
        exit_count=4,
        total_dwell_seconds=8.25,
        peak_occupancy=2,
    )
    assert zrec.to_row() == ["entrance_area", "5", "4", "8.2500", "2"]

    with pytest.raises(ValueError, match="zone_id must be a non-empty string"):
        ZoneCountRecord(
            zone_id="",
            entry_count=1,
            exit_count=1,
            total_dwell_seconds=1.0,
            peak_occupancy=1,
        )

    lrec = LineCountRecord(
        line_id="doorway_line",
        crossings_ab=2,
        crossings_ba=3,
        total=5,
    )
    assert lrec.to_row() == ["doorway_line", "2", "3", "5"]

    with pytest.raises(ValueError, match="total \\(4\\) must equal"):
        LineCountRecord(
            line_id="doorway_line",
            crossings_ab=2,
            crossings_ba=3,
            total=4,
        )

    stats = TrackDurationStats(
        track_count=3,
        min_duration_seconds=1.5,
        max_duration_seconds=5.0,
        mean_duration_seconds=3.0,
        median_duration_seconds=2.5,
        p90_duration_seconds=4.5,
        mean_path_length_px=120.456,
        max_path_length_px=210.789,
        duration_buckets={"0.0-2.0s": 1, "2.0-4.0s": 1, "4.0-6.0s": 1},
    )
    d = stats.to_dict()
    assert d["track_count"] == 3
    assert d["mean_path_length_px"] == pytest.approx(120.46)


def test_analytics_config_yaml_round_trip(tmp_path: Path) -> None:
    cfg = AnalyticsConfig(
        grid_bin_size_px=8,
        gaussian_sigma_px=18.0,
        normalization_mode="percentile",
        percentile_clip=95.0,
        max_points_per_track_cell=4,
        heatmap_alpha=0.60,
        colormap="inferno",
        overlay_scene_geometry=False,
        reference_frame_index=10,
        duration_bucket_edges_seconds=(0.0, 1.0, 3.0, 5.0),
    )
    yaml_path = tmp_path / "analytics.yaml"
    save_analytics_config(cfg, yaml_path)
    loaded = load_analytics_config(yaml_path)
    assert loaded == cfg

    default_yaml = (
        Path(__file__).resolve().parents[1]
        / "cv_projects"
        / "08_heatmaps_analytics"
        / "configs"
        / "analytics.yaml"
    )
    assert load_analytics_config(default_yaml) == AnalyticsConfig()

    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    assert load_analytics_config(empty_yaml) == AnalyticsConfig()

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Expected YAML mapping"):
        load_analytics_config(bad_yaml)

    with pytest.raises(FileNotFoundError):
        load_analytics_config(tmp_path / "missing.yaml")


def test_zone_counts_csv_round_trip(tmp_path: Path) -> None:
    records = [
        ZoneCountRecord(
            zone_id="entrance_area",
            entry_count=6,
            exit_count=6,
            total_dwell_seconds=10.5833,
            peak_occupancy=2,
        ),
        ZoneCountRecord(
            zone_id="center_concourse",
            entry_count=6,
            exit_count=6,
            total_dwell_seconds=8.0833,
            peak_occupancy=2,
        ),
    ]
    csv_path = tmp_path / "zone_counts.csv"
    write_zone_counts_csv(records, csv_path)
    loaded = read_zone_counts_csv(csv_path)
    assert loaded == records

    bad_csv = tmp_path / "bad_zone.csv"
    bad_csv.write_text("zone_id,entry_count\nz1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected zone_counts.csv header"):
        read_zone_counts_csv(bad_csv)

    with pytest.raises(FileNotFoundError):
        read_zone_counts_csv(tmp_path / "missing_zone_counts.csv")


def test_line_counts_csv_round_trip(tmp_path: Path) -> None:
    records = [
        LineCountRecord(
            line_id="doorway_line",
            crossings_ab=0,
            crossings_ba=6,
            total=6,
        ),
        LineCountRecord(
            line_id="aisle_split_line",
            crossings_ab=1,
            crossings_ba=2,
            total=3,
        ),
    ]
    csv_path = tmp_path / "line_counts.csv"
    write_line_counts_csv(records, csv_path)
    loaded = read_line_counts_csv(csv_path)
    assert loaded == records

    bad_csv = tmp_path / "bad_line.csv"
    bad_csv.write_text("line_id,total\nl1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected line_counts.csv header"):
        read_line_counts_csv(bad_csv)

    with pytest.raises(FileNotFoundError):
        read_line_counts_csv(tmp_path / "missing_line_counts.csv")


def test_accumulate_trajectory_grid_and_smooth_normalize() -> None:
    # 10 stationary points for track 1 in cell (1, 2) + 2 points for track 2
    points = [
        TrajectoryPoint(track_id=1, frame_id=i, timestamp=i * 0.1, x=35.0, y=20.0)
        for i in range(10)
    ] + [
        TrajectoryPoint(track_id=2, frame_id=0, timestamp=0.0, x=35.0, y=20.0),
        TrajectoryPoint(track_id=2, frame_id=1, timestamp=0.1, x=36.0, y=21.0),
    ]

    raw_grid, full_res = accumulate_trajectory_grid(
        points,
        frame_width=160,
        frame_height=96,
        grid_bin_size_px=16,
        max_points_per_track_cell=4,
    )
    assert raw_grid.shape == (6, 10)
    assert full_res.shape == (96, 160)
    # Track 1 capped at 4 + Track 2 contributes 2 = 6.0 in cell (row=1, col=2)
    assert raw_grid[1, 2] == pytest.approx(6.0)

    for mode in ("log", "percentile", "linear"):
        norm = smooth_and_normalize_heatmap(
            full_res,
            gaussian_sigma_px=8.0,
            normalization_mode=mode,
            percentile_clip=95.0,
        )
        assert norm.shape == (96, 160)
        assert float(norm.min()) >= 0.0
        assert float(norm.max()) == pytest.approx(1.0)

    empty_norm = smooth_and_normalize_heatmap(
        np.zeros((32, 32), dtype=np.float32),
        gaussian_sigma_px=0.0,
        normalization_mode="linear",
    )
    assert float(empty_norm.max()) == 0.0
