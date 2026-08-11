"""Tests for txt_metadata_parser.py."""

import os
import tempfile
import pytest
from rixs_app.core.txt_metadata_parser import parse_scan_log, validate_tif_coverage


SAMPLE_HEADER = """Date: 6/28/2026\r
\r
Start, Stop, Increment\r
SM3 Mirror Pitch\r
Start: 98.00000000\r
Stop: 116.00000000\r
Increment: 1.00000000\r
Delay After Move (s): 0.00000000\r
Count Time (s): 0.50000000\r
Scan Number: 1\r
Bi-directional: No\r
Stay at End: 0\r
Description Length: 7\r
descrip\r
"""

SAMPLE_COL_HEADER = "Time of Day\tTime (s)\tSM3 Mirror Pitch Goal\tSM3 Mirror Pitch Actual\tMarana Counts\tBeam Current\tImage - Stdev\tImage - Center\tImage - Amplitude\tImage - FWHM\tPicoScale CH1\tPicoScale CH2\tPicoScale CH3\tA3200 X Current\tA3200 Y Current\tA3200 Z Current\tA3200 THETA Current\tA3200 X Position\tA3200 Y Position\tA3200 Z Position\tA3200 THETA Position\tLakeshore 218 - 1\tLakeshore 218 - 2\tLakeshore 218 - 3\tDetector Temperature\tMarana Temperature\tDIAG 202 Upper\tDIAG 202 Left\tDIAG 202 Lower\tDIAG 202 Right\tSLIT 201 Left\tSLIT 201 Right\tDIAG 204\tSample\tBPM Right\tBPM Left\tBPM horiz control\tDIODE Wide\tDIODE Narrow\tTemp-Ch1 Blue Frame\tTemp-Ch2  M204 frame\tTemp-Ch3 M204 mirror\tTemp-Ch4 Air Temp\tTemp-Ch5 chiller water\tTemp-Ch6 M203 box\tMarana\r\n"

def make_data_row(goal, idx):
    """Build a minimal tab-separated data row with the correct number of columns."""
    parts = ["17:45:23", "0.9010", str(goal), str(goal + 0.03)]
    # Pad with dummy zeros up to column 44
    parts += ["0.0"] * 41
    # Column 45 = Marana file path
    parts.append(f"..\\Single Motor Scan 004202 Images\\Single Motor Scan 004202 Marana {idx:03d}.tiff")
    return "\t".join(parts) + "\r\n"


def write_sample_txt(path, num_rows=3):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(SAMPLE_HEADER)
        f.write(SAMPLE_COL_HEADER)
        for i in range(num_rows):
            f.write(make_data_row(98.0 + i, i))


# ----- Tests -----

def test_parse_scan_log_basic(tmp_path):
    txt = tmp_path / "scan.txt"
    write_sample_txt(str(txt), num_rows=3)
    
    meta = parse_scan_log(str(txt))
    
    assert meta["motor_name"] == "SM3 Mirror Pitch"
    assert abs(meta["start"] - 98.0) < 1e-6
    assert abs(meta["stop"] - 116.0) < 1e-6
    assert abs(meta["increment"] - 1.0) < 1e-6
    assert len(meta["frames"]) == 3


def test_parse_scan_log_frame_contents(tmp_path):
    txt = tmp_path / "scan.txt"
    write_sample_txt(str(txt), num_rows=3)
    
    meta = parse_scan_log(str(txt))
    frames = meta["frames"]
    
    # Check that Windows-style basename was correctly extracted
    assert "Single Motor Scan 004202 Marana 000.tiff" in frames
    assert "Single Motor Scan 004202 Marana 001.tiff" in frames
    assert "Single Motor Scan 004202 Marana 002.tiff" in frames
    
    frame0 = frames["Single Motor Scan 004202 Marana 000.tiff"]
    assert abs(frame0["motor_goal"] - 98.0) < 1e-4
    assert frame0["row_index"] == 0


def test_parse_scan_log_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_scan_log("/non_existent_path/scan.txt")


def test_parse_scan_log_too_short(tmp_path):
    txt = tmp_path / "bad_scan.txt"
    txt.write_text("only one line\n")
    with pytest.raises(ValueError, match="at least"):
        parse_scan_log(str(txt))


def test_validate_tif_coverage_all_matched(tmp_path):
    txt = tmp_path / "scan.txt"
    write_sample_txt(str(txt), num_rows=3)
    meta = parse_scan_log(str(txt))
    
    file_list = [
        "/some/dir/Single Motor Scan 004202 Marana 000.tiff",
        "/some/dir/Single Motor Scan 004202 Marana 001.tiff",
        "/some/dir/Single Motor Scan 004202 Marana 002.tiff",
    ]
    matched, unmatched = validate_tif_coverage(file_list, meta)
    
    assert len(matched) == 3
    assert len(unmatched) == 0


def test_validate_tif_coverage_partial(tmp_path):
    txt = tmp_path / "scan.txt"
    write_sample_txt(str(txt), num_rows=3)
    meta = parse_scan_log(str(txt))
    
    file_list = [
        "/some/dir/Single Motor Scan 004202 Marana 000.tiff",
        "/some/dir/Single Motor Scan 004202 Marana_EXTRA_999.tiff",
    ]
    matched, unmatched = validate_tif_coverage(file_list, meta)
    
    assert len(matched) == 1
    assert len(unmatched) == 1
    assert "/some/dir/Single Motor Scan 004202 Marana_EXTRA_999.tiff" in unmatched


def test_validate_tif_coverage_empty_file_list(tmp_path):
    txt = tmp_path / "scan.txt"
    write_sample_txt(str(txt), num_rows=3)
    meta = parse_scan_log(str(txt))
    
    matched, unmatched = validate_tif_coverage([], meta)
    assert matched == []
    assert unmatched == []


def test_parse_scan_log_real_file():
    """Integration test against the real scan log on disk (skipped if not present)."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    real_txt = os.path.join(
        project_root,
        "RIXS_ZeroOrderScan",
        "Single Motor Scan 004202 Images",
        "Single Motor Scan 004202.txt",
    )
    if not os.path.exists(real_txt):
        pytest.skip("Real scan log not present on this machine")
    
    meta = parse_scan_log(real_txt)
    assert meta["motor_name"] == "SM3 Mirror Pitch"
    assert len(meta["frames"]) > 0
    
    # All frames should have a non-NaN motor_goal
    for basename, info in meta["frames"].items():
        import math
        assert not math.isnan(info["motor_goal"]), f"NaN motor_goal for {basename}"
