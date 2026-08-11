import re
import pytest
import time
import os
from zeroth_order_cli import extract_frame_index

def test_user_cases():
    """Verify the exact filenames specified by the user."""
    # 2026-06-25_Scan_3848_CMOS_Detector_005_temp.tif -> 5
    assert extract_frame_index("2026-06-25_Scan_3848_CMOS_Detector_005_temp.tif") == 5
    
    # frame_-10.tiff -> -10
    assert extract_frame_index("frame_-10.tiff") == -10
    
    # 123_456_789.tif -> 789
    assert extract_frame_index("123_456_789.tif") == 789


def test_standard_cases():
    """Verify baseline functionality as verified in previous iterations."""
    assert extract_frame_index("frame_001.tif") == 1
    assert extract_frame_index("frame_-001.tif") == -1
    assert extract_frame_index("CMOS Detector 123.tif") == 123
    assert extract_frame_index("123.tif") == 123
    assert extract_frame_index("-123.tif") == -123
    assert extract_frame_index("123_456.tif") == 456
    assert extract_frame_index("image_123.tif") == 123
    assert extract_frame_index("20260625_frame_12.tif") == 12


def test_adversarial_mismatches():
    """
    Stress-test the parser logic for logical inconsistencies.
    Particularly, check how underscore vs space representation of "CMOS Detector"
    results in different parsed indices when trailing digits are present.
    """
    # Spaced CMOS Detector with trailing numbers -> matches step 1 pattern, returns 5
    assert extract_frame_index("CMOS Detector 005_temp_123.tif") == 5

    # Underscored CMOS Detector with trailing numbers -> fails step 1 pattern (due to underscore),
    # falls back to step 2 findall('-?\\d+') and returns 123!
    val = extract_frame_index("CMOS_Detector_005_temp_123.tif")
    print(f"CMOS_Detector_005_temp_123.tif parsed as: {val}")
    assert val == 5, "Vulnerability: Underscored CMOS Detector falls back to last digit block"


def test_more_adversarial_cases():
    """Verify additional complex and inconsistent edge cases."""
    # 1. Hyphenated frame prefix: frame-005 vs frame_005
    # frame_005_temp_123.tif -> matches step 1, returns 5
    assert extract_frame_index("frame_005_temp_123.tif") == 5
    # frame-005_temp_123.tif -> fails step 1, falls back to step 2, returns 123
    assert extract_frame_index("frame-005_temp_123.tif") == 5

    # 2. Plus signs in frame index
    # Step 1 pattern `frame_(-?\d+)` does not match `+`
    # frame_+123.tif -> fails step 1, falls back to step 2 which finds '123'
    assert extract_frame_index("frame_+123.tif") == 123

    # 3. Double hyphens
    # frame_--123.tif -> step 1 matches `-123` (since -? matches one -)
    assert extract_frame_index("frame_--123.tif") == -123

    # 4. Multiple occurrences of the same prefixes
    # frame_123_frame_456.tif -> step 1 matches first occurrence
    assert extract_frame_index("frame_123_frame_456.tif") == 123
    assert extract_frame_index("CMOS Detector 123 CMOS Detector 456.tif") == 123

    # 5. Suffix numbers under standard fallback
    # detector_005_run_01.tif -> fails step 1, step 2 finds ['005', '01'], returns 1
    assert extract_frame_index("detector_005_run_01.tif") == 1


def test_no_digits():
    """Verify filenames with no digits raise ValueError."""
    with pytest.raises(ValueError):
        extract_frame_index("frame_abc.tif")
    with pytest.raises(ValueError):
        extract_frame_index("CMOS Detector.tif")
    with pytest.raises(ValueError):
        extract_frame_index("")


def test_multiple_spaces():
    """Verify that multiple spaces or tabs between CMOS and Detector still match."""
    assert extract_frame_index("CMOS\tDetector\t045.tif") == 45
    assert extract_frame_index("CMOS   Detector   -99.tif") == -99


def test_performance_benchmarking():
    """Benchmark execution time of extract_frame_index over a large dataset of filenames."""
    # Generate 100,000 filenames
    filenames = []
    base_names = [
        "2026-06-25_Scan_3848_CMOS_Detector_{:03d}_temp.tif",
        "frame_-{:d}.tiff",
        "123_456_{:d}.tif",
        "CMOS Detector {:d}.tif",
        "image_{:d}.tif"
    ]
    for i in range(100000):
        template = base_names[i % len(base_names)]
        filenames.append(template.format(i))
        
    start_time = time.time()
    for filename in filenames:
        _ = extract_frame_index(filename)
    duration = time.time() - start_time
    
    avg_time_us = (duration / len(filenames)) * 1000000.0
    print(f"\n[Performance] Parsed {len(filenames)} filenames in {duration:.4f} seconds.")
    print(f"[Performance] Average time per filename: {avg_time_us:.2f} microseconds.")
    
    assert avg_time_us < 50.0
