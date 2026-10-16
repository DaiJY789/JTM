from __future__ import annotations

import numpy as np
from PIL import Image

import app.agents.tools.Forensics as forensics


SCHEMA_NAMES = [
    "BAYAR_FILTER_SCHEMA",
    "DCT_HIGHPASS_FILTER_SCHEMA",
    "DWT_HH_MAP_SCHEMA",
    "FFT_HIGHPASS_RESIDUAL_SCHEMA",
    "HIGHPASS_FILTER_SCHEMA",
    "JPEG_GHOST_MAP_SCHEMA",
    "LOCAL_CORRELATION_MAP_SCHEMA",
    "MEDIAN_FILTER_TRACE_SCORE_SCHEMA",
    "PRNU_RESIDUAL_SCHEMA",
    "RESAMPLING_PERIODICITY_SCORE_SCHEMA",
    "SOBEL_FILTER_SCHEMA",
    "SRM_RESIDUAL_BANK_SCHEMA",
]


def _sample_image() -> Image.Image:
    rng = np.random.default_rng(20260919)
    array = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def _assert_finite_array(value: object) -> None:
    array = np.asarray(value)
    assert array.size > 0
    assert np.issubdtype(array.dtype, np.number)
    assert np.isfinite(array).all()


def test_forenagent_exports_twelve_unique_tool_schemas() -> None:
    schemas = [getattr(forensics, name) for name in SCHEMA_NAMES]
    tool_names = [schema["name"] for schema in schemas]

    assert len(tool_names) == 12
    assert len(set(tool_names)) == 12


def test_forenagent_tools_smoke_with_pil_image() -> None:
    image = _sample_image()

    array_outputs = [
        forensics.bayar_filter(image, 5),
        forensics.srm_residual_bank(image, 0),
        forensics.fft_highpass_residual(image),
        forensics.dwt_hh_map(image, 1),
        forensics.local_correlation_map(image, 3),
        forensics.prnu_residual(image),
        forensics.dct_highpass_filter(image),
        forensics.sobel_filter(image),
        forensics.highpass_filter(image),
    ]
    ghost_maps = forensics.jpeg_ghost_map(image, [75, 85, 95])
    scalar_outputs = [
        forensics.median_filter_trace_score(image),
        forensics.resampling_periodicity_score(image),
    ]

    for output in array_outputs:
        _assert_finite_array(output)
    assert set(ghost_maps) == {75, 85, 95}
    for output in ghost_maps.values():
        _assert_finite_array(output)
    for output in scalar_outputs:
        assert np.isfinite(output)
