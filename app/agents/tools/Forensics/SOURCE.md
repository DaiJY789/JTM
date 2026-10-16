# ForenAgent source record

- Upstream: <https://github.com/zfr00/ForenAgent>
- Synced commit: `83815b5a8644407d5367e38c57d2248581bf8e89`
- Commit date: 2026-02-06
- Synced on: 2026-09-19
- Upstream location: `low_level_tool/*.py`

The twelve upstream Python tool files and `UPSTREAM_README.md` are copied without
functional changes. Demo images, generated comparison images, and the upstream Git
metadata are intentionally excluded. This directory adds only `__init__.py` for package
imports and this source record.

## Licensing notice

The upstream repository did not contain a `LICENSE`, `COPYING`, or `NOTICE` file at the
synced commit. Treat the vendored source as research/evaluation-only until the upstream
author provides explicit reuse and redistribution terms. Do not assume that the absence
of a license grants redistribution or commercial-use rights.

## Runtime dependency

`dwt_hh_map.py` and `prnu_residual.py` import `pywt`, provided by the `PyWavelets`
package. The project pins a Python 3.10-compatible release in `requirements.txt`.
