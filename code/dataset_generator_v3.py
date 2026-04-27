"""
Legacy compatibility entrypoint.

`dataset_generator_v3.py` now delegates to `dataset_generator_v4_tracklevel.py`
so older commands keep working while the project uses the track-level v4 format.
"""

from dataset_generator_v4_tracklevel import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("dataset_generator_v4_tracklevel", run_name="__main__")
