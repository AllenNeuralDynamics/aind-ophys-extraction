"""Thin Code Ocean entry point for ophys ROI extraction.

All logic lives in the ``aind-ophys-extraction-library`` package;
this wrapper only parses settings (CLI / environment) and invokes ``run``.
"""

from aind_ophys_extraction_library.job import run

if __name__ == "__main__":
    run()
