"""Shared test fixtures."""

import pytest

from somnoscreen import config
from somnoscreen.data import load_dataset

DATA_AVAILABLE = (config.DATA_DIR / "questionnaire.csv").exists()
needs_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="NHANES data not present (it is gitignored)")


@pytest.fixture(scope="session")
def dataset():
    """The full decoded modelling table (only when the data files are present)."""
    if not DATA_AVAILABLE:
        pytest.skip("NHANES data not present")
    return load_dataset()
