from __future__ import annotations

from types import SimpleNamespace

import pytest

from vision_platform.student.experiment_gateway import StudentExperimentGateway


def test_v1_09_final_evidence_is_same_run_bound_and_read_only() -> None:
    gateway = object.__new__(StudentExperimentGateway)
    gateway._defect_evidence = None
    with pytest.raises(Exception, match="DEFECT_SORT"):
        gateway.record_defect_final(final_probe=None, primary_error=None)
