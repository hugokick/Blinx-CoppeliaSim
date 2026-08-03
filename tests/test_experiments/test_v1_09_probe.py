from __future__ import annotations

from types import SimpleNamespace

import pytest

from vision_platform.experiments.probes import probe_defect_entry_pre, probe_defect_entry_post, probe_defect_final


def test_v1_09_probe_exports_are_separate_from_ocr_contracts() -> None:
    assert callable(probe_defect_entry_pre)
    assert callable(probe_defect_entry_post)
    assert callable(probe_defect_final)


def test_v1_09_entry_probe_rejects_swapped_part_even_when_slot_is_occupied() -> None:
    sim = SimpleNamespace()
    with pytest.raises(ValueError, match="DEFECT_SORT"):
        probe_defect_entry_post(sim, SimpleNamespace(), entry_id="entry_a", run_id="run-1", frame_id="frame-1", plan_id="a" * 64, scene_hash="b" * 64, expected_part_id="part_a", actual_part_id="part_b", slot_id="slot_qualified")


def test_v1_09_final_probe_requires_six_same_run_proofs() -> None:
    with pytest.raises(ValueError, match="DEFECT_SORT"):
        probe_defect_final(SimpleNamespace(), SimpleNamespace(), run_id="run-1", frame_id="frame-1", plan_id="a" * 64, scene_hash="b" * 64, entry_evidence=[], consumed_entry_ids=[])
