import sys

from vision_platform.capabilities import (
    probe_coppeliasim,
    probe_hikvision,
    probe_replay,
)


def test_coppeliasim_probe_requires_executable_and_zmq_client(tmp_path):
    missing = probe_coppeliasim(tmp_path)
    assert missing.available is False
    assert "coppeliaSim.exe" in missing.reason

    (tmp_path / "coppeliaSim.exe").write_bytes(b"")
    source = (
        tmp_path
        / "programming"
        / "zmqRemoteApi"
        / "clients"
        / "python"
        / "src"
    )
    source.mkdir(parents=True)
    available = probe_coppeliasim(tmp_path)
    assert available.available is True


def test_hikvision_probe_does_not_import_sdk(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "MvImport.MvCameraControl_class", None)

    capability = probe_hikvision([tmp_path])

    assert capability.available is False
    assert "MVS" in capability.reason


def test_hikvision_probe_accepts_windows_runtime_layout(tmp_path):
    dll = tmp_path / "Development" / "MVS" / "Runtime" / "Win64_x64"
    dll.mkdir(parents=True)
    (dll / "MvCameraControl.dll").write_bytes(b"")

    capability = probe_hikvision([tmp_path])

    assert capability.available is True
    assert capability.details["runtime"] == str(dll.resolve())


def test_replay_probe_reports_missing_and_available_manifests(tmp_path):
    manifest = tmp_path / "replay.json"
    assert probe_replay(manifest).available is False
    manifest.write_text('{"frames":[]}', encoding="utf-8")
    assert probe_replay(manifest).available is True
