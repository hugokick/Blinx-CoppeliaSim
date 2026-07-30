from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_vision_environment_contract_declares_required_packages():
    requirements = (ROOT / "requirements-vision.txt").read_text(encoding="utf-8")
    for package in (
        "numpy",
        "opencv-python",
        "PyQt5",
        "pyzmq",
        "cbor2",
        "pytest",
        "pytest-qt",
    ):
        assert package.lower() in requirements.lower()


def test_cad_environment_declares_mesh_and_ocp_tools():
    requirements = (ROOT / "requirements-vision-cad.txt").read_text(
        encoding="utf-8"
    )
    assert "trimesh" in requirements.lower()
    assert "cadquery-ocp" in requirements.lower()


def test_bootstrap_uses_project_local_venv_and_coppeliasim_root():
    script = (ROOT / "tools/vision_lab/bootstrap.ps1").read_text(
        encoding="utf-8"
    )
    assert ".venv-vision" in script
    assert "COPPELIASIM_ROOT" in script
    assert "zmqRemoteApi" in script
    assert "coppeliasim-local.pth" in script


def test_bootstrap_preserves_python_inline_quotes_and_native_failures():
    script = (ROOT / "tools/vision_lab/bootstrap.ps1").read_text(
        encoding="utf-8"
    )
    assert "print('VISION_ENV_OK')" in script
    assert "VISION_CAD_ENV_OK" in script
    assert "if ($LASTEXITCODE -ne 0)" in script


def test_bootstrap_sets_qt_plugin_path_for_unicode_projects():
    script = (ROOT / "tools/vision_lab/bootstrap.ps1").read_text(
        encoding="utf-8"
    )
    assert "vision-lab-qt.pth" in script
    assert "QT_QPA_PLATFORM_PLUGIN_PATH" in script
    assert "os.path.join(sys.prefix" in script


def test_python_wrapper_requires_the_project_venv():
    script = (ROOT / "tools/vision_lab/python.ps1").read_text(encoding="utf-8")
    assert ".venv-vision" in script
    assert "bootstrap.ps1" in script


def test_generated_environment_is_ignored_by_git():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".venv-vision/" in gitignore
    assert "artifacts/vision_lab/" in gitignore
