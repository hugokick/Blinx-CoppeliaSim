from config.backend import get_backend_settings
from robot_backends.real_robot import RealRobotBackend

try:
    from robot_backends.coppeliasim_robot import CoppeliaSimRobotBackend
except ImportError:
    CoppeliaSimRobotBackend = None


def create_robot_backend():
    settings = get_backend_settings()
    backend_name = settings["backend"]
    if backend_name == "real":
        return RealRobotBackend()
    if backend_name == "sim":
        if CoppeliaSimRobotBackend is None:
            raise RuntimeError("CoppeliaSim backend is not available")
        return CoppeliaSimRobotBackend(settings=settings["sim"])
    raise ValueError(f"Unsupported robot backend: {backend_name}")
