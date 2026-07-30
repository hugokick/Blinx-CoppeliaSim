import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_sim_smoke_check(backend=None):
    if backend is None:
        try:
            from robot_backends.factory import create_robot_backend
            backend = create_robot_backend()
        except Exception as exc:
            return {"status": "FAIL", "message": f"backend creation failed: {exc}"}
    try:
        backend.home()
        backend.move_angle(1, 50, 10)
        backend.pump_on()
        backend.pump_off()
        return {"status": "PASS", "message": "sim smoke check passed"}
    except Exception as exc:
        return {"status": "FAIL", "message": str(exc)}
    finally:
        try:
            backend.close()
        except Exception:
            pass


if __name__ == "__main__":
    result = run_sim_smoke_check()
    print(f"[{result['status']}] {result['message']}")
    sys.exit(0 if result["status"] == "PASS" else 1)
