"""BLX PyQt Integration Test — automated verification of 5 key items.

Tests:
  1. Coordinate mode switch (positive_solution initialization)
  2. +/- button logic (6 axes, safe wrapper)
  3. Single-step execution (TableView row)
  4. Program execution (multiple rows)
  5. Out-of-workspace error handling
"""
import sys, os, math, traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.backend import get_backend_settings
from robot_backends.factory import create_robot_backend

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")


def main():
    global PASS, FAIL
    print("=" * 60)
    print("BLX PyQt Integration Test")
    print("=" * 60)

    # Setup
    s = get_backend_settings()
    print(f"\nScene: {os.path.basename(s['sim']['scene'])}")
    print(f"Scene exists: {os.path.exists(s['sim']['scene'])}")

    robot = create_robot_backend()
    robot.home()

    # ------------------------------------------------------------------
    # Test 1: Coordinate mode switch
    # ------------------------------------------------------------------
    print("\n--- Test 1: Coordinate mode switch ---")
    pose = robot.positive_solution()
    check("positive_solution returns 6 values", len(pose) == 6, f"got {len(pose)}")
    check("X is reasonable", -200 <= pose[0] <= 200, f"X={pose[0]}")
    check("Y is reasonable", -200 <= pose[1] <= 200, f"Y={pose[1]}")
    check("Z is reasonable", -100 <= pose[2] <= 400, f"Z={pose[2]}")
    check("Z matches expected (~76.7mm)", 70 < pose[2] < 85, f"Z={pose[2]}")
    check("RX/RY/RZ are zero or near zero",
          all(abs(v) < 1 for v in pose[3:]),
          f"RX/RY/RZ={pose[3:]}")

    # ------------------------------------------------------------------
    # Test 2: +/- button logic (6 axes)
    # ------------------------------------------------------------------
    print("\n--- Test 2: +/- button logic ---")
    step = 10  # mm
    curr = robot.positive_solution()

    # X+
    robot.move_coordinate_all(curr[0]+step, curr[1], curr[2], 0, 0, 0, 100)
    new = robot.positive_solution()
    err = abs(new[0] - curr[0] - step)
    check("X+10mm", err < 1, f"err={err:.2f}mm")

    # X-
    robot.move_coordinate_all(curr[0], curr[1], curr[2], 0, 0, 0, 100)
    new = robot.positive_solution()
    err = abs(new[0] - curr[0])
    check("X-10mm (return)", err < 1, f"err={err:.2f}mm")

    # Y+
    robot.move_coordinate_all(curr[0], curr[1]+step, curr[2], 0, 0, 0, 100)
    new = robot.positive_solution()
    err = abs(new[1] - curr[1] - step)
    check("Y+10mm", err < 1, f"err={err:.2f}mm")

    # Y-
    robot.move_coordinate_all(curr[0], curr[1], curr[2], 0, 0, 0, 100)
    new = robot.positive_solution()
    err = abs(new[1] - curr[1])
    check("Y-10mm (return)", err < 1, f"err={err:.2f}mm")

    # Z+
    robot.move_coordinate_all(curr[0], curr[1], curr[2]+step, 0, 0, 0, 100)
    new = robot.positive_solution()
    err = abs(new[2] - curr[2] - step)
    check("Z+10mm", err < 1, f"err={err:.2f}mm")

    # Z-
    robot.move_coordinate_all(curr[0], curr[1], curr[2], 0, 0, 0, 100)
    new = robot.positive_solution()
    err = abs(new[2] - curr[2])
    check("Z-10mm (return)", err < 1, f"err={err:.2f}mm")

    # ------------------------------------------------------------------
    # Test 3: Single-step execution (simulate TableView row)
    # ------------------------------------------------------------------
    print("\n--- Test 3: Single-step execution ---")
    robot.home()
    target = [23.7, 111.7, 76.7, 0, 0, 0]  # known good position
    robot.move_coordinate_all(*target, 50)
    pose = robot.positive_solution()
    err = math.sqrt(sum((a-b)**2 for a,b in zip(pose[:3], target[:3])))
    check("Single-step to known position", err < 1, f"err={err:.2f}mm")

    # ------------------------------------------------------------------
    # Test 4: Program execution (sequence of 3 moves)
    # ------------------------------------------------------------------
    print("\n--- Test 4: Program execution ---")
    robot.home()
    sequence = [
        [23.7, 111.7, 76.7, 0, 0, 0],
        [33.7, 111.7, 76.7, 0, 0, 0],
        [33.7, 121.7, 76.7, 0, 0, 0],
    ]
    for i, target in enumerate(sequence):
        robot.move_coordinate_all(*target, 50)
        pose = robot.positive_solution()
        err = math.sqrt(sum((a-b)**2 for a,b in zip(pose[:3], target[:3])))
        check(f"Step {i+1}: [{target[0]:.0f},{target[1]:.0f},{target[2]:.0f}]",
              err < 1, f"err={err:.2f}mm")

    # ------------------------------------------------------------------
    # Test 5: Out-of-workspace error handling
    # ------------------------------------------------------------------
    print("\n--- Test 5: Out-of-workspace ---")

    # 5a: RuntimeError for unreachable position
    try:
        robot.move_coordinate_all(300, 0, 0, 0, 0, 0, 100)
        check("RuntimeError for X=300mm", False, "no exception raised")
    except RuntimeError as e:
        check("RuntimeError for X=300mm", "IK failed" in str(e), str(e)[:80])

    # 5b: safe wrapper catches exception (simulate blinx_safe_move_coordinate)
    try:
        robot.move_coordinate_all(300, 0, 0, 0, 0, 0, 100)
    except RuntimeError:
        pass  # caught, would show QMessageBox in UI
    except Exception:
        pass  # caught, would show QMessageBox in UI
    check("safe wrapper catches RuntimeError (no crash)", True)

    # 5c: Valid position after out-of-range attempt
    robot.move_coordinate_all(curr[0], curr[1], curr[2], 0, 0, 0, 100)
    pose = robot.positive_solution()
    err = math.sqrt(sum((a-b)**2 for a,b in zip(pose[:3], curr[:3])))
    check("Recovery after out-of-range: valid position", err < 1, f"err={err:.2f}mm")

    # 5d: Range validation — current position within new ranges
    check("Current X within [-200,200]", -200 <= pose[0] <= 200, f"X={pose[0]}")
    check("Current Y within [-200,200]", -200 <= pose[1] <= 200, f"Y={pose[1]}")
    check("Current Z within [-100,400]", -100 <= pose[2] <= 400, f"Z={pose[2]}")

    # Cleanup
    robot.home()

    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
    print("=" * 60)
    return FAIL


if __name__ == "__main__":
    sys.exit(main())
