import math

from robot_backends.base import RobotBackend


class CoppeliaSimRobotBackend(RobotBackend):
    """CoppeliaSim simulation backend for BLX openr6_arm.

    Key mapping rules (compared to real Blinx robot):
      - Real robot uses degrees; CoppeliaSim joints use radians.
      - BLX has no mechanical offset (joint5_offset_deg = 0).
      - speed parameter is logged and ignored (first-phase degradation).
      - IK uses simIK.addElementFromScene + handleGroup + applyIkEnvironmentToScene.
      - IK uses constraint_position only (orientation not constrained).
      - Both move_coordinate_all() input and positive_solution() output
        use the same world-frame convention for TCP pose.

    Workspace constraints (verified 2026-06-03):
      - <=150mm from base: IK precision < 0.2mm
      - >150mm from base: may hit joint limits, IK may fail
      - TCP is 80mm from link6 end (BLX_tool_suction dummy)
    """

    def __init__(self, settings, sim_client=None):
        self._settings = settings
        self._joint_paths = settings["joint_paths"]
        self._tool_path = settings["tool_path"]
        self._home_angles = list(settings["home_angles"])
        self._joint5_offset_deg = settings.get("joint5_offset_deg", -90)

        # IK config
        self._base_path = settings.get("base_path", "/IRB140")
        self._tip_path = settings.get("tip_path", "/IRB140/tip")
        self._target_path = settings.get("target_path", "/IRB140/target")
        self._ik_search_time = settings.get("ik_search_time", 0.25)
        self._ik_search_precision = settings.get("ik_search_precision", 0.1)
        self._ik_metric = settings.get("ik_metric", [1, 1, 1, 0.1])
        self._ik_global_max_dist = settings.get("ik_global_max_dist", 0.5)

        # Internal state: always stored in *real-robot degrees*
        self.current_angles = list(self._home_angles)

        # Lazy connection objects
        self._client = sim_client
        self._sim = None
        self._joint_handles = None
        self._tool_handle = None

        # Lazy IK objects (populated by _ensure_ik)
        self._simIK = None
        self._ik_env = None
        self._ik_group = None
        self._ik_base_handle = None
        self._ik_tip_handle = None
        self._ik_target_handle = None
        self._ik_mapping = None
        self._ik_joint_handles = None  # IK-space joint handles

    # ------------------------------------------------------------------
    # Lazy connection helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self):
        if self._sim is not None:
            return
        if self._client is None:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            self._client = RemoteAPIClient()
        self._sim = self._client.getObject("sim")
        self._resolve_handles()

    def _resolve_handles(self):
        """Resolve CoppeliaSim object handles for joints and tool."""
        import logging
        logger = logging.getLogger("CoppeliaSimBackend")
        self._joint_handles = []
        for path in self._joint_paths:
            h = self._sim.getObject(path)
            self._joint_handles.append(h)
            logger.info("[SIM] resolved joint: %s -> handle %s", path, h)
        print(f"[SIM] joints: {self._joint_paths}")
        print(f"[SIM] tool:   {self._tool_path}")

        tool_no_error = self._settings.get("tool_no_error", False)
        if tool_no_error:
            self._tool_handle = self._sim.getObject(self._tool_path, {"noError": True})
        else:
            self._tool_handle = self._sim.getObject(self._tool_path)
        print(f"[SIM] tool_suction handle: {self._tool_handle}")
        print(f"[SIM] joint5_offset_deg: {self._joint5_offset_deg}")

    def _ensure_ik(self):
        """Lazy-initialise the IK environment on first coordinate call.

        Uses addElementFromScene + handleGroup (verified with BLX model).
        Creates a target dummy (/BLX_target) if it doesn't exist.
        """
        if self._simIK is not None:
            return
        self._ensure_connected()
        self._simIK = self._client.require("simIK")

        # Create target dummy if not in scene
        try:
            self._ik_target_handle = self._sim.getObject(self._target_path)
        except Exception:
            self._ik_target_handle = self._sim.createDummy(0.01)
            self._sim.setObjectAlias(self._ik_target_handle, "BLX_target")

        self._ik_base_handle = self._sim.getObject(self._base_path)
        self._ik_tip_handle = self._sim.getObject(self._tip_path)

        # Create IK environment from scene
        self._ik_env = self._simIK.createEnvironment()
        self._ik_group = self._simIK.createGroup(self._ik_env)

        # Position target at current TCP before adding element
        tcp_pos = self._sim.getObjectPosition(self._ik_tip_handle, -1)
        tcp_orient = self._sim.getObjectOrientation(self._ik_tip_handle, -1)
        self._sim.setObjectPosition(self._ik_target_handle, tcp_pos, -1)
        self._sim.setObjectOrientation(self._ik_target_handle, tcp_orient, -1)

        # Add IK element: position only (verified 2026-06-03)
        result = self._simIK.addElementFromScene(
            self._ik_env,
            self._ik_group,
            self._ik_base_handle,
            self._ik_tip_handle,
            self._ik_target_handle,
            self._simIK.constraint_position,
        )
        self._ik_mapping = result[1]

        # Extract IK-space joint handles from the scene->IK mapping
        self._ik_joint_handles = []
        for scene_handle in self._joint_handles:
            ik_handle = self._ik_mapping[scene_handle]
            self._ik_joint_handles.append(ik_handle)

        print(f"[SIM] IK initialised: env={self._ik_env} group={self._ik_group}")
        print(f"[SIM] IK chain: base={self._base_path} tip={self._tip_path}")
        print(f"[SIM] IK constraint: position only")
        print(f"[SIM] IK joint handles: {self._ik_joint_handles}")

    # ------------------------------------------------------------------
    # Degree / radian + offset helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deg_to_rad(deg):
        return math.radians(deg)

    def _apply_joint5_offset(self, angle_deg):
        """Real-robot angle -> CoppeliaSim angle (degrees)."""
        return angle_deg + self._joint5_offset_deg

    def _to_sim_position(self, joint_index, angle_deg):
        """Convert a real-robot joint angle (degrees) to the CoppeliaSim
        joint position (radians), applying the joint5 offset when needed.
        """
        adjusted = angle_deg
        if joint_index == 4:  # joint5 is index 4 (0-based)
            adjusted = self._apply_joint5_offset(angle_deg)
        return self._deg_to_rad(adjusted)

    def _from_sim_position(self, joint_index, pos_rad):
        """Reverse of _to_sim_position: CoppeliaSim rad -> real-robot deg."""
        angle_deg = math.degrees(pos_rad)
        if joint_index == 4:
            angle_deg = angle_deg - self._joint5_offset_deg
        return angle_deg

    # ------------------------------------------------------------------
    # Speed degradation (first-phase strategy)
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_speed(speed):
        """First-phase speed degradation: log and ignore."""
        import logging
        logging.getLogger("CoppeliaSimBackend").debug(
            "[SIM] speed=%s ignored (first-phase: direct kinematic mode)", speed
        )

    # ------------------------------------------------------------------
    # RobotBackend interface
    # ------------------------------------------------------------------

    def home(self):
        """Initialize / reset: resolve handles and move joints to home position."""
        self._ensure_connected()
        self.current_angles = list(self._home_angles)
        for i, handle in enumerate(self._joint_handles):
            pos_rad = self._to_sim_position(i, self._home_angles[i])
            self._sim.setJointPosition(handle, pos_rad)

    def move_home(self):
        """Move robot to home position."""
        if self._sim is None:
            self._ensure_connected()
        self.current_angles = list(self._home_angles)
        for i, handle in enumerate(self._joint_handles):
            pos_rad = self._to_sim_position(i, self._home_angles[i])
            self._sim.setJointPosition(handle, pos_rad)

    def move_angle(self, joint_id, speed, value):
        self._ensure_connected()
        self._handle_speed(speed)
        idx = joint_id - 1
        self.current_angles[idx] = value
        pos_rad = self._to_sim_position(idx, value)
        self._sim.setJointPosition(self._joint_handles[idx], pos_rad)

    def move_angle_all(self, value1, value2, value3, value4, value5, value6, speed):
        self._ensure_connected()
        self._handle_speed(speed)
        angles = [value1, value2, value3, value4, value5, value6]
        self.current_angles = list(angles)
        for i, handle in enumerate(self._joint_handles):
            pos_rad = self._to_sim_position(i, angles[i])
            self._sim.setJointPosition(handle, pos_rad)

    def move_coordinate_all(self, value1, value2, value3, value4, value5, value6, speed):
        """IK-based Cartesian coordinate control (POSITION ONLY).

        Input convention:
          X, Y, Z  in mm   — world frame  ✅ supported
          RX, RY, RZ in deg — world frame  ⚠️ IGNORED (position-only IK)

        Only X, Y, Z are used for IK solving. RX/RY/RZ are accepted as
        parameters for interface compatibility but have no effect on the
        robot pose. Do NOT rely on RX/RY/RZ for task verification.

        Workspace constraints:
          <=150mm from base: IK precision < 0.2mm
          >150mm from base: may hit joint limits, IK may fail
        """
        self._ensure_connected()
        self._ensure_ik()
        self._handle_speed(speed)

        # Log ignored orientation values
        if any(v != 0 for v in (value4, value5, value6)):
            import logging
            logging.getLogger("CoppeliaSimBackend").warning(
                "[SIM] RX/RY/RZ=[%.1f, %.1f, %.1f] ignored (position-only IK)",
                value4, value5, value6,
            )

        # mm -> m (world frame)
        position = [value1 / 1000.0, value2 / 1000.0, value3 / 1000.0]

        # Set target position in WORLD frame (-1 = world)
        self._sim.setObjectPosition(self._ik_target_handle, position, -1)

        # First try the fast local solve.  A teaching task often asks the arm
        # to move from home to a distant Cartesian point, where a local solve
        # can fail even though the point is reachable.  In that case, use
        # simIK's randomized global configuration search.
        result = self._simIK.handleGroup(self._ik_env, self._ik_group, {'syncWorlds': True})
        ik_result = result[0]

        if ik_result != 1:
            configurations = self._simIK.findConfigs(
                self._ik_env,
                self._ik_group,
                self._ik_joint_handles,
                {
                    "maxDist": self._ik_global_max_dist,
                    "maxTime": self._ik_search_time,
                    "pMetric": list(self._ik_metric),
                    "cMetric": [1.0] * len(self._ik_joint_handles),
                    "findAlt": False,
                    "findMultiple": False,
                },
            )
            if not configurations:
                raise RuntimeError(
                    f"[SIM] IK failed (local result={ik_result}, "
                    "global search found no configuration), "
                    f"target=[{value1:.1f}, {value2:.1f}, {value3:.1f}]mm. "
                    f"Workspace limit: 150mm from base."
                )
            configuration = configurations[0]
            if len(configuration) != len(self._ik_joint_handles):
                raise RuntimeError(
                    "[SIM] IK global search returned an invalid "
                    f"{len(configuration)}-joint configuration"
                )
            for handle, joint_position in zip(
                self._ik_joint_handles,
                configuration,
            ):
                self._simIK.setJointPosition(
                    self._ik_env,
                    handle,
                    joint_position,
                )
            # findConfigs leaves the IK environment unchanged.  After
            # installing the selected configuration, sync it directly to the
            # scene.  applyIkEnvironmentToScene cannot be used here because
            # that deprecated helper first overwrites the configuration from
            # the scene and launches another local solve.
            self._simIK.syncToSim(
                self._ik_env,
                [self._ik_group],
            )
        else:
            # handleGroup(syncWorlds=True) has already synchronized the
            # successful local solution to the scene.  Keep the historical
            # helper call for compatibility with older CoppeliaSim releases.
            self._simIK.applyIkEnvironmentToScene(
                self._ik_env,
                self._ik_group,
                True,
            )

        # Read back joint positions
        for i, handle in enumerate(self._joint_handles):
            pos_rad = self._sim.getJointPosition(handle)
            self.current_angles[i] = self._from_sim_position(i, pos_rad)

    def pump_on(self):
        self._ensure_connected()
        if self._tool_handle is not None:
            try:
                self._sim.setObjectColor(
                    self._tool_handle, 0,
                    self._sim.colorcomponent_ambient_diffuse,
                    [0.0, 1.0, 0.0],
                )
            except Exception:
                pass
            self._pump_state = True

    def pump_off(self):
        self._ensure_connected()
        if self._tool_handle is not None:
            try:
                self._sim.setObjectColor(
                    self._tool_handle, 0,
                    self._sim.colorcomponent_ambient_diffuse,
                    [1.0, 0.0, 0.0],
                )
            except Exception:
                pass
            self._pump_state = False

    def positive_solution(self):
        """Return Cartesian pose [X, Y, Z, RX, RY, RZ] of /tool_suction.

        Convention (same as move_coordinate_all input):
          X, Y, Z  in mm   — world frame
          RX, RY, RZ in deg — world frame (Euler XYZ)
        """
        self._ensure_connected()
        if self._tool_handle is None:
            return [0.0] * 6
        pos = self._sim.getObjectPosition(self._tool_handle, -1)  # world
        orient = self._sim.getObjectOrientation(self._tool_handle, -1)  # Euler rad
        x_mm = pos[0] * 1000.0
        y_mm = pos[1] * 1000.0
        z_mm = pos[2] * 1000.0
        rx_deg = math.degrees(orient[0])
        ry_deg = math.degrees(orient[1])
        rz_deg = math.degrees(orient[2])
        return [round(x_mm, 1), round(y_mm, 1), round(z_mm, 1),
                round(rx_deg, 1), round(ry_deg, 1), round(rz_deg, 1)]

    def close(self):
        self._sim = None
        self._client = None
        self._joint_handles = None
        self._tool_handle = None
        # IK state cleanup
        self._simIK = None
        self._ik_env = None
        self._ik_group = None
        self._ik_base_handle = None
        self._ik_tip_handle = None
        self._ik_target_handle = None
        self._ik_mapping = None
        self._ik_joint_handles = None
