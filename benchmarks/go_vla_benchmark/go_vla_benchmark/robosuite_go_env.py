"""Robosuite-backed 5x5 Go environment with rigid stone bodies."""

from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np

try:
    import pyspiel
except Exception as exc:  # pragma: no cover - import diagnostics path
    pyspiel = None
    _PYSPIEL_IMPORT_ERROR = exc
else:
    _PYSPIEL_IMPORT_ERROR = None

import tempfile

from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import new_geom

from .board_texture import generate_board_texture
from .common import GoResetOptions

SELF = 0
OPPONENT = 1


class _OpenSpielGoLogic:
    """Small OpenSpiel go wrapper for legal actions and board-state tracking."""

    def __init__(self, board_size: int):
        if pyspiel is None:
            raise ImportError(
                "pyspiel is required for the robosuite Go backend but could not be imported"
            ) from _PYSPIEL_IMPORT_ERROR
        self._board_size = int(board_size)
        self._game = pyspiel.load_game("go", {"board_size": self._board_size})
        self.reset()

    def reset(self) -> None:
        self._state = self._game.new_initial_state()
        self._moves = np.full((self._board_size * self._board_size * 2,), fill_value=-1, dtype=np.int32)
        self._move_id = 0

    @property
    def board_size(self) -> int:
        return self._board_size

    @property
    def is_game_over(self) -> bool:
        return bool(self._state.is_terminal())

    @property
    def current_player(self) -> int:
        return int(self._state.current_player())

    def legal_actions(self) -> List[int]:
        return [int(a) for a in self._state.legal_actions()]

    def apply(self, player: int, action_int: int) -> bool:
        action_int = int(action_int)
        if int(self._state.current_player()) != int(player):
            return False
        legal = self._state.legal_actions()
        if action_int not in legal:
            return False
        self._state.apply_action(action_int)
        if self._move_id < self._moves.shape[0]:
            self._moves[self._move_id] = action_int
            self._move_id += 1
        return True

    def get_board_state(self) -> np.ndarray:
        board_state = np.reshape(
            np.array(self._state.observation_tensor(0), dtype=bool),
            [4, self._board_size, self._board_size],
        )
        board_state = np.transpose(board_state, [1, 2, 0])
        # Match physics_planning_games channel convention used by the benchmark.
        return board_state[:, :, [2, 0, 1, 3]]

    def get_move_history(self) -> np.ndarray:
        return self._moves.copy()


class _GoStoneObject(MujocoXMLObject):
    """Go stone with separate visual and collision geoms plus explicit inertial data."""

    def __init__(
        self,
        name: str,
        visual_radius: float,
        visual_half_height: float,
        collision_radius: float,
        collision_half_height: float,
        mass: float,
        rgba: List[float],
    ):
        a = float(collision_radius)
        c = float(collision_half_height)
        stone_mass = float(mass)
        diaginertia = np.array(
            [
                0.2 * stone_mass * (a * a + c * c),
                0.2 * stone_mass * (a * a + c * c),
                0.4 * stone_mass * (a * a),
            ],
            dtype=np.float64,
        )

        root = ET.Element("mujoco", model=name)
        worldbody = ET.SubElement(root, "worldbody")
        outer_body = ET.SubElement(worldbody, "body")
        body = ET.SubElement(outer_body, "body", name="object")
        ET.SubElement(
            body,
            "inertial",
            pos="0 0 0",
            mass=self._arr_to_str([stone_mass]),
            diaginertia=self._arr_to_str(diaginertia),
        )
        ET.SubElement(
            body,
            "geom",
            name="stone_visual",
            type="cylinder",
            pos="0 0 0",
            size=self._arr_to_str([visual_radius, visual_half_height]),
            group="1",
            contype="0",
            conaffinity="0",
            rgba=self._arr_to_str(rgba),
        )
        ET.SubElement(
            body,
            "geom",
            name="stone_collision",
            type="ellipsoid",
            pos="0 0 0",
            size=self._arr_to_str([collision_radius, collision_radius, collision_half_height]),
            group="0",
            rgba="0 0 0 0",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as tmp:
            tmp.write(ET.tostring(root, encoding="unicode"))
            xml_path = tmp.name

        try:
            super().__init__(
                fname=xml_path,
                name=name,
                joints=[dict(type="free")],
                obj_type="all",
                duplicate_collision_geoms=False,
            )
        finally:
            try:
                os.unlink(xml_path)
            except OSError:
                pass

    @staticmethod
    def _arr_to_str(values) -> str:
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        return " ".join(f"{float(v):.8g}" for v in vals)


class _Go5x5RigidRobosuite(ManipulationEnv):
    """Minimal robosuite task: table + board visual + rigid stone cylinders."""

    def __init__(
        self,
        robots: str = "Panda",
        controller_configs: Optional[dict] = None,
        gripper_types: str = "default",
        initialization_noise: str | dict | None = "default",
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        render_camera: str = "agentview",
        render_collision_mesh: bool = False,
        render_visual_mesh: bool = True,
        render_gpu_device_id: int = -1,
        control_freq: int = 8,
        lite_physics: bool = True,
        horizon: int = 400,
        ignore_done: bool = False,
        hard_reset: bool = True,
        camera_names: str = "agentview",
        camera_heights: int = 256,
        camera_widths: int = 256,
        renderer: str = "mujoco",
        renderer_config: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        self.board_size = 5
        self.table_full_size = np.array((0.9, 0.9, 0.05), dtype=np.float32)
        self.table_friction = (1.5, 0.05, 0.02)
        self.table_offset = np.array((0.0, 0.0, 0.8), dtype=np.float32)

        self.board_spacing = 0.045
        self.board_center_xy = np.array((0.02, 0.0), dtype=np.float32)
        self._board_center_xy_default = self.board_center_xy.copy()
        self._board_shift_range = 0.015  # ±1.5cm XY randomization
        self.stone_radius = 0.013
        self.stone_height = 0.006
        self.stone_half_height = self.stone_height * 0.5
        self.collision_stone_radius = self.stone_radius * 0.94
        self.collision_stone_half_height = self.stone_half_height
        self.stone_mass = 4000.0 * np.pi * (self.stone_radius ** 2) * self.stone_height
        self._table_stone_clearance = 0.0005
        self._board_stone_clearance = 0.005
        self._board_margin = 0.035  # margin beyond outermost grid lines

        self._stone_objects: List[_GoStoneObject] = []
        self._stone_joint_names: List[str] = []
        self._stone_body_ids: List[int] = []
        self._board_rotation_rad = 0.0

        self._white_count = self.board_size * self.board_size
        self._black_count = self.board_size * self.board_size

        super().__init__(
            robots=robots,
            env_configuration="default",
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=False,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=False,
            camera_segmentations=None,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    @property
    def table_top_z(self) -> float:
        """World z of the table support surface used for off-board stones."""
        return float(self.table_offset[2])

    @property
    def board_surface_z(self) -> float:
        """World z of the board collision surface top (where stones rest)."""
        if hasattr(self, "sim"):
            try:
                board_geom_id = self.sim.model.geom_name2id("go_board_surface_collision")
                return float(
                    self.sim.data.geom_xpos[board_geom_id, 2]
                    + self.sim.model.geom_size[board_geom_id, 2]
                )
            except Exception:
                pass
        table_half_h = 0.5 * float(self.table_full_size[2])
        board_thickness = 0.0025
        return float(self.table_offset[2]) + table_half_h + 2 * board_thickness

    @property
    def board_intersections_xyz(self) -> np.ndarray:
        """Board intersection world positions derived from the collision geom world pose.

        Reads ``geom_xpos`` for the board collision surface so the grid
        automatically reflects XY shift, Z-rotation, and the table-body
        world transform.
        """
        grid = np.zeros((self.board_size, self.board_size, 3), dtype=np.float32)
        z = self.board_surface_z + self.collision_stone_half_height
        half = 0.5 * (self.board_size - 1) * self.board_spacing
        # Use geom world position and orientation as the single source of truth
        if hasattr(self, "sim"):
            coll_gid = self.sim.model.geom_name2id("go_board_surface_collision")
            cx = float(self.sim.data.geom_xpos[coll_gid, 0])
            cy = float(self.sim.data.geom_xpos[coll_gid, 1])
            # Use the geom's world rotation matrix (already includes board rotation)
            xmat = self.sim.data.geom_xmat[coll_gid].reshape(3, 3)
        else:
            cx, cy = float(self.board_center_xy[0]), float(self.board_center_xy[1])
            xmat = np.eye(3)
        for row in range(self.board_size):
            for col in range(self.board_size):
                lx = -half + col * self.board_spacing
                ly = half - row * self.board_spacing  # row 0 = most-positive-Y
                # Transform local grid offset through the geom's world orientation
                grid[row, col, 0] = cx + xmat[0, 0] * lx + xmat[0, 1] * ly
                grid[row, col, 1] = cy + xmat[1, 0] * lx + xmat[1, 1] * ly
                grid[row, col, 2] = z
        return grid

    @property
    def source_stone_xyz(self) -> np.ndarray:
        board_half = 0.5 * float(self.board_size - 1) * self.board_spacing
        x = self.board_center_xy[0] - board_half - 0.12
        y = self.board_center_xy[1]
        z = self.table_top_z + self.collision_stone_half_height + self._table_stone_clearance
        return np.array([x, y, z], dtype=np.float32)

    @property
    def white_stone_indices(self) -> List[int]:
        return list(range(self._white_count))

    @property
    def black_stone_indices(self) -> List[int]:
        start = self._white_count
        return list(range(start, start + self._black_count))

    def reward(self, action=None):
        del action
        return 0.0

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](float(self.table_full_size[0]))
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=tuple(self.table_full_size.tolist()),
            table_friction=self.table_friction,
            table_offset=tuple(self.table_offset.tolist()),
        )
        mujoco_arena.set_origin([0.0, 0.0, 0.0])

        # Table texture pool — one material per texture for domain randomization
        # (MuJoCo caches textures on GPU so mat_texid swaps don't render;
        #  instead we swap geom_matid at runtime)
        import xml.etree.ElementTree as ET
        import robosuite as _robosuite_mod
        _tex_dir = Path(_robosuite_mod.__file__).parent / "models" / "assets" / "textures"
        self._table_mat_names = []
        for tex_file in [
            "light-wood.png", "ceramic.png", "clay.png",
            "cream-plaster.png", "gray-woodgrain.png",
            "wood-tiles.png", "wood-varnished-panels.png",
            "gray-felt.png",
        ]:
            stem = Path(tex_file).stem
            tex_name = f"tex-table-{stem}"
            mat_name = f"mat-table-{stem}"
            ET.SubElement(mujoco_arena.asset, "texture", {
                "name": tex_name,
                "file": str(_tex_dir / tex_file),
                "type": "2d",
            })
            ET.SubElement(mujoco_arena.asset, "material", {
                "name": mat_name,
                "texture": tex_name,
                "texrepeat": "3 3",
                "texuniform": "true",
                "reflectance": "0.02",
                "shininess": "0.1",
                "specular": "0.15",
            })
            self._table_mat_names.append(mat_name)
        mujoco_arena.table_visual.set("material", self._table_mat_names[0])

        # Stone materials for glossiness variation
        for mat_name, mat_rgba in [("stone_mat_white", "0.96 0.96 0.96 1"),
                                   ("stone_mat_black", "0.08 0.08 0.08 1")]:
            ET.SubElement(mujoco_arena.asset, "material", {
                "name": mat_name,
                "shininess": "0.3",
                "specular": "0.5",
                "reflectance": "0.01",
                "rgba": mat_rgba,
            })

        # Board grid texture
        board_half = 0.5 * float(self.board_size - 1) * self.board_spacing + self._board_margin
        board_tex_img = generate_board_texture(
            board_size=self.board_size,
            board_spacing=self.board_spacing,
            board_half=board_half,
        )
        self._board_tex_tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        board_tex_img.save(self._board_tex_tmpfile.name)
        ET.SubElement(mujoco_arena.asset, "texture", {
            "name": "tex-go-board",
            "file": self._board_tex_tmpfile.name,
            "type": "2d",
        })
        ET.SubElement(mujoco_arena.asset, "material", {
            "name": "go_board_mat",
            "texture": "tex-go-board",
            "texrepeat": "1 1",
            "texuniform": "false",
            "rgba": "0.74 0.62 0.46 1",
        })

        self._add_board_visuals(mujoco_arena=mujoco_arena)

        # Extra perturbation light (invisible by default, activated probabilistically)
        ET.SubElement(mujoco_arena.worldbody, "light", {
            "name": "extra_perturbation_light",
            "pos": "0 0 2.0",
            "dir": "0 0 -1",
            "diffuse": "0 0 0",
            "specular": "0 0 0",
            "ambient": "0 0 0",
            "directional": "false",
            "castshadow": "false",
        })

        # Stones use an explicit centered inertial model and a separate
        # collision ellipsoid for smoother board contact.

        self._stone_objects = []
        for color, count, rgba in [("white", self._white_count, [0.96, 0.96, 0.96, 1.0]),
                                   ("black", self._black_count, [0.08, 0.08, 0.08, 1.0])]:
            for idx in range(count):
                obj = _GoStoneObject(
                    name=f"{color}_stone_{idx}",
                    visual_radius=self.stone_radius,
                    visual_half_height=self.stone_half_height,
                    collision_radius=self.collision_stone_radius,
                    collision_half_height=self.collision_stone_half_height,
                    mass=self.stone_mass,
                    rgba=rgba,
                )
                self._stone_objects.append(obj)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self._stone_objects,
        )

    def _add_board_visuals(self, mujoco_arena: TableArena) -> None:
        table_body = mujoco_arena.table_body
        table_half_h = 0.5 * float(self.table_full_size[2])
        board_half = 0.5 * float(self.board_size - 1) * self.board_spacing + self._board_margin
        board_thickness = 0.0025
        board_center_local = np.array(
            [
                self.board_center_xy[0],
                self.board_center_xy[1],
                table_half_h + board_thickness,
            ],
            dtype=np.float32,
        )
        # Visual board surface (textured with grid lines via go_board_mat)
        # NOTE: explicit rgba is required alongside the material — omitting it
        # changes the compiled geom_rgba to the MuJoCo default (0.5 0.5 0.5 1)
        # which, despite contype=0, destabilises the contact solver for nearby
        # collision geoms.  The material tint overrides rgba for rendering.
        vis_geom = new_geom(
            name="go_board_surface_visual",
            type="box",
            size=[board_half, board_half, board_thickness],
            pos=board_center_local,
            group=1,
            rgba=[0.74, 0.62, 0.46, 1.0],
            contype="0",
            conaffinity="0",
        )
        vis_geom.set("material", "go_board_mat")
        table_body.append(vis_geom)
        # Collision board surface (contact params set at runtime in _patch_physics_params)
        table_body.append(
            new_geom(
                name="go_board_surface_collision",
                type="box",
                size=[board_half, board_half, board_thickness],
                pos=board_center_local,
                group=2,
                rgba=[0.74, 0.62, 0.46, 0.0],  # invisible
            )
        )

        # Invisible line geoms — the grid lines are now painted on via the
        # board texture, but these geoms must remain in the model because
        # MuJoCo's broadphase uses their bounding volumes to stabilise the
        # contact graph for the overlapping collision surface.
        line_half = 0.0012
        line_h = 0.001
        half = 0.5 * float(self.board_size - 1) * self.board_spacing
        line_z = table_half_h + (2.0 * board_thickness) + line_h
        for i in range(self.board_size):
            delta = -half + float(i) * self.board_spacing
            table_body.append(
                new_geom(
                    name=f"go_line_col_{i}",
                    type="box",
                    size=[line_half, half, line_h],
                    pos=[
                        self.board_center_xy[0] + delta,
                        self.board_center_xy[1],
                        line_z,
                    ],
                    group=2,
                    rgba=[0.0, 0.0, 0.0, 0.0],
                    contype="0",
                    conaffinity="0",
                )
            )
            table_body.append(
                new_geom(
                    name=f"go_line_row_{i}",
                    type="box",
                    size=[half, line_half, line_h],
                    pos=[
                        self.board_center_xy[0],
                        self.board_center_xy[1] + delta,
                        line_z,
                    ],
                    group=2,
                    rgba=[0.0, 0.0, 0.0, 0.0],
                    contype="0",
                    conaffinity="0",
                )
            )


    def _setup_references(self):
        super()._setup_references()
        self._stone_joint_names = [obj.joints[0] for obj in self._stone_objects]
        self._stone_body_ids = [self.sim.model.body_name2id(obj.root_body) for obj in self._stone_objects]

        # Cache default geom positions, RGBA, and lighting for absolute randomization
        model = self.sim.model
        board_geom_names = ["go_board_surface_visual", "go_board_surface_collision"]
        for i in range(self.board_size):
            board_geom_names.append(f"go_line_col_{i}")
            board_geom_names.append(f"go_line_row_{i}")
        import mujoco as _mj
        self._board_mat_id = _mj.mj_name2id(
            model._model, _mj.mjtObj.mjOBJ_MATERIAL, "go_board_mat"
        )
        self._default_board_mat_rgba = model.mat_rgba[self._board_mat_id].copy()
        self._default_board_mat_shininess = float(model.mat_shininess[self._board_mat_id])
        self._default_board_mat_specular = float(model.mat_specular[self._board_mat_id])

        # Cache table material IDs and geom ID for texture-swap randomization
        self._table_mat_ids = []
        self._default_table_mat_rgba = {}
        self._default_table_mat_shininess = {}
        self._default_table_mat_specular = {}
        for mat_name in self._table_mat_names:
            mid = _mj.mj_name2id(model._model, _mj.mjtObj.mjOBJ_MATERIAL, mat_name)
            self._table_mat_ids.append(mid)
            self._default_table_mat_rgba[mid] = model.mat_rgba[mid].copy()
            self._default_table_mat_shininess[mid] = float(model.mat_shininess[mid])
            self._default_table_mat_specular[mid] = float(model.mat_specular[mid])
        self._table_visual_gid = model.geom_name2id("table_visual")

        # Clean up the board texture temp file (MuJoCo has already read it)
        import os
        try:
            os.unlink(self._board_tex_tmpfile.name)
        except OSError:
            pass
        self._default_geom_pos = {}
        self._default_geom_rgba = {}
        self._default_geom_quat = {}
        for name in board_geom_names:
            try:
                gid = model.geom_name2id(name)
                self._default_geom_pos[name] = model.geom_pos[gid].copy()
                self._default_geom_rgba[name] = model.geom_rgba[gid].copy()
                self._default_geom_quat[name] = model.geom_quat[gid].copy()
            except Exception:
                pass
        self._default_light_pos = model.light_pos.copy()
        self._default_light_dir = model.light_dir.copy()
        self._default_light_diffuse = model.light_diffuse.copy()
        self._default_light_ambient = model.light_ambient.copy()
        self._default_light_specular = model.light_specular.copy()
        try:
            self._default_headlight_diffuse = model.vis.headlight.diffuse.copy()
            self._default_headlight_ambient = model.vis.headlight.ambient.copy()
        except Exception:
            self._default_headlight_diffuse = None
            self._default_headlight_ambient = None

        # Cache camera defaults for FOV and angle perturbation
        cam_name = self.camera_names[0] if hasattr(self, "camera_names") and self.camera_names else "agentview"
        try:
            cam_id = model.camera_name2id(cam_name)
            self._default_cam_fovy = float(model.cam_fovy[cam_id])
            self._default_cam_quat = model.cam_quat[cam_id].copy()
            self._cam_id = cam_id
        except Exception:
            self._default_cam_fovy = 45.0
            self._default_cam_quat = np.array([1.0, 0.0, 0.0, 0.0])
            self._cam_id = None

        # Cache extra light ID
        try:
            self._extra_light_id = model.light_name2id("extra_perturbation_light")
        except Exception:
            self._extra_light_id = None

        # Assign materials to stone geoms and cache defaults
        self._stone_geom_ids = []
        self._stone_default_rgba = {}
        for obj in self._stone_objects:
            body_id = model.body_name2id(obj.root_body)
            for gid in range(model.ngeom):
                if (model.geom_bodyid[gid] == body_id) and (model.geom_group[gid] == 1):
                    self._stone_geom_ids.append(gid)
                    self._stone_default_rgba[gid] = model.geom_rgba[gid].copy()
                    mat_name = "stone_mat_white" if "white" in obj.name else "stone_mat_black"
                    try:
                        mat_id = model.mat_name2id(mat_name)
                        model.geom_matid[gid] = mat_id
                    except Exception:
                        pass

        # Cache material defaults for stone material randomization
        self._stone_mat_ids = set()
        self._default_mat_shininess = {}
        self._default_mat_specular = {}
        for mat_name in ["stone_mat_white", "stone_mat_black"]:
            try:
                mid = model.mat_name2id(mat_name)
                self._stone_mat_ids.add(mid)
                self._default_mat_shininess[mid] = float(model.mat_shininess[mid])
                self._default_mat_specular[mid] = float(model.mat_specular[mid])
            except Exception:
                pass


    def _reset_internal(self):
        super()._reset_internal()
        self._patch_physics_params()
        self._randomize_board_position(rng=self.rng)
        self._randomize_lighting(rng=self.rng)
        self._randomize_camera(rng=self.rng)
        self._randomize_stone_material(rng=self.rng)
        self._randomize_table_material(rng=self.rng)
        for stone_idx in range(len(self._stone_objects)):
            self.hide_stone(stone_idx)
        self.sim.forward()

    def _patch_physics_params(self):
        """Patch compiled MuJoCo model for stable stone-on-board contacts.

        Root cause of stone jitter: solref[0] (contact time constant) equals
        the simulation timestep (0.002 s).  For 8 g stones this creates
        contact-force overshoot each step, causing z-oscillation (27 sign
        changes in 50 steps) and lateral drift (~26 mm over 300 arm-moving
        steps with the Euler integrator).

        Fix: use MuJoCo's implicit integrator (integrator=2) which damps
        high-frequency contact oscillations without softening contacts.
        Reduces arm-moving drift from 26 mm to ~1.4 mm.
        """
        model = self.sim.model

        # --- Integrator: implicit for stable light-object contacts ---
        model.opt.integrator = 2  # mjtIntegrator.mjINT_IMPLICIT

        # --- Solver options ---
        model.opt.timestep = 0.0005
        # Sync cached timestep so robosuite's substep loop
        # (control_timestep / model_timestep) uses the new value.
        self.model_timestep = model.opt.timestep
        model.opt.solver = 2  # mjtSolver.mjSOL_NEWTON
        model.opt.iterations = 500
        model.opt.tolerance = 1e-10
        model.opt.noslip_iterations = 100
        model.opt.noslip_tolerance = 1e-9

        # --- Stone geom contact parameters ---
        for obj in self._stone_objects:
            body_name = obj.root_body
            body_id = model.body_name2id(body_name)
            for geom_id in range(model.ngeom):
                if model.geom_bodyid[geom_id] == body_id:
                    model.geom_condim[geom_id] = 4
                    model.geom_friction[geom_id] = [1.5, 0.0698, 0.0785]
                    model.geom_solref[geom_id] = [0.002, 1.0]
                    model.geom_solimp[geom_id] = [0.9932, 0.9932, 0.001, 0.5, 2.0]
                    model.geom_margin[geom_id] = 0.0002
                    model.geom_gap[geom_id] = 0.0

        # --- Board collision surface ---
        try:
            board_geom_id = model.geom_name2id("go_board_surface_collision")
            model.geom_condim[board_geom_id] = 4
            model.geom_friction[board_geom_id] = [1.5, 0.0698, 0.0785]
            model.geom_solref[board_geom_id] = [0.002, 1.0]
            model.geom_solimp[board_geom_id] = [0.9932, 0.9932, 0.001, 0.5, 2.0]
        except Exception:
            pass

        # --- Table surface ---
        try:
            for name in ["table_collision", "table_visual"]:
                gid = model.geom_name2id(name)
                model.geom_condim[gid] = 4
                model.geom_friction[gid] = [1.5, 0.0698, 0.0785]
                model.geom_solref[gid] = [0.002, 1.0]
                model.geom_solimp[gid] = [0.9932, 0.9932, 0.001, 0.5, 2.0]
        except Exception:
            pass

        # --- Joint damping on stone free joints ---
        for jnt_name in self._stone_joint_names:
            jnt_id = model.joint_name2id(jnt_name)
            dof_start = model.jnt_dofadr[jnt_id]
            for i in range(6):
                model.dof_damping[dof_start + i] = 0.1

    def _randomize_board_position(self, rng: np.random.RandomState) -> None:
        """Shift the board ±1.5cm XY and optionally perturb surface color (absolute, not incremental)."""
        shift_xy = rng.uniform(-self._board_shift_range, self._board_shift_range, size=(2,)).astype(np.float32)
        self.board_center_xy = self._board_center_xy_default + shift_xy

        model = self.sim.model
        for name, default_pos in self._default_geom_pos.items():
            try:
                gid = model.geom_name2id(name)
                model.geom_pos[gid, 0] = default_pos[0] + float(shift_xy[0])
                model.geom_pos[gid, 1] = default_pos[1] + float(shift_xy[1])
            except Exception:
                pass

        # Board rotation ±5deg around Z
        rotation_rad = float(rng.uniform(np.radians(-5.0), np.radians(5.0)))
        self._board_rotation_rad = rotation_rad
        cos_r = np.cos(rotation_rad)
        sin_r = np.sin(rotation_rad)
        board_cx = float(self.board_center_xy[0])
        board_cy = float(self.board_center_xy[1])

        for name in self._default_geom_pos.keys():
            try:
                gid = model.geom_name2id(name)
                # Rotate position around board center
                px = model.geom_pos[gid, 0] - board_cx
                py = model.geom_pos[gid, 1] - board_cy
                model.geom_pos[gid, 0] = cos_r * px - sin_r * py + board_cx
                model.geom_pos[gid, 1] = sin_r * px + cos_r * py + board_cy
                # Rotate quaternion for line geoms and collision surface
                if "line" in name or "collision" in name or "visual" in name:
                    default_quat = self._default_geom_quat.get(name)
                    if default_quat is not None:
                        z_rot_quat = np.array([
                            np.cos(rotation_rad * 0.5), 0.0, 0.0, np.sin(rotation_rad * 0.5),
                        ], dtype=np.float64)
                        model.geom_quat[gid] = self._quat_mul(z_rot_quat, default_quat)
            except Exception:
                pass

        # Perturb board surface color, shininess, and specular
        try:
            color_perturb = rng.uniform(-0.06, 0.06, size=(3,))
            model.mat_rgba[self._board_mat_id, :3] = np.clip(
                self._default_board_mat_rgba[:3] + color_perturb, 0.0, 1.0
            )
            model.mat_shininess[self._board_mat_id] = np.clip(
                self._default_board_mat_shininess + rng.uniform(-0.15, 0.15), 0.02, 0.6
            )
            model.mat_specular[self._board_mat_id] = np.clip(
                self._default_board_mat_specular + rng.uniform(-0.2, 0.2), 0.05, 0.7
            )
        except Exception:
            pass

    def _randomize_lighting(self, rng: np.random.RandomState) -> None:
        """Perturb light positions, directions, and intensities from defaults (absolute)."""
        model = self.sim.model
        extra_lid = getattr(self, "_extra_light_id", None)
        for light_id in range(model.nlight):
            if light_id == extra_lid:
                continue
            model.light_pos[light_id] = self._default_light_pos[light_id] + rng.uniform(-0.3, 0.3, size=(3,))
            model.light_dir[light_id] = self._default_light_dir[light_id] + rng.uniform(-0.15, 0.15, size=(3,))
            model.light_diffuse[light_id] = np.clip(
                self._default_light_diffuse[light_id] + rng.uniform(-0.15, 0.15, size=(3,)),
                0.1, 1.0,
            )
            model.light_ambient[light_id] = np.clip(
                self._default_light_ambient[light_id] + rng.uniform(-0.08, 0.08, size=(3,)),
                0.0, 0.5,
            )
            model.light_specular[light_id] = np.clip(
                self._default_light_specular[light_id] + rng.uniform(-0.1, 0.1, size=(3,)),
                0.0, 1.0,
            )
        # Perturb headlight from defaults
        try:
            if self._default_headlight_diffuse is not None:
                model.vis.headlight.diffuse[:] = np.clip(
                    self._default_headlight_diffuse + rng.uniform(-0.1, 0.1, size=(3,)),
                    0.0, 1.0,
                )
            if self._default_headlight_ambient is not None:
                model.vis.headlight.ambient[:] = np.clip(
                    self._default_headlight_ambient + rng.uniform(-0.05, 0.05, size=(3,)),
                    0.0, 0.5,
                )
        except Exception:
            pass

        # Extra perturbation light: P=0.4 activation
        if extra_lid is not None:
            if rng.random() < 0.4:
                model.light_pos[extra_lid] = rng.uniform(-0.5, 0.5, size=(3,))
                model.light_pos[extra_lid, 2] = max(1.0, model.light_pos[extra_lid, 2] + 1.5)
                intensity = float(rng.uniform(0.15, 0.5))
                model.light_diffuse[extra_lid] = np.full(3, intensity)
                model.light_specular[extra_lid] = np.full(3, intensity * 0.5)
            else:
                model.light_diffuse[extra_lid] = np.zeros(3)
                model.light_specular[extra_lid] = np.zeros(3)

    @staticmethod
    def _quat_mul(q1, q2):
        """Multiply two quaternions (wxyz convention)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float64)

    @staticmethod
    def _euler_to_quat(euler_rad):
        """Convert Euler angles (roll, pitch, yaw) to quaternion (wxyz)."""
        cx, cy, cz = np.cos(np.asarray(euler_rad) * 0.5)
        sx, sy, sz = np.sin(np.asarray(euler_rad) * 0.5)
        return np.array([
            cx*cy*cz + sx*sy*sz,
            sx*cy*cz - cx*sy*sz,
            cx*sy*cz + sx*cy*sz,
            cx*cy*sz - sx*sy*cz,
        ], dtype=np.float64)

    def _randomize_camera(self, rng: np.random.RandomState) -> None:
        """Perturb camera FOV (±15%) and angle (±5deg)."""
        if getattr(self, "_cam_id", None) is None:
            return
        model = self.sim.model
        # FOV variance ±10%
        model.cam_fovy[self._cam_id] = self._default_cam_fovy * (1.0 + rng.uniform(-0.10, 0.10))
        # Camera angle ±5deg
        euler_deg = rng.uniform(-5.0, 5.0, size=(3,))
        euler_rad = np.radians(euler_deg)
        dq = self._euler_to_quat(euler_rad)
        new_quat = self._quat_mul(self._default_cam_quat, dq)
        model.cam_quat[self._cam_id] = new_quat

    def _randomize_stone_material(self, rng: np.random.RandomState) -> None:
        """Perturb stone shininess, specular, and color."""
        model = self.sim.model
        for mid in getattr(self, "_stone_mat_ids", set()):
            default_shin = self._default_mat_shininess[mid]
            default_spec = self._default_mat_specular[mid]
            model.mat_shininess[mid] = np.clip(default_shin + rng.uniform(-0.2, 0.2), 0.05, 0.95)
            model.mat_specular[mid] = np.clip(default_spec + rng.uniform(-0.3, 0.3), 0.1, 0.9)
        for gid, default_rgba in getattr(self, "_stone_default_rgba", {}).items():
            color_noise = rng.uniform(-0.05, 0.05, size=(3,))
            model.geom_rgba[gid, :3] = np.clip(default_rgba[:3] + color_noise, 0.0, 1.0)

    def _randomize_table_material(self, rng: np.random.RandomState) -> None:
        """Swap table material (texture) and perturb tint, shininess, specular."""
        model = self.sim.model
        # Pick a random material from the pool and assign it to the table geom
        idx = int(rng.integers(len(self._table_mat_ids)))
        mid = self._table_mat_ids[idx]
        model.geom_matid[self._table_visual_gid] = mid
        # Slight color tint ±0.08 RGB
        color_perturb = rng.uniform(-0.08, 0.08, size=(3,))
        model.mat_rgba[mid, :3] = np.clip(
            self._default_table_mat_rgba[mid][:3] + color_perturb, 0.0, 1.0
        )
        # Shininess and specular variation
        model.mat_shininess[mid] = np.clip(
            self._default_table_mat_shininess[mid] + rng.uniform(-0.15, 0.15), 0.02, 0.5
        )
        model.mat_specular[mid] = np.clip(
            self._default_table_mat_specular[mid] + rng.uniform(-0.2, 0.2), 0.05, 0.6
        )

    def _check_success(self):
        return False

    def _offboard_slot_xyz(self, slot_idx: int) -> np.ndarray:
        slot_idx = int(slot_idx)
        row = slot_idx // 8
        col = slot_idx % 8
        x0 = self.board_center_xy[0] + 0.24
        y0 = self.board_center_xy[1] - 0.26
        x = x0 + 0.026 * float(col)
        y = y0 + 0.026 * float(row)
        z = self.table_top_z + self.collision_stone_half_height + self._table_stone_clearance
        return np.array([x, y, z], dtype=np.float32)

    def set_stone_pose(self, stone_idx: int, pos: np.ndarray, quat_wxyz: Optional[np.ndarray] = None) -> None:
        stone_idx = int(stone_idx)
        if quat_wxyz is None:
            quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        qpos = np.concatenate([np.asarray(pos, dtype=np.float64), np.asarray(quat_wxyz, dtype=np.float64)], axis=0)
        self.sim.data.set_joint_qpos(self._stone_joint_names[stone_idx], qpos)
        self.sim.data.set_joint_qvel(self._stone_joint_names[stone_idx], np.zeros((6,), dtype=np.float64))

    def hide_stone(self, stone_idx: int) -> None:
        self.set_stone_pose(stone_idx=stone_idx, pos=self._offboard_slot_xyz(stone_idx))

    def get_stone_pos(self, stone_idx: int) -> np.ndarray:
        return np.array(self.sim.data.body_xpos[self._stone_body_ids[int(stone_idx)]], dtype=np.float32)

    def check_stone_grasped(self, stone_idx: int) -> bool:
        return bool(
            self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self._stone_objects[int(stone_idx)],
            )
        )

    def get_eef_pose(self) -> np.ndarray:
        robot = self.robots[0]
        arm = robot.arms[0] if len(robot.arms) > 0 else "right"
        site_id = int(robot.eef_site_id[arm])
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = np.asarray(self.sim.data.site_xpos[site_id], dtype=np.float32)
        pose[:3, :3] = np.asarray(self.sim.data.site_xmat[site_id], dtype=np.float32).reshape(3, 3)
        return pose

    def render_rgb(self, height: int, width: int, camera_name: Optional[str] = None) -> np.ndarray:
        camera = self.camera_names[0] if camera_name is None else camera_name
        return self.sim.render(
            camera_name=camera,
            width=int(width),
            height=int(height),
        )[::-1]


class GoRobosuiteBenchmarkEnv:
    """
    MimicGen-friendly wrapper around a robosuite Go-like 5x5 rigid-stone task.

    This backend keeps the benchmark API from GoJacoBenchmarkEnv while making
    move commits depend on a physically placed rigid stone.
    """

    def __init__(
        self,
        seed: int = 0,
        environment_name: str = "robosuite_go_5x5_rigid_bodies",
        include_image_obs: bool = True,
        camera_height: int = 84,
        camera_width: int = 84,
        action_scale: float = 0.03,
        hover_height: float = 0.16,
        press_height: float = 0.03,
        reach_xy_threshold: float = 0.03,
        max_steps: int = 500,
        success_hold_steps: int = 0,
        drive_physical_arm: bool = True,
        enable_opponent_moves: bool = False,
        opening_with_opponent: bool = True,
        render_carried_stone: bool = True,
        render_eef_overlay: bool = True,
        eef_overlay_trail: int = 10,
        gnugo_path: Optional[str] = None,
        robot: str = "Panda",
        gripper_types: str = "default",
    ):
        del drive_physical_arm
        del render_carried_stone
        del gnugo_path
        self.seed = int(seed)
        self._rng = np.random.RandomState(self.seed)
        self.environment_name = str(environment_name)

        controller_config = self._make_controller_config(robot=robot)
        self._rs_env = _Go5x5RigidRobosuite(
            robots=robot,
            controller_configs=controller_config,
            gripper_types=gripper_types,
            initialization_noise={"magnitude": 0.08, "type": "uniform"},
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="agentview",
            camera_names="agentview",
            camera_heights=int(max(camera_height, 84)),
            camera_widths=int(max(camera_width, 84)),
            horizon=max(200, int(max_steps) + 80),
            ignore_done=True,
            hard_reset=True,
            renderer="mujoco",
            seed=self.seed,
        )

        self.include_image_obs = bool(include_image_obs)
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)

        self.action_scale = float(action_scale)
        self.reach_xy_threshold = float(reach_xy_threshold)
        self.max_steps = int(max_steps)
        self.success_hold_steps = max(0, int(success_hold_steps))
        self.enable_opponent_moves = bool(enable_opponent_moves)
        self.opening_with_opponent = bool(opening_with_opponent)
        self.render_eef_overlay = bool(render_eef_overlay)
        self.eef_overlay_trail = max(0, int(eef_overlay_trail))

        self.board_size = int(self._rs_env.board_size)
        self._logic = _OpenSpielGoLogic(board_size=self.board_size)
        self._intersection_xyz = self._rs_env.board_intersections_xyz.copy()
        self._source_xyz = self._rs_env.source_stone_xyz.copy()
        self.table_top_z = float(self._rs_env.table_top_z)
        self.board_surface_z = float(self._rs_env.board_surface_z)
        self.stone_height = float(self._rs_env.stone_height)
        self.board_thickness = 0.0025
        self.hover_height = self.table_top_z + float(hover_height)
        self.press_height = self.table_top_z + float(press_height)

        self.place_xy_threshold = 0.026
        self.place_z_threshold = 0.018

        self.workspace_low, self.workspace_high = self._compute_workspace_bounds()
        self._robot = self._rs_env.robots[0]
        self._arm_key, self._arm_dim, self._gripper_key, self._gripper_dim = self._resolve_action_keys()
        self._controller_xyz_max = self._infer_controller_xyz_max()

        self._target_rc: Tuple[int, int] = (0, 0)
        self._target_pose = np.eye(4, dtype=np.float32)
        self._gripper_action = np.zeros((1,), dtype=np.float32)
        self._eef_trail: List[np.ndarray] = []

        self._available_stones: Dict[int, List[int]] = {
            SELF: [],
            OPPONENT: [],
        }
        self._stone_assignments: Dict[Tuple[int, int, int], int] = {}
        self._active_white_stone_idx: Optional[int] = None
        self._stone_color: str = "white"

        self._step_count = 0
        self._move_committed = False
        self._success_step: Optional[int] = None
        self._queued_reset_options: Optional[GoResetOptions] = None
        self._wb_shift = np.zeros(3, dtype=np.float32)

        self.reset()

    @staticmethod
    def _make_controller_config(robot: str) -> dict:
        return load_composite_controller_config(robot=robot)

    def _resolve_action_keys(self) -> Tuple[str, int, Optional[str], int]:
        action_splits = dict(self._robot._action_split_indexes)
        primary_arm = self._robot.arms[0] if len(self._robot.arms) > 0 else "right"
        arm_key = primary_arm if primary_arm in action_splits else None
        arm_dim = 0
        if arm_key is not None:
            arm_dim = int(action_splits[arm_key][1] - action_splits[arm_key][0])
        primary_gripper_key = self._robot.get_gripper_name(primary_arm)
        gripper_key = primary_gripper_key if primary_gripper_key in action_splits else None
        gripper_dim = 0
        if gripper_key is not None:
            gripper_dim = int(action_splits[gripper_key][1] - action_splits[gripper_key][0])
        for key, (start, end) in action_splits.items():
            dim = int(end - start)
            if dim <= 0:
                continue
            if ("gripper" in key) and (gripper_key is None):
                gripper_key = key
                gripper_dim = dim
            elif arm_key is None:
                arm_key = key
                arm_dim = dim
        if arm_key is None:
            raise RuntimeError("Could not resolve robosuite arm action key")
        return arm_key, arm_dim, gripper_key, gripper_dim

    def _infer_controller_xyz_max(self) -> np.ndarray:
        try:
            part_ctrl = self._robot.part_controllers.get(self._arm_key, None)
            if (part_ctrl is not None) and hasattr(part_ctrl, "output_max"):
                output_max = np.asarray(part_ctrl.output_max, dtype=np.float32).reshape(-1)
                if output_max.size >= 3:
                    return np.maximum(output_max[:3], 1e-4)
        except Exception:
            pass
        return np.ones((3,), dtype=np.float32)

    @property
    def base_env(self):
        return self

    @property
    def physics(self):
        return self._rs_env.sim

    def _compute_workspace_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        board_xy = self._intersection_xyz[:, :, :2].reshape(-1, 2)
        source_xy = self._source_xyz[:2].reshape(1, 2)
        full_xy = np.concatenate([board_xy, source_xy], axis=0)
        low_xy = full_xy.min(axis=0) - np.array([0.16, 0.16], dtype=np.float32)
        high_xy = full_xy.max(axis=0) + np.array([0.16, 0.16], dtype=np.float32)
        low = np.array([low_xy[0], low_xy[1], self.table_top_z], dtype=np.float32)
        high = np.array([high_xy[0], high_xy[1], self.table_top_z + 0.38], dtype=np.float32)
        return low, high

    @staticmethod
    def _pose_from_xyz(xyz: np.ndarray) -> np.ndarray:
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = np.asarray(xyz, dtype=np.float32)
        return pose

    def _decode_action_int(self, action_int: int) -> Tuple[int, int, bool]:
        pass_id = self.board_size * self.board_size
        if int(action_int) == pass_id:
            return -1, -1, True
        row = int(action_int // self.board_size)
        col = int(action_int % self.board_size)
        return row, col, False

    def _player_channel(self, player_id: int) -> int:
        return 1 if int(player_id) == SELF else 2

    def _choose_random_legal_action(self, exclude_pass: bool = True) -> Optional[int]:
        legal_actions = self._logic.legal_actions()
        if exclude_pass:
            pass_id = self.board_size * self.board_size
            legal_actions = [a for a in legal_actions if int(a) != pass_id]
        if not legal_actions:
            return None
        return int(self._rng.choice(np.asarray(legal_actions, dtype=np.int32)))

    def _reserve_next_stone(self, player_id: int) -> Optional[int]:
        pool = self._available_stones[int(player_id)]
        if len(pool) == 0:
            return None
        return int(pool.pop(0))

    def _release_stone(self, player_id: int, stone_idx: int) -> None:
        pool = self._available_stones[int(player_id)]
        if int(stone_idx) not in pool:
            pool.append(int(stone_idx))
        self._rs_env.hide_stone(int(stone_idx))

    def _place_stone_at_intersection(self, stone_idx: int, row: int, col: int) -> None:
        xyz = self._intersection_xyz[int(row), int(col)].copy()
        xyz[2] += self._rs_env._board_stone_clearance
        self._rs_env.set_stone_pose(stone_idx=int(stone_idx), pos=xyz)


    def _sync_captures(self, prev_board: np.ndarray, new_board: np.ndarray) -> None:
        for player_id in (SELF, OPPONENT):
            channel = self._player_channel(player_id)
            removed = np.logical_and(
                prev_board[:, :, channel].astype(bool),
                np.logical_not(new_board[:, :, channel].astype(bool)),
            )
            removed_rc = np.transpose(np.nonzero(removed))
            for row, col in removed_rc:
                key = (int(player_id), int(row), int(col))
                stone_idx = self._stone_assignments.pop(key, None)
                if stone_idx is not None:
                    self._release_stone(player_id=player_id, stone_idx=int(stone_idx))

    def _apply_single_player_move(
        self,
        player_id: int,
        action_int: int,
        use_active_white_stone: bool = False,
    ) -> bool:
        row, col, is_pass = self._decode_action_int(action_int)
        prev_board = self._logic.get_board_state().astype(np.float32)
        valid = self._logic.apply(player=player_id, action_int=action_int)
        if not valid:
            return False
        new_board = self._logic.get_board_state().astype(np.float32)
        self._sync_captures(prev_board=prev_board, new_board=new_board)

        if not is_pass:
            if use_active_white_stone and (self._active_white_stone_idx is not None):
                stone_idx = int(self._active_white_stone_idx)
            else:
                stone_idx = self._reserve_next_stone(player_id=player_id)
                if stone_idx is None:
                    return False
            if not use_active_white_stone:
                # Only teleport stones that are being placed programmatically
                # (e.g. opening moves, opponent moves). For the active stone
                # being carried by the gripper, leave it at its current
                # physical position so the arm can place it naturally.
                self._place_stone_at_intersection(stone_idx=stone_idx, row=row, col=col)
            self._stone_assignments[(int(player_id), int(row), int(col))] = int(stone_idx)
            if use_active_white_stone:
                self._active_white_stone_idx = None

        self._rs_env.sim.forward()
        return True

    def _apply_go_action(self, action_int: int, apply_opponent: Optional[bool] = None) -> bool:
        if apply_opponent is None:
            apply_opponent = self.enable_opponent_moves

        valid = self._apply_single_player_move(
            player_id=SELF,
            action_int=int(action_int),
            use_active_white_stone=True,
        )
        if not valid:
            return False

        if bool(apply_opponent) and (not self._logic.is_game_over):
            opp_action = self._choose_random_legal_action(exclude_pass=False)
            if opp_action is not None:
                self._apply_single_player_move(
                    player_id=OPPONENT,
                    action_int=int(opp_action),
                    use_active_white_stone=False,
                )
        return True

    def _spawn_active_stone(self) -> None:
        if self._active_white_stone_idx is not None:
            return
        # Pick from the correct color pool
        if self._stone_color == "black":
            pool = self._available_stones[OPPONENT]
            if not pool:
                self._active_white_stone_idx = None
                return
            stone_idx = int(pool.pop(0))
        else:
            stone_idx = self._reserve_next_stone(player_id=SELF)
            if stone_idx is None:
                self._active_white_stone_idx = None
                return
        self._active_white_stone_idx = int(stone_idx)
        spawn_xyz = self._source_xyz.copy()
        self._rs_env.set_stone_pose(stone_idx=self._active_white_stone_idx, pos=spawn_xyz)
        self._rs_env.sim.forward()

    def _settle_stones(self, num_steps: int = 500) -> None:
        """Run physics steps with zero action to let stones settle.

        Uses full env.step with zero action so the arm controller holds
        position (prevents arm from falling and shaking the table).
        Then zeros all stone velocities to eliminate residual drift.
        """
        zero_action = np.zeros(self._rs_env.action_dim, dtype=np.float32)
        for _ in range(num_steps):
            self._rs_env.step(zero_action)
        # Zero out all stone velocities after settling
        for jnt_name in self._rs_env._stone_joint_names:
            self._rs_env.sim.data.set_joint_qvel(jnt_name, np.zeros(6, dtype=np.float64))
        self._rs_env.sim.forward()

    def _commit_target_move_fallback(self) -> bool:
        """Force-commit the selected target move when EEF is correctly pressing target."""
        if self._active_white_stone_idx is not None:
            row, col = self._target_rc
            self._place_stone_at_intersection(
                stone_idx=int(self._active_white_stone_idx),
                row=int(row),
                col=int(col),
            )
            self._rs_env.sim.forward()
        action_int = int(self._target_rc[0] * self.board_size + self._target_rc[1])
        return bool(self._apply_go_action(action_int=action_int))

    def seed_random_opening(self, opening_moves: int) -> int:
        opening_moves = max(0, int(opening_moves))
        applied = 0
        for _ in range(opening_moves):
            if self._logic.is_game_over:
                break
            action_int = self._choose_random_legal_action(exclude_pass=True)
            if action_int is None:
                break
            ok = self._apply_single_player_move(
                player_id=SELF,
                action_int=int(action_int),
                use_active_white_stone=False,
            )
            if not ok:
                break
            applied += 1
            if self.opening_with_opponent and (not self._logic.is_game_over):
                opp_action = self._choose_random_legal_action(exclude_pass=False)
                if opp_action is not None:
                    self._apply_single_player_move(
                        player_id=OPPONENT,
                        action_int=int(opp_action),
                        use_active_white_stone=False,
                    )
        return applied

    def replay_opening_move_history(self, move_history: np.ndarray) -> int:
        history = np.asarray(move_history, dtype=np.int32).reshape(-1)
        applied = 0
        for action_int in history.tolist():
            if int(action_int) < 0 or self._logic.is_game_over:
                break
            if not self._apply_single_player_move(
                player_id=self._logic.current_player,
                action_int=int(action_int),
                use_active_white_stone=False,
            ):
                break
            applied += 1
        return applied

    def _set_target_pose_from_rc(self, row: int, col: int) -> None:
        target_xyz = self._intersection_xyz[row, col].copy()
        self._target_pose = self._pose_from_xyz(target_xyz)
        self._target_rc = (int(row), int(col))

    def set_target_intersection(self, row: int, col: int) -> None:
        if not (0 <= int(row) < self.board_size and 0 <= int(col) < self.board_size):
            raise ValueError(f"target ({row}, {col}) outside board")
        self._set_target_pose_from_rc(row=int(row), col=int(col))

    def set_target_from_random_legal_move(self) -> Tuple[int, int]:
        action_int = self._choose_random_legal_action(exclude_pass=True)
        if action_int is None:
            self.set_target_intersection(row=self.board_size // 2, col=self.board_size // 2)
            return self._target_rc
        row, col, _ = self._decode_action_int(action_int)
        self.set_target_intersection(row=row, col=col)
        return self._target_rc

    def reset(self, options: Optional[GoResetOptions] = None):
        if options is None and self._queued_reset_options is not None:
            options = self._queued_reset_options
            self._queued_reset_options = None
        if options is None:
            options = GoResetOptions()

        self._rs_env.reset()
        self._logic.reset()
        # board_intersections_xyz reads from geom_xpos, so XY shift +
        # Z-rotation + body transform are already baked in.
        self._intersection_xyz = self._rs_env.board_intersections_xyz.copy()
        self._source_xyz = self._rs_env.source_stone_xyz.copy()
        self.board_surface_z = float(self._rs_env.board_surface_z)

        # White balance shift for color augmentation
        self._wb_shift = self._rng.uniform(-0.08, 0.08, size=(3,)).astype(np.float32)

        self.workspace_low, self.workspace_high = self._compute_workspace_bounds()
        self._available_stones = {
            SELF: self._rs_env.white_stone_indices,
            OPPONENT: self._rs_env.black_stone_indices,
        }
        self._stone_assignments = {}
        self._active_white_stone_idx = None
        self._gripper_action = np.zeros((1,), dtype=np.float32)

        self._step_count = 0
        self._move_committed = False
        self._success_step = None
        self._committed_stone_idx = None

        # Choose stone color
        if options.stone_color is not None:
            self._stone_color = options.stone_color
        else:
            self._stone_color = self._rng.choice(["black", "white"])

        if options.opening_move_history is not None:
            self.replay_opening_move_history(options.opening_move_history)
        else:
            self.seed_random_opening(options.opening_moves)
        # Settle stones after opening placement; high joint damping (0.1)
        # and contact params allow fast convergence.
        self._settle_stones(num_steps=200)
        if options.target_row is not None and options.target_col is not None:
            self.set_target_intersection(row=int(options.target_row), col=int(options.target_col))
        else:
            self.set_target_from_random_legal_move()

        # Randomize source stone position near EEF
        eef_xyz = self.get_eef_pose()[:3, 3]
        stone_offset = self._rng.uniform(-0.015, 0.015, size=(2,))
        source_xy = eef_xyz[:2] + stone_offset
        source_xy = np.clip(source_xy, self.workspace_low[:2], self.workspace_high[:2])
        self._source_xyz = np.array(
            [source_xy[0], source_xy[1],
             self._rs_env.table_top_z
             + self._rs_env.collision_stone_half_height
             + self._rs_env._table_stone_clearance],
            dtype=np.float32,
        )

        self._spawn_active_stone()
        self._settle_stones(num_steps=100)
        self._eef_trail = [self.get_eef_pose()[:3, 3].copy()]
        return self.get_observation()

    def queue_reset_options(self, options: GoResetOptions) -> None:
        self._queued_reset_options = GoResetOptions(
            opening_moves=int(options.opening_moves),
            opening_move_history=None
            if options.opening_move_history is None
            else np.asarray(options.opening_move_history, dtype=np.int32).copy(),
            target_row=options.target_row,
            target_col=options.target_col,
            stone_color=options.stone_color,
        )

    def _nearest_intersection(self, xy: np.ndarray) -> Tuple[int, int, float]:
        flat_xy = self._intersection_xyz[:, :, :2].reshape(-1, 2)
        dists = np.linalg.norm(flat_xy - xy.reshape(1, 2), axis=1)
        flat_idx = int(np.argmin(dists))
        row = flat_idx // self.board_size
        col = flat_idx % self.board_size
        return row, col, float(dists[flat_idx])

    def _is_active_stone_grasped(self) -> bool:
        if self._active_white_stone_idx is None:
            return False
        return bool(self._rs_env.check_stone_grasped(self._active_white_stone_idx))

    def is_active_stone_grasped(self) -> bool:
        return self._is_active_stone_grasped()

    def _build_low_level_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        arm_action = np.clip(action[:3], -1.0, 1.0)
        self._gripper_action = np.array([np.clip(action[3], -1.0, 1.0)], dtype=np.float32)

        delta_world = arm_action * self.action_scale
        arm_controller = self._robot.part_controllers.get(self._arm_key, None)
        input_ref_frame = getattr(arm_controller, "input_ref_frame", "world")
        if str(input_ref_frame).lower() == "base":
            # Convert world delta into base frame expected by default OSC controllers.
            delta_cmd = np.asarray(self._robot.base_ori, dtype=np.float32).T.dot(delta_world)
        else:
            delta_cmd = delta_world

        arm_vector = np.zeros((self._arm_dim,), dtype=np.float32)
        scaled_xyz = delta_cmd / np.maximum(self._controller_xyz_max, 1e-6)
        arm_vector[:3] = np.clip(scaled_xyz, -1.0, 1.0)
        if self._arm_dim > 3:
            arm_vector[3:] = 0.0

        action_dict = OrderedDict()
        action_dict[self._arm_key] = arm_vector
        if (self._gripper_key is not None) and (self._gripper_dim > 0):
            gripper_value = float(np.clip((2.0 * float(self._gripper_action[0])) - 1.0, -1.0, 1.0))
            action_dict[self._gripper_key] = np.full((self._gripper_dim,), gripper_value, dtype=np.float32)
        return self._robot.create_action_vector(action_dict)

    def reached_target(self) -> bool:
        eef_xy = self.get_eef_pose()[:2, 3]
        tgt_xy = self._target_pose[:2, 3]
        return bool(np.linalg.norm(eef_xy - tgt_xy) <= self.reach_xy_threshold)

    def get_eef_pose(self) -> np.ndarray:
        return self._rs_env.get_eef_pose().astype(np.float32)

    def get_target_pose(self) -> np.ndarray:
        return self._target_pose.copy()

    def get_target_intersection(self) -> Tuple[int, int]:
        return self._target_rc

    def get_source_stone_pose(self) -> np.ndarray:
        return self._pose_from_xyz(self._source_xyz.copy())

    def get_task_stone_pose(self) -> Optional[np.ndarray]:
        stone_idx: Optional[int] = self._active_white_stone_idx
        if stone_idx is None and self._committed_stone_idx is not None:
            stone_idx = int(self._committed_stone_idx)
        if stone_idx is None or int(stone_idx) < 0:
            return None
        return self._pose_from_xyz(self._rs_env.get_stone_pos(int(stone_idx)).copy())

    def get_board_origin_pose(self) -> np.ndarray:
        center = self._intersection_xyz.mean(axis=(0, 1))
        center[2] = self.board_surface_z
        return self._pose_from_xyz(center)

    def get_board_xy_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        flat_xy = self._intersection_xyz[:, :, :2].reshape(-1, 2)
        return flat_xy.min(axis=0).astype(np.float32), flat_xy.max(axis=0).astype(np.float32)

    def get_board_state(self) -> np.ndarray:
        return self._logic.get_board_state().astype(np.float32)

    def get_move_history(self) -> np.ndarray:
        return self._logic.get_move_history().astype(np.float32)

    def get_state(self) -> Dict[str, np.ndarray]:
        board_flat = self.get_board_state().reshape(-1)
        eef_xyz = self.get_eef_pose()[:3, 3]
        tgt_xyz = self._target_pose[:3, 3]
        state = np.concatenate(
            [
                board_flat,
                eef_xyz,
                tgt_xyz,
                np.array([float(self._move_committed), float(self._step_count)], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        return {"states": state}

    def get_observation(self) -> Dict[str, np.ndarray]:
        eef_pose = self.get_eef_pose()
        obs: Dict[str, np.ndarray] = {
            "board_state": self.get_board_state(),
            "eef_pos": eef_pose[:3, 3].copy().astype(np.float32),
            "target_pos": self._target_pose[:3, 3].copy().astype(np.float32),
            "move_history": self.get_move_history(),
        }
        if self.include_image_obs:
            image = self.render(mode="rgb_array", height=self.camera_height, width=self.camera_width)
            # Apply white balance shift before debug overlay
            wb = getattr(self, "_wb_shift", None)
            if wb is not None and np.any(np.abs(wb) > 1e-6):
                img_f = image.astype(np.float32)
                for c in range(3):
                    img_f[:, :, c] *= (1.0 + wb[c])
                image = np.clip(img_f, 0, 255).astype(np.uint8)
            if self.render_eef_overlay:
                image = self._overlay_debug_markers(image=image)
            obs["agentview_image"] = image
        return obs

    def is_success(self) -> Dict[str, bool]:
        return {"task": bool(self._move_committed)}

    def _update_stone_damping(self) -> None:
        """Set high damping on placed stones so table vibrations / contacts
        don't make them drift, and low damping on the active stone so the
        arm can carry it naturally.  Stones remain fully dynamic.

        The just-committed stone keeps low damping for 30 steps after
        commit so it can settle naturally on the board."""
        model = self._rs_env.sim.model
        PLACED_DAMPING = 20.0
        ACTIVE_DAMPING = 0.1
        SETTLE_GRACE_STEPS = 30
        placed_indices = set(self._stone_assignments.values())

        for idx, jnt_name in enumerate(self._rs_env._stone_joint_names):
            is_active = (idx == self._active_white_stone_idx)
            # Keep low damping on the committed stone for a grace period
            # after commit so it can physically settle on the board.
            is_settling = (
                idx == self._committed_stone_idx
                and self._success_step is not None
                and (self._step_count - self._success_step) < SETTLE_GRACE_STEPS
            )
            damping = ACTIVE_DAMPING if (is_active or is_settling) else (
                PLACED_DAMPING if idx in placed_indices else ACTIVE_DAMPING
            )
            jnt_id = model.joint_name2id(jnt_name)
            dof = model.jnt_dofadr[jnt_id]
            for d in range(6):
                model.dof_damping[dof + d] = damping

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 4:
            raise ValueError("GoRobosuiteBenchmarkEnv expects action shape (4,)")

        self._update_stone_damping()
        low_level_action = self._build_low_level_action(action=action)
        self._rs_env.step(low_level_action)

        self._eef_trail.append(self.get_eef_pose()[:3, 3].copy())
        if len(self._eef_trail) > self.eef_overlay_trail + 1:
            self._eef_trail = self._eef_trail[-(self.eef_overlay_trail + 1) :]

        if (not self._move_committed) and (self._active_white_stone_idx is not None):
            stone_xyz = self._rs_env.get_stone_pos(self._active_white_stone_idx)
            row, col, dist = self._nearest_intersection(stone_xyz[:2])
            near_target = ((int(row), int(col)) == self._target_rc)
            if (
                (dist <= self.place_xy_threshold)
                and (abs(float(stone_xyz[2] - self._intersection_xyz[row, col, 2])) <= self.place_z_threshold)
                and near_target
            ):
                action_int = row * self.board_size + col
                committed_stone = int(self._stone_assignments.get(
                    (SELF, int(row), int(col)),
                    self._active_white_stone_idx if self._active_white_stone_idx is not None else -1,
                ))
                self._move_committed = self._apply_go_action(action_int=int(action_int))
                if self._move_committed and (self._success_step is None):
                    self._success_step = int(self._step_count)
                    self._committed_stone_idx = committed_stone

        self._step_count += 1
        success_terminal = False
        if self._move_committed:
            # Wait at least success_hold_steps, then require the committed
            # stone to have settled (low velocity) before terminating.
            held_long_enough = (
                self.success_hold_steps <= 0
                or (self._step_count - self._success_step) >= self.success_hold_steps
            )
            stone_settled = True
            if self._committed_stone_idx is not None and self._committed_stone_idx >= 0:
                jnt_name = self._rs_env._stone_joint_names[self._committed_stone_idx]
                vel = self._rs_env.sim.data.get_joint_qvel(jnt_name)
                stone_settled = bool(np.linalg.norm(vel) < 0.01)
            success_terminal = held_long_enough and stone_settled

        done = bool(success_terminal or self._logic.is_game_over or (self._step_count >= self.max_steps))
        reward = float(self._move_committed)
        obs = self.get_observation()
        info = {
            "target_row": int(self._target_rc[0]),
            "target_col": int(self._target_rc[1]),
            "move_committed": bool(self._move_committed),
            "step_count": int(self._step_count),
            "active_stone": (
                int(self._active_white_stone_idx) if self._active_white_stone_idx is not None else None
            ),
        }
        return obs, reward, done, info

    def render(
        self,
        mode: str = "rgb_array",
        height: int = 256,
        width: int = 256,
        camera_name: Optional[str] = None,
    ):
        if mode == "rgb_array":
            return self._rs_env.render_rgb(
                height=int(height),
                width=int(width),
                camera_name=camera_name,
            )
        if mode == "human":
            return None
        raise ValueError(f"unsupported render mode: {mode}")

    def _world_to_image_rc(self, xyz: np.ndarray, height: int, width: int) -> Tuple[int, int]:
        """Project a 3D world point to image (row, col) using the MuJoCo camera.

        MuJoCo camera body frame: +x right, +y up, +z backward (looks along -z).
        CV/image convention: +x right, +y down, +z forward.
        The axis correction diag(1, -1, -1) converts between them.
        The rendered image is flipped via [::-1] to standard top-down convention.
        """
        sim = self._rs_env.sim
        cam_name = self._rs_env.camera_names[0]
        cam_id = sim.model.camera_name2id(cam_name)

        cam_pos = sim.data.cam_xpos[cam_id]
        cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)

        # Transform world point into MuJoCo camera frame (right, up, backward)
        p_world = np.asarray(xyz, dtype=np.float64).ravel()[:3]
        p_cam = cam_mat.T @ (p_world - cam_pos)

        # Convert to CV convention: x_cv = x, y_cv = -y, z_cv = -z
        # z_cv (depth) is positive for visible points
        depth = -p_cam[2]
        if depth < 1e-8:
            return height // 2, width // 2

        fovy = sim.model.cam_fovy[cam_id]
        f = (0.5 * height) / np.tan(np.radians(fovy) * 0.5)

        col = f * p_cam[0] / depth + (width - 1) * 0.5
        row = f * (-p_cam[1]) / depth + (height - 1) * 0.5

        col = int(np.clip(round(col), 0, width - 1))
        row = int(np.clip(round(row), 0, height - 1))
        return row, col

    @staticmethod
    def _draw_disk(image: np.ndarray, row: int, col: int, radius: int, color: np.ndarray) -> None:
        h, w = image.shape[:2]
        r0 = max(0, row - radius)
        r1 = min(h - 1, row + radius)
        c0 = max(0, col - radius)
        c1 = min(w - 1, col + radius)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if (rr - row) * (rr - row) + (cc - col) * (cc - col) <= radius * radius:
                    image[rr, cc] = color

    def _overlay_debug_markers(self, image: np.ndarray) -> np.ndarray:
        rendered = image.copy()
        h, w = rendered.shape[:2]
        # Project at the line geom Z height so the dot lands on the visible
        # grid intersection (line geoms sit on top of the board surface).
        target_vis_xyz = self._target_pose[:3, 3].copy()
        try:
            board_gid = self._rs_env.sim.model.geom_name2id("go_board_surface_visual")
            target_vis_xyz[2] = float(self._rs_env.sim.data.geom_xpos[board_gid, 2])
        except Exception:
            pass
        target_rc = self._world_to_image_rc(target_vis_xyz, h, w)
        self._draw_disk(
            rendered,
            row=target_rc[0],
            col=target_rc[1],
            radius=2,
            color=np.array([20, 130, 255], dtype=np.uint8),
        )

        for idx, xyz in enumerate(self._eef_trail):
            row, col = self._world_to_image_rc(xyz, h, w)
            alpha = float(idx + 1) / float(max(len(self._eef_trail), 1))
            color = np.array(
                [int(220 * alpha), int(40 + 180 * alpha), int(20 + 20 * alpha)],
                dtype=np.uint8,
            )
            self._draw_disk(rendered, row=row, col=col, radius=1, color=color)

        eef_rc = self._world_to_image_rc(self.get_eef_pose()[:3, 3], h, w)
        self._draw_disk(
            rendered,
            row=eef_rc[0],
            col=eef_rc[1],
            radius=2,
            color=np.array([255, 70, 40], dtype=np.uint8),
        )
        return rendered

    def serialize(self) -> Dict[str, object]:
        return {
            "env_name": self.environment_name,
            "env_type": "robosuite_go_benchmark",
            "board_size": self.board_size,
            "action_shape": [4],
            "action_scale": self.action_scale,
            "press_height": self.press_height,
            "reach_xy_threshold": self.reach_xy_threshold,
            "success_hold_steps": self.success_hold_steps,
            "enable_opponent_moves": self.enable_opponent_moves,
            "opening_with_opponent": self.opening_with_opponent,
            "render_eef_overlay": self.render_eef_overlay,
            "eef_overlay_trail": self.eef_overlay_trail,
            "camera_height": self.camera_height,
            "camera_width": self.camera_width,
            "robot": self._robot.name,
        }

    def sample_random_action(self) -> np.ndarray:
        return self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)

    def close(self) -> None:
        self._rs_env.close()
