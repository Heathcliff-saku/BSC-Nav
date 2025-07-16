from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from habitat.core.simulator import Simulator
from habitat.tasks.vln.vln import VLNEpisode
from habitat.utils.visualizations import maps as habitat_maps
import cv2
import gzip
import json
import pickle
from typing import Any, List, Union

import numpy as np
from habitat.core.dataset import Episode
from habitat.core.embodied_task import Action, EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations import fog_of_war
from numpy import ndarray



@registry.register_measure
class OracleSPL(Measure):
    """OracleSPL (Oracle Success weighted by Path Length)
    OracleSPL = max(SPL) over all points in the agent path.
    """

    cls_uuid: str = "oracle_spl"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, ["spl"])
        self._metric = 0.0

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        spl = task.measurements.measures["spl"].get_metric()
        self._metric = max(self._metric, spl)


AGENT_SPRITE = habitat_maps.AGENT_SPRITE

MAP_THICKNESS_SCALAR: int = 128

MAP_INVALID_POINT = 0
MAP_VALID_POINT = 1
MAP_BORDER_INDICATOR = 2
MAP_SOURCE_POINT_INDICATOR = 4
MAP_TARGET_POINT_INDICATOR = 6
MAP_MP3D_WAYPOINT = 7
MAP_VIEW_POINT_INDICATOR = 8
MAP_TARGET_BOUNDING_BOX = 9
MAP_REFERENCE_POINT = 10
MAP_MP3D_REFERENCE_PATH = 11
MAP_WAYPOINT_PREDICTION = 12
MAP_ORACLE_WAYPOINT = 13
MAP_SHORTEST_PATH_WAYPOINT = 14

TOP_DOWN_MAP_COLORS = np.full((256, 3), 150, dtype=np.uint8)
TOP_DOWN_MAP_COLORS[15:] = cv2.applyColorMap(
    np.arange(241, dtype=np.uint8), cv2.COLORMAP_JET
).squeeze(1)[:, ::-1]
TOP_DOWN_MAP_COLORS[MAP_INVALID_POINT] = [255, 255, 255]  # White
TOP_DOWN_MAP_COLORS[MAP_VALID_POINT] = [150, 150, 150]  # Light Grey
TOP_DOWN_MAP_COLORS[MAP_BORDER_INDICATOR] = [50, 50, 50]  # Grey
TOP_DOWN_MAP_COLORS[MAP_SOURCE_POINT_INDICATOR] = [0, 0, 200]  # Blue
TOP_DOWN_MAP_COLORS[MAP_TARGET_POINT_INDICATOR] = [200, 0, 0]  # Red
TOP_DOWN_MAP_COLORS[MAP_MP3D_WAYPOINT] = [0, 200, 0]  # Green
TOP_DOWN_MAP_COLORS[MAP_VIEW_POINT_INDICATOR] = [245, 150, 150]  # Light Red
TOP_DOWN_MAP_COLORS[MAP_TARGET_BOUNDING_BOX] = [0, 175, 0]  # Dark Green
TOP_DOWN_MAP_COLORS[MAP_REFERENCE_POINT] = [0, 0, 0]  # Black
TOP_DOWN_MAP_COLORS[MAP_MP3D_REFERENCE_PATH] = [0, 0, 0]  # Black
TOP_DOWN_MAP_COLORS[MAP_WAYPOINT_PREDICTION] = [255, 255, 0]  # Yellow
TOP_DOWN_MAP_COLORS[MAP_ORACLE_WAYPOINT] = [255, 165, 0]  # Orange
TOP_DOWN_MAP_COLORS[MAP_SHORTEST_PATH_WAYPOINT] = [0, 150, 0]  # Dark Green


def get_top_down_map(sim, map_resolution, meters_per_pixel):
    base_height = sim.get_agent(0).state.position[1]
    td_map = habitat_maps.get_topdown_map(
        sim.pathfinder,
        base_height,
        map_resolution,
        False,
        meters_per_pixel,
    )
    return td_map


def colorize_topdown_map(
    top_down_map: np.ndarray,
    fog_of_war_mask: Optional[np.ndarray] = None,
    fog_of_war_desat_amount: float = 0.5,
) -> np.ndarray:
    """Same as `maps.colorize_topdown_map` in Habitat-Lab, but with different
    colors.
    """
    _map = TOP_DOWN_MAP_COLORS[top_down_map]

    if fog_of_war_mask is not None:
        fog_of_war_desat_values = np.array([[fog_of_war_desat_amount], [1.0]])
        # Only desaturate valid points as only valid points get revealed
        desat_mask = top_down_map != MAP_INVALID_POINT

        _map[desat_mask] = (
            _map * fog_of_war_desat_values[fog_of_war_mask]
        ).astype(np.uint8)[desat_mask]

    return _map


def static_to_grid(
    realworld_x: float,
    realworld_y: float,
    grid_resolution: Tuple[int, int],
    bounds: Dict[str, Tuple[float, float]],
) -> Tuple[int, int]:
    """Return gridworld index of realworld coordinates assuming top-left
    corner is the origin. The real world coordinates of lower left corner are
    (coordinate_min, coordinate_min) and of top right corner are
    (coordinate_max, coordinate_max). Same as the habitat-Lab maps.to_grid
    function but with a static `bounds` instead of requiring a simulator or
    pathfinder instance.
    """
    grid_size = (
        abs(bounds["upper"][2] - bounds["lower"][2]) / grid_resolution[0],
        abs(bounds["upper"][0] - bounds["lower"][0]) / grid_resolution[1],
    )
    grid_x = int((realworld_x - bounds["lower"][2]) / grid_size[0])
    grid_y = int((realworld_y - bounds["lower"][0]) / grid_size[1])
    return grid_x, grid_y


def drawline(
    img: np.ndarray,
    pt1: Union[Tuple[float], List[float]],
    pt2: Union[Tuple[float], List[float]],
    color: List[int],
    thickness: int = 1,
    style: str = "dotted",
    gap: int = 15,
) -> None:
    """https://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines
    style: "dotted", "dashed", or "filled"
    """
    assert style in ["dotted", "dashed", "filled"]

    if style == "filled":
        cv2.line(img, pt1, pt2, color, thickness)
        return

    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
        pts.append((x, y))

    if style == "dotted":
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        for i, p in enumerate(pts):
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)


def drawpoint(
    img: np.ndarray,
    position: Union[Tuple[int], List[int]],
    color: List[int],
    meters_per_px: float,
    pad: float = 0.3,
) -> None:
    point_padding = int(pad / meters_per_px)
    img[
        position[0] - point_padding : position[0] + point_padding + 1,
        position[1] - point_padding : position[1] + point_padding + 1,
    ] = color


def draw_triangle(
    img: np.ndarray,
    centroid: Union[Tuple[int], List[int]],
    color: List[int],
    meters_per_px: float,
    pad: int = 0.35,
) -> None:
    point_padding = int(pad / meters_per_px)

    # (Y, X)
    left = (centroid[1] - point_padding, centroid[0] + point_padding)
    right = (centroid[1] + point_padding, centroid[0] + point_padding)
    top = (centroid[1], centroid[0] - point_padding)
    cv2.drawContours(img, [np.array([left, right, top])], 0, color, -1)


def draw_reference_path(
    img: np.ndarray,
    sim: Simulator,
    episode: VLNEpisode,
    map_resolution: int,
    meters_per_px: float,
) -> None:
    """Draws lines between each waypoint in the reference path."""
    shortest_path_points = [
        habitat_maps.to_grid(
            p[2],
            p[0],
            img.shape[0:2],
            sim,
        )[::-1]
        for p in episode.reference_path
    ]

    pt_from = None
    for i, pt_to in enumerate(shortest_path_points):

        if i != 0:
            drawline(
                img,
                (pt_from[0], pt_from[1]),
                (pt_to[0], pt_to[1]),
                MAP_SHORTEST_PATH_WAYPOINT,
                thickness=int(0.4 * map_resolution / MAP_THICKNESS_SCALAR),
                style="dashed",
                gap=10,
            )
        pt_from = pt_to

    for pt in shortest_path_points:
        drawpoint(
            img, (pt[1], pt[0]), MAP_SHORTEST_PATH_WAYPOINT, meters_per_px
        )


def draw_straight_shortest_path_points(
    img: np.ndarray,
    sim: Simulator,
    map_resolution: int,
    shortest_path_points: List[List[float]],
) -> None:
    """Draws the shortest path from start to goal assuming a standard
    discrete action space.
    """
    shortest_path_points = [
        habitat_maps.to_grid(p[2], p[0], img.shape[0:2], sim)[::-1]
        for p in shortest_path_points
    ]

    habitat_maps.draw_path(
        img,
        [(p[1], p[0]) for p in shortest_path_points],
        MAP_SHORTEST_PATH_WAYPOINT,
        int(0.4 * map_resolution / MAP_THICKNESS_SCALAR),
    )


def draw_source_and_target(
    img: np.ndarray, sim: Simulator, episode: VLNEpisode, meters_per_px: float
) -> None:
    s_x, s_y = habitat_maps.to_grid(
        episode.start_position[2],
        episode.start_position[0],
        img.shape[0:2],
        sim,
    )
    drawpoint(img, (s_x, s_y), MAP_SOURCE_POINT_INDICATOR, meters_per_px)

    # mark target point
    t_x, t_y = habitat_maps.to_grid(
        episode.goals[0].position[2],
        episode.goals[0].position[0],
        img.shape[0:2],
        sim,
    )
    drawpoint(img, (t_x, t_y), MAP_TARGET_POINT_INDICATOR, meters_per_px)


def draw_waypoint_prediction(
    img: np.ndarray,
    waypoint: Union[Tuple[float], List[float]],
    meters_per_px: float,
    bounds: Dict[str, Tuple[float]],
) -> None:
    w_x, w_y = static_to_grid(waypoint[1], waypoint[0], img.shape[0:2], bounds)
    if w_x < img.shape[0] and w_x > 0 and w_y < img.shape[1] and w_y > 0:
        draw_triangle(img, (w_x, w_y), MAP_WAYPOINT_PREDICTION, meters_per_px)


def draw_oracle_waypoint(
    img: np.ndarray,
    waypoint: Union[Tuple[float], List[float]],
    meters_per_px: float,
    bounds: Dict[str, Tuple[float]],
) -> None:
    w_x, w_y = static_to_grid(waypoint[1], waypoint[0], img.shape[0:2], bounds)
    draw_triangle(img, (w_x, w_y), MAP_ORACLE_WAYPOINT, meters_per_px, pad=0.2)


def get_nearest_node(graph: nx.Graph, current_position: List[float]) -> str:
    """Determine the closest MP3D node to the agent's start position as given
    by a [x,z] position vector.
    Returns:
        node ID
    """
    nearest = None
    dist = float("inf")
    for node in graph:
        node_pos = graph.nodes[node]["position"]
        node_pos = np.take(node_pos, (0, 2))
        cur_dist = np.linalg.norm(
            np.array(node_pos) - np.array(current_position), ord=2
        )
        if cur_dist < dist:
            dist = cur_dist
            nearest = node
    return nearest


def update_nearest_node(
    graph: nx.Graph, nearest_node: str, current_position: np.ndarray
) -> str:
    """Determine the closest MP3D node to the agent's current position as
    given by a [x,z] position vector. The selected node must be reachable
    from the previous MP3D node as specified in the nav-graph edges.
    Returns:
        node ID
    """
    nearest = None
    dist = float("inf")

    for node in [nearest_node] + [e[1] for e in graph.edges(nearest_node)]:
        node_pos = graph.nodes[node]["position"]
        node_pos = np.take(node_pos, (0, 2))
        cur_dist = np.linalg.norm(
            np.array(node_pos) - np.array(current_position), ord=2
        )
        if cur_dist < dist:
            dist = cur_dist
            nearest = node
    return nearest


def draw_mp3d_nodes(
    img: np.ndarray,
    sim: Simulator,
    episode: VLNEpisode,
    graph: nx.Graph,
    meters_per_px: float,
) -> None:
    n = get_nearest_node(
        graph, (episode.start_position[0], episode.start_position[2])
    )
    starting_height = graph.nodes[n]["position"][1]
    for node in graph:
        pos = graph.nodes[node]["position"]

        # no obvious way to differentiate between floors. Use this for now.
        if abs(pos[1] - starting_height) < 1.0:
            r_x, r_y = habitat_maps.to_grid(
                pos[2], pos[0], img.shape[0:2], sim
            )

            # only paint if over a valid point
            if img[r_x, r_y]:
                drawpoint(img, (r_x, r_y), MAP_MP3D_WAYPOINT, meters_per_px)



@registry.register_measure
class TopDownMapVLNCE(Measure):
    """A top down map that optionally shows VLN-related visual information
    such as MP3D node locations and MP3D agent traversals.
    """

    cls_uuid: str = "top_down_map_vlnce"

    def __init__(
        self, *args: Any, sim: Simulator, **kwargs: Any
    ) -> None:
        self._sim = sim
        self._step_count = None
        self._map_resolution = 1024
        self._previous_xy_location = None
        self._top_down_map = None
        self._meters_per_pixel = None
        self.current_node = ""
        with open('/home/orbit/桌面/Nav-2025/GES_vlnce/VLN_CE/data/connectivity_graphs.pkl', "rb") as f:
            self._conn_graphs = pickle.load(f)
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_original_map(self) -> ndarray:
        top_down_map = habitat_maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=self._map_resolution,
            draw_border=True,
            meters_per_pixel=self._meters_per_pixel,
        )

        self._fog_of_war_mask = None
        self._fog_of_war_mask = np.zeros_like(top_down_map)

        return top_down_map

    def reset_metric(
        self, *args: Any, episode: Episode, **kwargs: Any
    ) -> None:
        self._scene_id = episode.scene_id.split("/")[-2]
        self._step_count = 0
        self._metric = None
        self._meters_per_pixel = habitat_maps.calculate_meters_per_pixel(
            self._map_resolution, self._sim
        )
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        scene_id = episode.scene_id.split("/")[-1].split(".")[0]
        a_x, a_y = habitat_maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._previous_xy_location = (a_y, a_x)

        self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
            self._top_down_map,
            self._fog_of_war_mask,
            np.array([a_x, a_y]),
            self.get_polar_angle(),
            fov=90,
            max_line_len=5.0
            / habitat_maps.calculate_meters_per_pixel(
                self._map_resolution, sim=self._sim
            ),
        )

        draw_mp3d_nodes(
            self._top_down_map,
            self._sim,
            episode,
            self._conn_graphs[scene_id],
            self._meters_per_pixel,
        )

        shortest_path_points = self._sim.get_straight_shortest_path_points(
            agent_position, episode.goals[0].position
        )
        draw_straight_shortest_path_points(
            self._top_down_map,
            self._sim,
            self._map_resolution,
            shortest_path_points,
        )

        draw_reference_path(
            self._top_down_map,
            self._sim,
            episode,
            self._map_resolution,
            self._meters_per_pixel,
        )

        # draw source and target points last to avoid overlap
        draw_source_and_target(
            self._top_down_map,
            self._sim,
            episode,
            self._meters_per_pixel,
        )

        # MP3D START NODE
        self._nearest_node = get_nearest_node(
            self._conn_graphs[scene_id], np.take(agent_position, (0, 2))
        )
        nn_position = self._conn_graphs[self._scene_id].nodes[
            self._nearest_node
        ]["position"]
        self.s_x, self.s_y = habitat_maps.to_grid(
            nn_position[2],
            nn_position[0],
            self._top_down_map.shape[0:2],
            self._sim,
        )
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        self._step_count += 1
        (
            house_map,
            map_agent_pos,
        ) = self.update_map(self._sim.get_agent_state().position)

        self._metric = {
            "map": house_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": map_agent_pos,
            "agent_angle": self.get_polar_angle(),
            "bounds": {
                k: v
                for k, v in zip(
                    ["lower", "upper"],
                    self._sim.pathfinder.get_bounds(),
                )
            },
            "meters_per_px": self._meters_per_pixel,
        }

    def get_polar_angle(self) -> float:
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def update_map(self, agent_position: List[float]) -> None:
        a_x, a_y = habitat_maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            self._sim,
        )
        # Don't draw over the source point
        gradient_color = 15 + min(
            self._step_count * 245 // 5000, 245
        )
        if self._top_down_map[a_x, a_y] != MAP_SOURCE_POINT_INDICATOR:
            drawline(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                gradient_color,
                thickness=int(
                    self._map_resolution * 1.4 / MAP_THICKNESS_SCALAR
                ),
                style="filled",
            )

        self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
            self._top_down_map,
            self._fog_of_war_mask,
            np.array([a_x, a_y]),
            self.get_polar_angle(),
            90,
            max_line_len=5.0
            / habitat_maps.calculate_meters_per_pixel(
                self._map_resolution, sim=self._sim
            ),
        )

        point_padding = int(0.2 / self._meters_per_pixel)
        prev_nearest_node = self._nearest_node
        self._nearest_node = update_nearest_node(
            self._conn_graphs[self._scene_id],
            self._nearest_node,
            np.take(agent_position, (0, 2)),
        )
        if (
            self._nearest_node != prev_nearest_node
        ):
            nn_position = self._conn_graphs[self._scene_id].nodes[
                self._nearest_node
            ]["position"]
            (prev_s_x, prev_s_y) = (self.s_x, self.s_y)
            self.s_x, self.s_y = habitat_maps.to_grid(
                nn_position[2],
                nn_position[0],
                self._top_down_map.shape[0:2],
                self._sim,
            )
            self._top_down_map[
                self.s_x
                - int(2.0 / 3.0 * point_padding) : self.s_x
                + int(2.0 / 3.0 * point_padding)
                + 1,
                self.s_y
                - int(2.0 / 3.0 * point_padding) : self.s_y
                + int(2.0 / 3.0 * point_padding)
                + 1,
            ] = gradient_color

            drawline(
                self._top_down_map,
                (prev_s_y, prev_s_x),
                (self.s_y, self.s_x),
                gradient_color,
                thickness=int(
                    1.0
                    / 2.0
                    * np.round(
                        self._map_resolution / MAP_THICKNESS_SCALAR
                    )
                ),
            )

        self._previous_xy_location = (a_y, a_x)
        map_agent_pos = (a_x, a_y)
        return self._top_down_map, map_agent_pos




import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio
import numpy as np
import scipy.ndimage
from habitat.utils.visualizations import utils
try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    pass
import cv2


AGENT_SPRITE = imageio.imread('/home/orbit/桌面/Nav-2025/GES_vlnce/habitat-lab/habitat/utils/visualizations/assets/maps_topdown_agent_sprite/100x100.png')
AGENT_SPRITE = np.ascontiguousarray(np.flipud(AGENT_SPRITE))

MAP_INVALID_POINT = 0
MAP_VALID_POINT = 1
MAP_BORDER_INDICATOR = 2
MAP_SOURCE_POINT_INDICATOR = 4
MAP_TARGET_POINT_INDICATOR = 6
MAP_SHORTEST_PATH_COLOR = 7
MAP_VIEW_POINT_INDICATOR = 8
MAP_TARGET_BOUNDING_BOX = 9
TOP_DOWN_MAP_COLORS = np.full((256, 3), 150, dtype=np.uint8)
TOP_DOWN_MAP_COLORS[10:] = cv2.applyColorMap(
    np.arange(246, dtype=np.uint8), cv2.COLORMAP_JET
).squeeze(1)[:, ::-1]
TOP_DOWN_MAP_COLORS[MAP_INVALID_POINT] = [255, 255, 255]  # White
TOP_DOWN_MAP_COLORS[MAP_VALID_POINT] = [150, 150, 150]  # Light Grey
TOP_DOWN_MAP_COLORS[MAP_BORDER_INDICATOR] = [50, 50, 50]  # Grey
TOP_DOWN_MAP_COLORS[MAP_SOURCE_POINT_INDICATOR] = [0, 0, 200]  # Blue
TOP_DOWN_MAP_COLORS[MAP_TARGET_POINT_INDICATOR] = [200, 0, 0]  # Red
TOP_DOWN_MAP_COLORS[MAP_SHORTEST_PATH_COLOR] = [0, 200, 0]  # Green
TOP_DOWN_MAP_COLORS[MAP_VIEW_POINT_INDICATOR] = [245, 150, 150]  # Light Red
TOP_DOWN_MAP_COLORS[MAP_TARGET_BOUNDING_BOX] = [0, 175, 0]  # Green


def draw_agent(
    image: np.ndarray,
    agent_center_coord: Tuple[int, int],
    agent_rotation: float,
    agent_radius_px: int = 5,
) -> np.ndarray:
    r"""Return an image with the agent image composited onto it.
    Args:
        image: the image onto which to put the agent.
        agent_center_coord: the image coordinates where to paste the agent.
        agent_rotation: the agent's current rotation in radians.
        agent_radius_px: 1/2 number of pixels the agent will be resized to.
    Returns:
        The modified background image. This operation is in place.
    """

    # Rotate before resize to keep good resolution.
    rotated_agent = scipy.ndimage.interpolation.rotate(
        AGENT_SPRITE, agent_rotation * 180 / np.pi
    )
    # Rescale because rotation may result in larger image than original, but
    # the agent sprite size should stay the same.
    initial_agent_size = AGENT_SPRITE.shape[0]
    new_size = rotated_agent.shape[0]
    agent_size_px = max(
        1, int(agent_radius_px * 2 * new_size / initial_agent_size)
    )
    resized_agent = cv2.resize(
        rotated_agent,
        (agent_size_px, agent_size_px),
        interpolation=cv2.INTER_LINEAR,
    )
    utils.paste_overlapping_image(image, resized_agent, agent_center_coord)
    return image


def pointnav_draw_target_birdseye_view(
    agent_position: np.ndarray,
    agent_heading: float,
    goal_position: np.ndarray,
    resolution_px: int = 800,
    goal_radius: float = 0.2,
    agent_radius_px: int = 20,
    target_band_radii: Optional[List[float]] = None,
    target_band_colors: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    r"""Return an image of agent w.r.t. centered target location for pointnav
    tasks.

    Args:
        agent_position: the agent's current position.
        agent_heading: the agent's current rotation in radians. This can be
            found using the HeadingSensor.
        goal_position: the pointnav task goal position.
        resolution_px: number of pixels for the output image width and height.
        goal_radius: how near the agent needs to be to be successful for the
            pointnav task.
        agent_radius_px: 1/2 number of pixels the agent will be resized to.
        target_band_radii: distance in meters to the outer-radius of each band
            in the target image.
        target_band_colors: colors in RGB 0-255 for the bands in the target.
    Returns:
        Image centered on the goal with the agent's current relative position
        and rotation represented by an arrow. To make the rotations align
        visually with habitat, positive-z is up, positive-x is left and a
        rotation of 0 points upwards in the output image and rotates clockwise.
    """
    if target_band_radii is None:
        target_band_radii = [20, 10, 5, 2.5, 1]
    if target_band_colors is None:
        target_band_colors = [
            (47, 19, 122),
            (22, 99, 170),
            (92, 177, 0),
            (226, 169, 0),
            (226, 12, 29),
        ]

    assert len(target_band_radii) == len(
        target_band_colors
    ), "There must be an equal number of scales and colors."

    goal_agent_dist = np.linalg.norm(agent_position - goal_position, 2)

    goal_distance_padding = max(
        2, 2 ** np.ceil(np.log(max(1e-6, goal_agent_dist)) / np.log(2))
    )
    movement_scale = 1.0 / goal_distance_padding
    half_res = resolution_px // 2
    im_position = np.full(
        (resolution_px, resolution_px, 3), 255, dtype=np.uint8
    )

    # Draw bands:
    for scale, color in zip(target_band_radii, target_band_colors):
        if goal_distance_padding * 4 > scale:
            cv2.circle(
                im_position,
                (half_res, half_res),
                max(2, int(half_res * scale * movement_scale)),
                color,
                thickness=-1,
            )

    # Draw such that the agent being inside the radius is the circles
    # overlapping.
    cv2.circle(
        im_position,
        (half_res, half_res),
        max(2, int(half_res * goal_radius * movement_scale)),
        (127, 0, 0),
        thickness=-1,
    )

    relative_position = agent_position - goal_position
    # swap x and z, remove y for (x,y,z) -> image coordinates.
    relative_position = relative_position[[2, 0]]
    relative_position *= half_res * movement_scale
    relative_position += half_res
    relative_position = np.round(relative_position).astype(np.int32)

    # Draw the agent
    draw_agent(im_position, relative_position, agent_heading, agent_radius_px)

    # Rotate twice to fix coordinate system to upwards being positive-z.
    # Rotate instead of flip to keep agent rotations in sync with egocentric
    # view.
    im_position = np.rot90(im_position, 2)
    return im_position


def to_grid(
    realworld_x: float,
    realworld_y: float,
    grid_resolution: Tuple[int, int],
    sim: Optional["HabitatSim"] = None,
    pathfinder=None,
) -> Tuple[int, int]:
    r"""Return gridworld index of realworld coordinates assuming top-left corner
    is the origin. The real world coordinates of lower left corner are
    (coordinate_min, coordinate_min) and of top right corner are
    (coordinate_max, coordinate_max)
    """
    if sim is None and pathfinder is None:
        raise RuntimeError(
            "Must provide either a simulator or pathfinder instance"
        )

    if pathfinder is None:
        pathfinder = sim.pathfinder

    lower_bound, upper_bound = pathfinder.get_bounds()

    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    grid_x = int((realworld_x - lower_bound[2]) / grid_size[0])
    grid_y = int((realworld_y - lower_bound[0]) / grid_size[1])
    return grid_x, grid_y


def from_grid(
    grid_x: int,
    grid_y: int,
    grid_resolution: Tuple[int, int],
    sim: Optional["HabitatSim"] = None,
    pathfinder=None,
) -> Tuple[float, float]:
    r"""Inverse of _to_grid function. Return real world coordinate from
    gridworld assuming top-left corner is the origin. The real world
    coordinates of lower left corner are (coordinate_min, coordinate_min) and
    of top right corner are (coordinate_max, coordinate_max)
    """

    if sim is None and pathfinder is None:
        raise RuntimeError(
            "Must provide either a simulator or pathfinder instance"
        )

    if pathfinder is None:
        pathfinder = sim.pathfinder

    lower_bound, upper_bound = pathfinder.get_bounds()

    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    realworld_x = lower_bound[2] + grid_x * grid_size[0]
    realworld_y = lower_bound[0] + grid_y * grid_size[1]
    return realworld_x, realworld_y


def _outline_border(top_down_map):
    left_right_block_nav = (top_down_map[:, :-1] == 1) & (
        top_down_map[:, :-1] != top_down_map[:, 1:]
    )
    left_right_nav_block = (top_down_map[:, 1:] == 1) & (
        top_down_map[:, :-1] != top_down_map[:, 1:]
    )

    up_down_block_nav = (top_down_map[:-1] == 1) & (
        top_down_map[:-1] != top_down_map[1:]
    )
    up_down_nav_block = (top_down_map[1:] == 1) & (
        top_down_map[:-1] != top_down_map[1:]
    )

    top_down_map[:, :-1][left_right_block_nav] = MAP_BORDER_INDICATOR
    top_down_map[:, 1:][left_right_nav_block] = MAP_BORDER_INDICATOR

    top_down_map[:-1][up_down_block_nav] = MAP_BORDER_INDICATOR
    top_down_map[1:][up_down_nav_block] = MAP_BORDER_INDICATOR


def calculate_meters_per_pixel(
    map_resolution: int, sim: Optional["HabitatSim"] = None, pathfinder=None
):
    r"""Calculate the meters_per_pixel for a given map resolution"""
    if sim is None and pathfinder is None:
        raise RuntimeError(
            "Must provide either a simulator or pathfinder instance"
        )

    if pathfinder is None:
        pathfinder = sim.pathfinder

    lower_bound, upper_bound = pathfinder.get_bounds()
    return min(
        abs(upper_bound[coord] - lower_bound[coord]) / map_resolution
        for coord in [0, 2]
    )


def get_topdown_map(
    pathfinder,
    height: float,
    map_resolution: int = 1024,
    draw_border: bool = True,
    meters_per_pixel: Optional[float] = None,
) -> np.ndarray:
    r"""Return a top-down occupancy map for a sim. Note, this only returns valid
    values for whatever floor the agent is currently on.

    :param pathfinder: A habitat-sim pathfinder instances to get the map from
    :param height: The height in the environment to make the topdown map
    :param map_resolution: Length of the longest side of the map.  Used to calculate :p:`meters_per_pixel`
    :param draw_border: Whether or not to draw a border
    :param meters_per_pixel: Overrides map_resolution an

    :return: Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
        the flag is set).
    """

    if meters_per_pixel is None:
        meters_per_pixel = calculate_meters_per_pixel(
            map_resolution, pathfinder=pathfinder
        )

    top_down_map = pathfinder.get_topdown_view(
        meters_per_pixel=meters_per_pixel, height=height
    ).astype(np.uint8)

    # Draw border if necessary
    if draw_border:
        _outline_border(top_down_map)

    return np.ascontiguousarray(top_down_map)


def get_topdown_map_from_sim(
    sim: "HabitatSim",
    map_resolution: int = 1024,
    draw_border: bool = True,
    meters_per_pixel: Optional[float] = None,
    agent_id: int = 0,
) -> np.ndarray:
    r"""Wrapper around :py:`get_topdown_map` that retrieves that pathfinder and heigh from the current simulator

    :param sim: Simulator instance.
    :param agent_id: The agent ID
    """
    return get_topdown_map(
        sim.pathfinder,
        sim.get_agent(agent_id).state.position[1],
        map_resolution,
        draw_border,
        meters_per_pixel,
    )


def colorize_topdown_map(
    top_down_map: np.ndarray,
    fog_of_war_mask: Optional[np.ndarray] = None,
    fog_of_war_desat_amount: float = 0.5,
) -> np.ndarray:
    r"""Convert the top down map to RGB based on the indicator values.
    Args:
        top_down_map: A non-colored version of the map.
        fog_of_war_mask: A mask used to determine which parts of the
            top_down_map are visible
            Non-visible parts will be desaturated
        fog_of_war_desat_amount: Amount to desaturate the color of unexplored areas
            Decreasing this value will make unexplored areas darker
            Default: 0.5
    Returns:
        A colored version of the top-down map.
    """
    _map = TOP_DOWN_MAP_COLORS[top_down_map]

    if fog_of_war_mask is not None:
        fog_of_war_desat_values = np.array([[fog_of_war_desat_amount], [1.0]])
        # Only desaturate things that are valid points as only valid points get revealed
        desat_mask = top_down_map != MAP_INVALID_POINT

        _map[desat_mask] = (
            _map * fog_of_war_desat_values[fog_of_war_mask]
        ).astype(np.uint8)[desat_mask]

    return _map


def draw_path(
    top_down_map: np.ndarray,
    path_points: Sequence[Tuple],
    color: int = 10,
    thickness: int = 2,
) -> None:
    r"""Draw path on top_down_map (in place) with specified color.
    Args:
        top_down_map: A colored version of the map.
        color: color code of the path, from TOP_DOWN_MAP_COLORS.
        path_points: list of points that specify the path to be drawn
        thickness: thickness of the path.
    """
    for prev_pt, next_pt in zip(path_points[:-1], path_points[1:]):
        # Swapping x y
        cv2.line(
            top_down_map,
            prev_pt[::-1],
            next_pt[::-1],
            color,
            thickness=thickness,
        )


def colorize_draw_agent_and_fit_to_height_vlnce(
    topdown_map_info: Dict[str, Any], output_height: int
):
    r"""Given the output of the TopDownMap measure, colorizes the map, draws the agent,
    and fits to a desired output height

    :param topdown_map_info: The output of the TopDownMap measure
    :param output_height: The desired output height
    """
    top_down_map = topdown_map_info["map"]
    top_down_map = colorize_topdown_map(
        top_down_map, topdown_map_info["fog_of_war_mask"]
    )
    map_agent_pos = topdown_map_info["agent_map_coord"]
    top_down_map = draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=topdown_map_info["agent_angle"],
        agent_radius_px=min(top_down_map.shape[0:2]) // 32,
    )

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)

    # scale top down map to align with rgb view
    old_h, old_w, _ = top_down_map.shape
    top_down_height = output_height
    top_down_width = int(float(top_down_height) / old_h * old_w)
    # cv2 resize (dsize is width first)
    top_down_map = cv2.resize(
        top_down_map,
        (top_down_width, top_down_height),
        interpolation=cv2.INTER_CUBIC,
    )

    return top_down_map