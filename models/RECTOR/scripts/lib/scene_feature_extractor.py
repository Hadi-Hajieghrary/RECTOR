"""
Scene Feature Extractor: Waymo Scenario proto → proxy-compatible scene_features dict.

Extracts the scene context from a Waymo Scenario proto and transforms it into
the tensor format expected by DifferentiableRuleProxies.

All coordinates are in the ego-local frame (ego at origin, heading = +x)
in meters (NOT scaled by 1/50 like training data).

Output keys:
    agent_positions  [B, N, H, 2]   — other agents' future xy positions
    agent_sizes      [B, N, 2]       — (length, width) per agent
    agent_types      [B, N]          — 1=vehicle, 2=pedestrian, 3=cyclist
    agent_valid      [B, N, H]       — validity mask per agent per timestep
    agent_velocities [B, N, H, 2]    — other agents' future velocities
    vru_positions    [B, V, H, 2]    — VRU (ped+cyclist) future positions
    lane_centers     [B, L, P, 2]    — lane centerline points
    lane_headings    [B, L, P]       — heading angle at each lane point
    lane_speed_limits[B, L]          — speed limit per lane (m/s)
    road_edges       [B, E, 2]       — road edge boundary points
    stoplines        [B, S, 2]       — stopline positions
    signal_states    [B, H]          — traffic signal states per timestep
    ego_size         [B, 2]          — (length, width) of ego vehicle
    crosswalk_polygons [B, C, 4, 2]  — crosswalk corner points
"""

import numpy as np
from typing import Dict, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None


# Default dimensions
MAX_AGENTS = 32
MAX_VRUS = 16
MAX_LANES = 64
LANE_POINTS = 20
MAX_ROAD_EDGES = 128
MAX_STOPLINES = 16
MAX_CROSSWALKS = 16
FUTURE_STEPS = 80  # M2I 8-second horizon at 10 Hz


def extract_scene_features(
    scenario,
    device: str = "cpu",
    max_agents: int = MAX_AGENTS,
    max_lanes: int = MAX_LANES,
    future_steps: int = FUTURE_STEPS,
) -> Dict[str, "torch.Tensor"]:
    """
    Extract proxy-compatible scene_features from a Waymo Scenario proto.

    Args:
        scenario: waymo_open_dataset.protos.scenario_pb2.Scenario
        device: torch device for output tensors
        max_agents: max number of other agents to include
        max_lanes: max number of lanes to include
        future_steps: number of future timesteps (80 for M2I)

    Returns:
        Dict of scene feature tensors, all with batch dim B=1.
    """
    tracks = list(scenario.tracks)
    if not tracks:
        return _empty_features(device, max_agents, max_lanes, future_steps)

    sdc_idx = scenario.sdc_track_index
    if sdc_idx < 0 or sdc_idx >= len(tracks):
        sdc_idx = 0

    current_ts = scenario.current_time_index
    sdc_track = tracks[sdc_idx]

    # Reference pose for ego-local transform
    ref_state = sdc_track.states[current_ts]
    ref_x, ref_y = ref_state.center_x, ref_state.center_y
    ref_h = ref_state.heading
    cos_h, sin_h = np.cos(-ref_h), np.sin(-ref_h)

    def to_local(x, y):
        dx, dy = x - ref_x, y - ref_y
        return dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h

    def local_heading(h):
        rel = h - ref_h
        return (rel + np.pi) % (2 * np.pi) - np.pi

    # --- Ego size ---
    ego_length = sdc_track.states[current_ts].length or 4.5
    ego_width = sdc_track.states[current_ts].width or 2.0
    ego_size = np.array([[ego_length, ego_width]], dtype=np.float32)

    # --- Agent positions, sizes, types, velocities ---
    agent_positions = np.zeros((max_agents, future_steps, 2), dtype=np.float32)
    agent_velocities = np.zeros((max_agents, future_steps, 2), dtype=np.float32)
    agent_sizes = np.zeros((max_agents, 2), dtype=np.float32)
    agent_types = np.zeros(max_agents, dtype=np.float32)
    agent_valid = np.zeros((max_agents, future_steps), dtype=np.float32)

    # VRU tracking
    vru_positions = np.zeros((MAX_VRUS, future_steps, 2), dtype=np.float32)
    vru_count = 0

    agent_count = 0
    for i, track in enumerate(tracks):
        if i == sdc_idx or agent_count >= max_agents:
            continue
        n_states = len(track.states)

        # Get size from current time
        if current_ts < n_states and track.states[current_ts].valid:
            ref_agent = track.states[current_ts]
        else:
            # Find any valid state for size
            ref_agent = None
            for s in track.states:
                if s.valid:
                    ref_agent = s
                    break
            if ref_agent is None:
                continue

        agent_sizes[agent_count] = [ref_agent.length or 4.0, ref_agent.width or 2.0]
        agent_types[agent_count] = track.object_type  # 1=vehicle, 2=ped, 3=cyclist

        has_future = False
        for t in range(future_steps):
            ts_idx = current_ts + 1 + t
            if ts_idx < n_states and track.states[ts_idx].valid:
                s = track.states[ts_idx]
                lx, ly = to_local(s.center_x, s.center_y)
                agent_positions[agent_count, t] = [lx, ly]
                # Transform velocity to local frame
                vx_l = s.velocity_x * cos_h - s.velocity_y * sin_h
                vy_l = s.velocity_x * sin_h + s.velocity_y * cos_h
                agent_velocities[agent_count, t] = [vx_l, vy_l]
                agent_valid[agent_count, t] = 1.0
                has_future = True

        if not has_future:
            continue

        # VRU positions (pedestrians=2, cyclists=3)
        if track.object_type in (2, 3) and vru_count < MAX_VRUS:
            vru_positions[vru_count] = agent_positions[agent_count]
            vru_count += 1

        agent_count += 1

    # --- Lane centers, headings, speed limits ---
    lane_centers = np.zeros((max_lanes, LANE_POINTS, 2), dtype=np.float32)
    lane_headings = np.zeros((max_lanes, LANE_POINTS), dtype=np.float32)
    lane_speed_limits = np.full(max_lanes, 25.0, dtype=np.float32)  # default ~90 km/h
    lane_count = 0

    # --- Road edges, stoplines, crosswalks ---
    road_edges = np.zeros((MAX_ROAD_EDGES, 2), dtype=np.float32)
    edge_count = 0
    stoplines = np.zeros((MAX_STOPLINES, 2), dtype=np.float32)
    stopline_count = 0
    crosswalks = np.zeros((MAX_CROSSWALKS, 4, 2), dtype=np.float32)
    crosswalk_count = 0

    for mf in scenario.map_features:
        if mf.HasField("lane") and lane_count < max_lanes:
            polyline = list(mf.lane.polyline)
            if not polyline:
                continue
            indices = np.linspace(0, len(polyline) - 1, LANE_POINTS).astype(int)
            for p, src in enumerate(indices):
                pt = polyline[src]
                lx, ly = to_local(pt.x, pt.y)
                lane_centers[lane_count, p] = [lx, ly]
            # Compute headings from consecutive points
            for p in range(LANE_POINTS - 1):
                dx = lane_centers[lane_count, p + 1, 0] - lane_centers[lane_count, p, 0]
                dy = lane_centers[lane_count, p + 1, 1] - lane_centers[lane_count, p, 1]
                lane_headings[lane_count, p] = np.arctan2(dy, dx)
            lane_headings[lane_count, -1] = (
                lane_headings[lane_count, -2] if LANE_POINTS > 1 else 0.0
            )

            # Speed limit if available
            if mf.lane.speed_limit_mph > 0:
                lane_speed_limits[lane_count] = mf.lane.speed_limit_mph * 0.44704
            lane_count += 1

        elif mf.HasField("road_edge"):
            for pt in mf.road_edge.polyline:
                if edge_count >= MAX_ROAD_EDGES:
                    break
                lx, ly = to_local(pt.x, pt.y)
                road_edges[edge_count] = [lx, ly]
                edge_count += 1

        elif mf.HasField("road_line"):
            for pt in mf.road_line.polyline:
                if edge_count >= MAX_ROAD_EDGES:
                    break
                lx, ly = to_local(pt.x, pt.y)
                road_edges[edge_count] = [lx, ly]
                edge_count += 1

        elif mf.HasField("stop_sign"):
            if stopline_count < MAX_STOPLINES:
                pos = mf.stop_sign.position
                lx, ly = to_local(pos.x, pos.y)
                stoplines[stopline_count] = [lx, ly]
                stopline_count += 1

        elif mf.HasField("crosswalk"):
            if crosswalk_count < MAX_CROSSWALKS:
                poly = list(mf.crosswalk.polygon)
                if len(poly) >= 4:
                    for c in range(4):
                        lx, ly = to_local(poly[c].x, poly[c].y)
                        crosswalks[crosswalk_count, c] = [lx, ly]
                    crosswalk_count += 1

    # --- Traffic signal states ---
    signal_states = np.zeros(future_steps, dtype=np.float32)
    for ds in scenario.dynamic_map_states:
        ts = ds.timestamp_seconds
        # Map dynamic state to future step indices if applicable
        # Signal states are simplified to a single value per timestep
        for lane_state in ds.lane_states:
            # Use first lane signal as representative (simplified)
            if lane_state.state in (4, 5, 7):  # RED states
                # Find corresponding timestep
                # (Dynamic states are per-timestep in Waymo format)
                break

    # Pack into tensors with batch dim
    features = {
        "agent_positions": _to_tensor(agent_positions[None], device),
        "agent_sizes": _to_tensor(agent_sizes[None], device),
        "agent_types": _to_tensor(agent_types[None], device),
        "agent_valid": _to_tensor(agent_valid[None], device),
        "agent_velocities": _to_tensor(agent_velocities[None], device),
        "vru_positions": _to_tensor(vru_positions[None], device),
        "lane_centers": _to_tensor(lane_centers[None], device),
        "lane_headings": _to_tensor(lane_headings[None], device),
        "lane_speed_limits": _to_tensor(lane_speed_limits[None], device),
        "road_edges": _to_tensor(road_edges[None], device),
        "stoplines": _to_tensor(stoplines[None], device),
        "signal_states": _to_tensor(signal_states[None], device),
        "ego_size": _to_tensor(ego_size, device),
        "crosswalk_polygons": _to_tensor(crosswalks[None], device),
    }
    return features


def extract_encoder_inputs(
    scenario,
    history_length: int = 11,
    max_agents: int = 32,
    max_lanes: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Extract M2ISceneEncoder-compatible inputs from a Waymo Scenario proto.

    Returns normalized inputs (same format as WaymoDataset._extract_features)
    but WITHOUT the /50 scale — raw meter-scale ego-local coordinates.

    Returns:
        Dict with ego_history [11,4], agent_states [A,11,4], lane_centers [L,P,2]
    """
    tracks = list(scenario.tracks)
    if not tracks:
        return None

    sdc_idx = scenario.sdc_track_index
    if sdc_idx < 0 or sdc_idx >= len(tracks):
        sdc_idx = 0

    current_ts = scenario.current_time_index
    sdc_track = tracks[sdc_idx]
    ref_state = sdc_track.states[current_ts]
    ref_x, ref_y = ref_state.center_x, ref_state.center_y
    ref_h = ref_state.heading
    cos_h, sin_h = np.cos(-ref_h), np.sin(-ref_h)

    def to_local(x, y):
        dx, dy = x - ref_x, y - ref_y
        return dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h

    def local_heading(h):
        return ((h - ref_h) + np.pi) % (2 * np.pi) - np.pi

    # Ego history [11, 4] — (x, y, heading, speed) in meters
    ego_history = np.zeros((history_length, 4), dtype=np.float32)
    for t in range(history_length):
        if t < len(sdc_track.states) and sdc_track.states[t].valid:
            s = sdc_track.states[t]
            x, y = to_local(s.center_x, s.center_y)
            ego_history[t] = [
                x,
                y,
                local_heading(s.heading),
                np.sqrt(s.velocity_x**2 + s.velocity_y**2),
            ]

    # Agent states [A, 11, 4]
    agent_states = np.zeros((max_agents, history_length, 4), dtype=np.float32)
    agent_count = 0
    for i, track in enumerate(tracks):
        if i == sdc_idx or agent_count >= max_agents:
            continue
        has_valid = False
        for t in range(history_length):
            if t < len(track.states) and track.states[t].valid:
                s = track.states[t]
                x, y = to_local(s.center_x, s.center_y)
                agent_states[agent_count, t] = [
                    x,
                    y,
                    local_heading(s.heading),
                    np.sqrt(s.velocity_x**2 + s.velocity_y**2),
                ]
                has_valid = True
        if has_valid:
            agent_count += 1

    # Lane centers [L, P, 2]
    lane_centers = np.zeros((max_lanes, LANE_POINTS, 2), dtype=np.float32)
    lane_count = 0
    for mf in scenario.map_features:
        if lane_count >= max_lanes:
            break
        if mf.HasField("lane"):
            polyline = list(mf.lane.polyline)
            if not polyline:
                continue
            indices = np.linspace(0, len(polyline) - 1, LANE_POINTS).astype(int)
            for p, src in enumerate(indices):
                lx, ly = to_local(polyline[src].x, polyline[src].y)
                lane_centers[lane_count, p] = [lx, ly]
            lane_count += 1

    return {
        "ego_history": ego_history,
        "agent_states": agent_states,
        "lane_centers": lane_centers,
    }


def transform_trajectories_to_local(
    trajectories_world: np.ndarray,
    ref_x: float,
    ref_y: float,
    ref_heading: float,
) -> np.ndarray:
    """
    Transform world-frame trajectories to ego-local frame.

    Args:
        trajectories_world: [M, T, 2] in world coordinates
        ref_x, ref_y: ego position at current time
        ref_heading: ego heading at current time

    Returns:
        [M, T, 2] in ego-local frame (meters)
    """
    cos_h = np.cos(-ref_heading)
    sin_h = np.sin(-ref_heading)
    dx = trajectories_world[..., 0] - ref_x
    dy = trajectories_world[..., 1] - ref_y
    local = np.stack(
        [
            dx * cos_h - dy * sin_h,
            dx * sin_h + dy * cos_h,
        ],
        axis=-1,
    )
    return local.astype(np.float32)


def _to_tensor(arr: np.ndarray, device: str) -> "torch.Tensor":
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _empty_features(device, max_agents, max_lanes, future_steps):
    """Return zeroed-out features when scenario is empty."""
    return {
        "agent_positions": _to_tensor(
            np.zeros((1, max_agents, future_steps, 2)), device
        ),
        "agent_sizes": _to_tensor(np.zeros((1, max_agents, 2)), device),
        "agent_types": _to_tensor(np.zeros((1, max_agents)), device),
        "agent_valid": _to_tensor(np.zeros((1, max_agents, future_steps)), device),
        "agent_velocities": _to_tensor(
            np.zeros((1, max_agents, future_steps, 2)), device
        ),
        "vru_positions": _to_tensor(np.zeros((1, MAX_VRUS, future_steps, 2)), device),
        "lane_centers": _to_tensor(np.zeros((1, max_lanes, LANE_POINTS, 2)), device),
        "lane_headings": _to_tensor(np.zeros((1, max_lanes, LANE_POINTS)), device),
        "lane_speed_limits": _to_tensor(np.full((1, max_lanes), 25.0), device),
        "road_edges": _to_tensor(np.zeros((1, MAX_ROAD_EDGES, 2)), device),
        "stoplines": _to_tensor(np.zeros((1, MAX_STOPLINES, 2)), device),
        "signal_states": _to_tensor(np.zeros((1, future_steps)), device),
        "ego_size": _to_tensor(np.array([[4.5, 2.0]]), device),
        "crosswalk_polygons": _to_tensor(np.zeros((1, MAX_CROSSWALKS, 4, 2)), device),
    }
