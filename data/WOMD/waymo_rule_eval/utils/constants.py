"""
Constants for Waymo Rule Evaluation Pipeline.

Defines Waymo object types, agent mappings, default dimensions,
and threshold constants for rule evaluation.
"""

# --- Waymo Object Types ---

WAYMO_TYPE_UNSET = 0
WAYMO_TYPE_VEHICLE = 1
WAYMO_TYPE_PEDESTRIAN = 2
WAYMO_TYPE_CYCLIST = 3

AGENT_TYPE_MAP: dict = {
    WAYMO_TYPE_VEHICLE: "vehicle",
    WAYMO_TYPE_PEDESTRIAN: "pedestrian",
    WAYMO_TYPE_CYCLIST: "cyclist",
}

# Reverse mapping for convenience
TYPE_TO_WAYMO_MAP: dict = {
    "vehicle": WAYMO_TYPE_VEHICLE,
    "pedestrian": WAYMO_TYPE_PEDESTRIAN,
    "cyclist": WAYMO_TYPE_CYCLIST,
}


# --- Default Dimensions (meters) ---

DEFAULT_DIMENSIONS: dict = {
    "vehicle": (4.5, 1.8),  # (length, width)
    "pedestrian": (0.5, 0.5),
    "cyclist": (1.8, 0.6),
}

DEFAULT_EGO_LENGTH = 4.8
DEFAULT_EGO_WIDTH = 1.9


# --- Time Constants ---

DEFAULT_DT_S = 0.1  # Waymo uses 10 Hz


# --- Traffic Signal States ---

SIGNAL_STATE_UNKNOWN = 0
SIGNAL_STATE_ARROW_STOP = 1
SIGNAL_STATE_ARROW_CAUTION = 2
SIGNAL_STATE_ARROW_GO = 3
SIGNAL_STATE_STOP = 4
SIGNAL_STATE_CAUTION = 5
SIGNAL_STATE_GO = 6
SIGNAL_STATE_FLASHING_STOP = 7
SIGNAL_STATE_FLASHING_CAUTION = 8

RED_SIGNAL_STATES = frozenset(
    {
        SIGNAL_STATE_ARROW_STOP,
        SIGNAL_STATE_STOP,
        SIGNAL_STATE_FLASHING_STOP,
    }
)

YELLOW_SIGNAL_STATES = frozenset(
    {
        SIGNAL_STATE_ARROW_CAUTION,
        SIGNAL_STATE_CAUTION,
        SIGNAL_STATE_FLASHING_CAUTION,
    }
)

GREEN_SIGNAL_STATES = frozenset(
    {
        SIGNAL_STATE_ARROW_GO,
        SIGNAL_STATE_GO,
    }
)


# --- Following Distance Thresholds ---

TWO_SECOND_RULE_TIME_S = 2.0
FOLLOWING_DISTANCE_MIN_SPEED_MS = 3.0  # Below this, don't apply 2-second rule
FOLLOWING_DISTANCE_MIN_RATIO = 0.5  # Fraction of ideal following distance
TAILGATING_TIME_GAP_S = 0.5  # Very aggressive tailgating threshold


# --- Speed/Acceleration Thresholds ---

SPEED_LIMIT_TOLERANCE_MS = 2.24  # ~5 mph tolerance over speed limit
MAX_REASONABLE_SPEED_MS = 45.0  # ~100 mph sanity check

COMFORT_ACCELERATION_LIMIT = 3.0  # m/s^2
COMFORT_DECELERATION_LIMIT = 2.5  # m/s^2 (positive value, deceleration)
EMERGENCY_DECELERATION_LIMIT = 4.0  # m/s^2 - hard braking threshold
HARSH_BRAKING_THRESHOLD = 3.5  # m/s^2 - uncomfortable braking

COMFORT_JERK_LIMIT = 2.5  # m/s^3
MAX_JERK_LIMIT = 5.0  # m/s^3 - safety threshold

MAX_STEERING_RATE = 0.5  # rad/s - comfort steering rate


# --- Collision Detection Thresholds ---

COLLISION_PENETRATION_THRESHOLD_M = 0.1  # Minimum overlap to count as collision
MAX_COLLISION_CHECK_RADIUS_M = 30.0  # Pre-filter agents beyond this distance

NEAR_MISS_THRESHOLD_M = 1.0  # Close call distance


# --- VRU Clearance Thresholds ---

VRU_PEDESTRIAN_CLEARANCE_M = 2.0  # Minimum clearance to pedestrians
VRU_CYCLIST_CLEARANCE_M = 1.5  # Minimum clearance to cyclists
VRU_CLEARANCE_SPEED_THRESHOLD_MS = 5.0  # Speed above which clearance applies


# --- Spatial Query Parameters ---

DEFAULT_SPATIAL_RADIUS_M = 50.0  # Default radius for agent queries
K_NEAREST_NEIGHBORS = 10  # Default k for k-NN queries


# --- Map Geometry Constants ---

STOPLINE_DISTANCE_THRESHOLD_M = 2.0  # Distance to consider ego at stopline
CROSSWALK_ENTRY_MARGIN_M = 1.0  # Margin for crosswalk entry detection
LANE_WIDTH_DEFAULT_M = 3.7  # Standard lane width

ROAD_EDGE_TYPE_UNKNOWN = 0
ROAD_EDGE_TYPE_BOUNDARY = 1
ROAD_EDGE_TYPE_MEDIAN = 2


# --- L0: Critical Safety Rule Constants ---

# L0.R2: Safe Longitudinal Distance
SAFE_FOLLOWING_TIME_GAP_S = 2.0  # Minimum 2-second rule
SAFE_FOLLOWING_DETECTION_RANGE_M = 60.0  # Detection range for leader

# L0.R3: Safe Lateral Clearance
MIN_LATERAL_CLEARANCE_VEHICLE_M = 0.5  # Clearance to vehicles
MIN_LATERAL_CLEARANCE_CYCLIST_M = 1.0  # Clearance to cyclists
MIN_LATERAL_CLEARANCE_PEDESTRIAN_M = 1.5  # Clearance to pedestrians
LATERAL_DETECTION_RANGE_M = 30.0  # Detection range for lateral checks

# L0.R4: Crosswalk Occupancy
CROSSWALK_DETECTION_BUFFER_M = 5.0  # Buffer around crosswalk
MIN_PEDESTRIAN_SPEED_MPS = 0.3  # Min speed to consider ped active
CROSSWALK_MAX_DISTANCE_M = 50.0  # Max distance to consider crosswalk


# --- L1: Comfort/Smoothness Rule Constants ---

# L1.R1: Smooth Acceleration
L1_COMFORTABLE_ACCEL_MPS2 = 2.0  # m/s^2 - comfort threshold
L1_CRITICAL_ACCEL_MPS2 = 3.0  # m/s^2 - critical threshold
L1_COMFORTABLE_JERK_MPS3 = 2.0  # m/s^3 - comfort jerk
L1_CRITICAL_JERK_MPS3 = 4.0  # m/s^3 - critical jerk

# L1.R2: Smooth Braking (re-use acceleration constants)
SMOOTH_ACCELERATION_THRESHOLD = 2.5  # m/s^2 - comfort threshold
MAX_ACCELERATION_THRESHOLD = 4.0  # m/s^2 - harsh acceleration

# L1.R3: Smooth Steering
L1_COMFORTABLE_HEADING_RATE_DEG_S = 15.0  # deg/s - comfort yaw rate
L1_CRITICAL_HEADING_RATE_DEG_S = 30.0  # deg/s - critical yaw rate
L1_COMFORTABLE_ANGULAR_JERK_DEG_S2 = 15.0  # deg/s^2 - comfort angular jerk
L1_CRITICAL_ANGULAR_JERK_DEG_S2 = 30.0  # deg/s^2 - critical angular jerk
SMOOTH_YAW_RATE_LIMIT = 0.3  # rad/s - comfort yaw rate
MAX_YAW_RATE_LIMIT = 0.6  # rad/s - max yaw rate
SMOOTH_LATERAL_ACCEL_LIMIT = 2.0  # m/s^2 - lateral acceleration

# L1.R4: Speed Consistency
L1_COMFORTABLE_SPEED_VARIANCE_MPS = 2.0  # m/s - comfort variance
L1_CRITICAL_SPEED_VARIANCE_MPS = 4.0  # m/s - critical variance
L1_SPEED_WINDOW_DURATION_S = 2.0  # Window for variance calc
L1_OSCILLATION_THRESHOLD = 3  # Sign changes for oscillation
SPEED_VARIATION_THRESHOLD_MPS = 5.0  # Max speed variation in window
SPEED_CONSISTENCY_WINDOW_S = 5.0  # Window for speed consistency

# L1.R5: Lane Change Smoothness
L1_COMFORTABLE_LATERAL_ACCEL_MPS2 = 1.5  # m/s^2
L1_CRITICAL_LATERAL_ACCEL_MPS2 = 2.5  # m/s^2
LANE_CHANGE_MAX_LATERAL_ACCEL = 1.5  # m/s^2
LANE_CHANGE_MIN_DURATION_S = 2.0  # Minimum lane change duration


# --- L3-L4: Surface and Maneuver Rule Constants ---

# L3.R3: Drivable Surface
L3_LANE_WIDTH_M = 3.7  # Standard lane width
L3_DRIVABLE_BUFFER_M = 0.5  # Buffer around lane center
ROAD_EDGE_MARGIN_M = 0.5  # Margin from road edge
OFFROAD_THRESHOLD_M = 1.0  # Distance to consider offroad

# Vehicle dimensions for drivable surface checks
VEHICLE_LENGTH_M = 4.5  # Default vehicle length
VEHICLE_WIDTH_M = 2.0  # Default vehicle width

# Minimum moving speed for rules
MIN_MOVING_SPEED_MPS = 0.5  # m/s - below this is stationary

# L4.R3: Left Turn Gap
L4_SAFE_TTC_S = 4.0  # Minimum safe TTC for left turn
L4_CRITICAL_TTC_S = 2.0  # Critical TTC threshold
L4_TURN_THRESHOLD_DEG = 15.0  # Heading change for left turn
L4_ONCOMING_DETECTION_RANGE_M = 50.0  # Detection range for oncoming
LEFT_TURN_MIN_GAP_S = 4.0  # Minimum gap for left turn
LEFT_TURN_DETECTION_RANGE_M = 50.0  # Detection range for oncoming


# --- L5: Zone Compliance Rule Constants ---

SCHOOL_ZONE_SPEED_LIMIT_MPS = 11.2  # ~25 mph
CONSTRUCTION_ZONE_SPEED_LIMIT_MPS = 13.4  # ~30 mph
PARKING_ZONE_SPEED_LIMIT_MPS = 4.5  # ~10 mph


# --- L6: Interaction Rule Constants ---

# L6.R3: Cyclist Interaction
CYCLIST_PASSING_CLEARANCE_M = 1.5  # Min clearance when passing cyclist
CYCLIST_DETECTION_RANGE_M = 30.0  # Detection range for cyclists

# L6.R4: Pedestrian Interaction
PEDESTRIAN_YIELD_DISTANCE_M = 5.0  # Distance to yield to pedestrians
PEDESTRIAN_DETECTION_RANGE_M = 30.0  # Detection range

# L6.R5: Intersection Negotiation
INTERSECTION_DETECTION_RANGE_M = 40.0  # Detection range at intersections
INTERSECTION_YIELD_GAP_S = 3.0  # Minimum gap at intersection

# L6.R6: Cooperative Lane Change
LANE_CHANGE_GAP_FRONT_M = 10.0  # Min gap in front for lane change
LANE_CHANGE_GAP_REAR_M = 8.0  # Min gap behind for lane change


# --- L7: Lane/Speed Rule Constants ---

# L7.R3: Lane Departure
L7_LANE_HALF_WIDTH_M = 1.8  # Half lane width for departure
L7_MIN_DEPARTURE_DURATION_S = 0.5  # Min duration for violation
LANE_DEPARTURE_THRESHOLD_M = 0.5  # Distance outside lane boundary
LANE_DEPARTURE_DURATION_S = 0.5  # Duration to count as departure

# L7.R4: Speed Limit
L7_SPEED_TOLERANCE_MPS = 0.5  # Speed tolerance before violation


# --- L8-L9: Traffic Control Rule Constants ---

# L8.R2: Stop Sign
L8_STOP_SPEED_MPS = 0.3  # Speed to consider stopped
L8_STOPLINE_EPSILON_M = 0.2  # Epsilon for stop line position
L8_NEAR_STOP_M = 8.0  # Distance to consider "near" stop
L8_WINDOW_FRAMES = 15  # Window for checking stop

# L8.R3: Crosswalk Yield
CROSSWALK_YIELD_DISTANCE_M = 3.0  # Distance to yield at crosswalk

# L8.R5: Wrong Way
L8_WRONGWAY_ANGLE_DEG = 90.0  # Angle threshold for wrong-way
L8_WRONGWAY_MIN_DURATION_S = 1.0  # Min duration for violation
WRONG_WAY_HEADING_TOLERANCE_RAD = 1.57  # ~90 degrees tolerance

# L9.R1: Offroad/Wrong Way
OFFROAD_DETECTION_MARGIN_M = 1.0  # Margin for offroad detection

# General stop sign constants
STOP_SIGN_DETECTION_RANGE_M = 30.0  # Detection range for stop signs
STOP_SIGN_STOP_DURATION_S = 1.0  # Required stop duration
STOP_SIGN_SPEED_THRESHOLD_MPS = 0.5  # Speed to consider stopped


# --- Normalization Factors for Severity ---

RULE_NORMALIZATION = {
    "L0.R2": 20.0,  # Safe longitudinal distance: 20 meter-seconds
    "L0.R3": 10.0,  # Safe lateral clearance: 10 meter-seconds
    "L0.R4": 20.0,  # Crosswalk occupancy: 20 m²·s
    "L1.R1": 10.0,  # Smooth braking: 10 m/s^2 total
    "L1.R2": 10.0,  # Smooth acceleration: 10 m/s^2 total
    "L1.R3": 5.0,  # Smooth steering: 5 rad total
    "L6.R2": 20.0,  # Following distance: 20 meter-seconds
    "L8.R1": 1.0,  # Red light: per violation
    "L10.R1": 2.0,  # Collision: 2m penetration
    "L10.R2": 5.0,  # VRU clearance: 5m total deficit
}
