from dataclasses import dataclass


@dataclass
class ObstaclesInfo:
    collision_risk: bool
    left_obstacle: bool
    right_obstacle: bool
    front_obstacle: bool


@dataclass
class CarInfo:
    lane_position: float
    lane: str
    distance_from_center: float
    angle_relative_to_road: float


@dataclass
class ControlOutput:
    target_turn_angle: float
    velocity_action: str