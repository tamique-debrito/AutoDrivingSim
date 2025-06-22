import math
from typing import Tuple, List, Dict, Any

class World:
    """
    World model for a car simulator with a circular road and two lanes.
    Cars travel clockwise. This means that the right lane is the innermost one and the left lane is the outermost
    """
    
    # Constants
    ROAD_RADIUS = 100.0
    LANE_WIDTH = 10.0
    ROAD_INNER = ROAD_RADIUS
    ROAD_OUTER = ROAD_RADIUS + 2 * LANE_WIDTH
    CROSSWALK_BUFFER = 10.0
    
    def __init__(self):
        """Initialize the world with crosswalks."""
        self.crosswalks = [
            ((0, World.ROAD_INNER - World.CROSSWALK_BUFFER), (0, World.ROAD_OUTER + World.CROSSWALK_BUFFER)),
            ((0, -World.ROAD_INNER + World.CROSSWALK_BUFFER), (0, -World.ROAD_OUTER - World.CROSSWALK_BUFFER)),
        ]
    
    def get_car_info(self, x: float, y: float, dx: float, dy: float) -> Dict[str, Any]:
        """
        Given x/y location and direction, return info relevant to a car at that location.
        
        Args:
            x: X coordinate of the car
            y: Y coordinate of the car
            dx: X component of the car's direction
            dy: Y component of the car's direction
        Returns:
            Dictionary containing:
            - 'lane_position': Signed distance from center of lane (negative = left, positive = right)
            - 'lane': Lane name ("Left" or "Right")
            - 'distance_from_center': Distance from road center
            - 'angle_relative_to_road': Angle relative to road direction
        """
        distance_from_center = math.sqrt(x**2 + y**2)
        
        right_lane_center_radius = self.ROAD_RADIUS + self.LANE_WIDTH / 2
        left_lane_center_radius = self.ROAD_RADIUS + 3 * self.LANE_WIDTH / 2
        
        if distance_from_center <= right_lane_center_radius + self.LANE_WIDTH / 2:
            lane = "Right"
            lane_center_distance = right_lane_center_radius
        elif distance_from_center >= left_lane_center_radius - self.LANE_WIDTH / 2:
            lane = "Left"
            lane_center_distance = left_lane_center_radius
        else:
            return {
                'lane_position': None,
                'lane': None,
                'distance_from_center': distance_from_center,
                'angle_relative_to_road': None,
            }
        
        # Calculate signed distance from lane center
        # Positive means car is to the right of lane center (when facing direction of travel)
        # Negative means car is to the left of lane center
        lane_position = -(distance_from_center - lane_center_distance)
        
        # Calculate angle relative to road direction
        angle_relative_to_road = None
        car_direction_magnitude = math.sqrt(dx**2 + dy**2)
        
        if distance_from_center > 0 and car_direction_magnitude > 0:
            road_dx, road_dy = y, -x
            x_component = dx * road_dx + dy * road_dy
            y_component = -dx * road_dy + dy * road_dx
            
            angle_relative_to_road = math.atan2(y_component, x_component)
            
        return {
            'lane_position': lane_position,
            'lane': lane,
            'distance_from_center': distance_from_center,
            'angle_relative_to_road': angle_relative_to_road,
        }