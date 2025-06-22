from calendar import c
import math
import random
import pygame
from typing import List, Tuple, Union
from World import World
from CarControl import CarControl

class Pedestrian:
    """Represents a pedestrian with position, direction, and expiry time."""
    
    PEDESTRIAN_VELOCITY = 2.0  # units per second
    
    def __init__(self, x: float, y: float, dir_x: float, dir_y: float, expiry_time: float):
        """
        Initialize a pedestrian.
        
        Args:
            x: X coordinate
            y: Y coordinate
            dir_x: X component of direction vector
            dir_y: Y component of direction vector
            expiry_time: Time when pedestrian should be removed
        """
        self.x = x
        self.y = y
        # Normalize direction vector
        length = math.sqrt(dir_x**2 + dir_y**2)
        self.dir_x = dir_x / length if length > 0 else 0
        self.dir_y = dir_y / length if length > 0 else 0
        self.expiry_time = expiry_time
    
    def step(self, time_delta: float, current_time: float) -> bool:
        """
        Step the pedestrian forward.
        
        Args:
            time_delta: Time step in seconds
            current_time: Current simulation time
            
        Returns:
            True if pedestrian should be removed (expired), False otherwise
        """
        if current_time >= self.expiry_time:
            return True
        
        # Move pedestrian in direction
        self.x += self.dir_x * self.PEDESTRIAN_VELOCITY * time_delta
        self.y += self.dir_y * self.PEDESTRIAN_VELOCITY * time_delta
        
        return False

class Car:
    """Represents a car with position, speed, direction, and turn angle."""
    
    CAR_MAX_VELOCITY = 15.0  # units per second
    CAR_ACCELERATION = 10.0 # units per second^2
    
    def __init__(self, x: float, y: float, dir_x: float, dir_y: float, world: World):
        """
        Initialize a car.
        
        Args:
            x: X coordinate
            y: Y coordinate
            dir_x: X component of direction vector
            dir_y: Y component of direction vector
            world: The simulation world object
        """
        self.x = x
        self.y = y
        # Normalize direction vector
        length = math.sqrt(dir_x**2 + dir_y**2)
        self.dir_x = dir_x / length if length > 0 else 0
        self.dir_y = dir_y / length if length > 0 else 0
        self.speed = 0.0
        self.turn_angle = 0.0  # radians
        self.controller = CarControl(world) # Initialize CarControl here
        self.world = world
    
    def step(self, time_delta: float, obstacles_info: dict, current_time: float):
        """
        Step the car forward based on speed, direction, and turn angle, with controls for turn and velocity.
        
        Args:
            time_delta: Time step in seconds
            obstacles_info: Dictionary containing collision and proximity information
            current_time: Current simulation time
        """
        car_info = self.world.get_car_info(self.x, self.y, self.dir_x, self.dir_y)
        control_output = self.controller.step(obstacles_info, time_delta, current_time, car_info)
        target_turn_angle = control_output["target_turn_angle"]
        velocity_control_action = control_output["velocity_action"]
        
        # Apply turn angle control
        self.turn_angle = (self.turn_angle + target_turn_angle) / 2.0

        # Apply velocity control
        if velocity_control_action == "drive":
            self.speed += self.CAR_ACCELERATION * time_delta
        elif velocity_control_action == "brake":
            self.speed -= self.CAR_ACCELERATION * time_delta
        
        # Clip speed to valid range
        self.speed = max(0.0, min(self.speed, self.CAR_MAX_VELOCITY))
        
        # Apply turn angle to direction
        cos_angle = math.cos(self.turn_angle * time_delta)
        sin_angle = math.sin(self.turn_angle * time_delta)
        
        new_dir_x = self.dir_x * cos_angle - self.dir_y * sin_angle
        new_dir_y = self.dir_x * sin_angle + self.dir_y * cos_angle
        
        # Normalize the new direction
        length = math.sqrt(new_dir_x**2 + new_dir_y**2)
        self.dir_x = new_dir_x / length if length > 0 else self.dir_x
        self.dir_y = new_dir_y / length if length > 0 else self.dir_y
        
        # Move car forward
        self.x += self.dir_x * self.speed * time_delta
        self.y += self.dir_y * self.speed * time_delta

class Simulator:
    """Main simulator class that manages cars and pedestrians."""
    
    # Detection thresholds
    THRESHOLD_DISTANCE = 20.0
    CAR_COLLISION_THRESHOLD = 4.0
    SIDE_PROXIMITY_THRESHOLD = 7.0
    FRONT_PROXIMITY_THRESHOLD = 20.0
    
    def __init__(self):
        self.world = World()
        self.cars: List[Car] = []
        self.pedestrians: List[Pedestrian] = []
        self.current_time = 0.0
        
        # Initialize pygame for rendering
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Car Simulator")
        
        # Colors
        self.DARK_GREY = (64, 64, 64)
        self.LIGHT_GREY = (128, 128, 128)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.VERY_LIGHT_GREY = (225, 225, 225)

    def step(self, time_delta: float):
        """
        Step all cars and pedestrians forward in time.
        
        Args:
            time_delta: Time step in seconds
        """
        self.current_time += time_delta
        
        # Step all cars
        for car in self.cars:
            obstacles_info = self.detect_collisions_and_proximity(car)
            car.step(time_delta, obstacles_info, self.current_time)
        
        # Step all pedestrians and remove expired ones
        self.pedestrians = [ped for ped in self.pedestrians if not ped.step(time_delta, self.current_time)]
    
    def add_car(self, x: float, y: float, dir_x: float, dir_y: float):
        """
        Add a car at the specified location with zero speed and turn angle.
        
        Args:
            x: X coordinate
            y: Y coordinate
            dir_x: X component of direction vector
            dir_y: Y component of direction vector
        """
        car = Car(x, y, dir_x, dir_y, self.world)
        self.cars.append(car)
    
    def find_nearby_objects(self, car: Car) -> List[Tuple[str, Union[Car, Pedestrian], float]]:
        """
        Find all cars and pedestrians within THRESHOLD_DISTANCE of the given car.
        
        Args:
            car: The car to check around
            
        Returns:
            List of tuples containing (object_type, object, distance)
            where object_type is 'car' or 'pedestrian'
        """
        nearby_objects = []
        
        # Check other cars
        for other_car in self.cars:
            if other_car != car:  # Don't check against self
                distance = math.sqrt((car.x - other_car.x)**2 + (car.y - other_car.y)**2)
                if distance <= self.THRESHOLD_DISTANCE:
                    nearby_objects.append(('car', other_car, distance))
        
        # Check pedestrians
        for pedestrian in self.pedestrians:
            distance = math.sqrt((car.x - pedestrian.x)**2 + (car.y - pedestrian.y)**2)
            if distance <= self.THRESHOLD_DISTANCE:
                nearby_objects.append(('pedestrian', pedestrian, distance))
        
        return nearby_objects
    
    def detect_collisions_and_proximity(self, car: Car) -> dict:
        """
        Detect collisions and determine proximity of objects around a car.
        
        Args:
            car: The car to check around
            
        Returns:
            Dictionary containing collision and proximity information:
            - 'collision_risk': True if any object is within CAR_COLLISION_THRESHOLD
            - 'left_obstacle': True if object is within 5 units to the left
            - 'right_obstacle': True if object is within 5 units to the right
            - 'front_obstacle': True if object is within 5 units in front
        """
        nearby_objects = self.find_nearby_objects(car)
        
        result = {
            'collision_risk': False,
            'left_obstacle': False,
            'right_obstacle': False,
            'front_obstacle': False
        }
        
        for obj_type, obj, distance in nearby_objects:
            # Check for collision risk
            if distance <= self.CAR_COLLISION_THRESHOLD:
                result['collision_risk'] = True
            
            # Determine relative position
            # Calculate vector from car to object
            dx = obj.x - car.x
            dy = obj.y - car.y
            
            # Project this vector onto car's direction and perpendicular direction
            # Forward component (dot product with car direction)
            forward_component = dx * car.dir_x + dy * car.dir_y
            
            # Perpendicular component (cross product magnitude)
            perpendicular_component = dx * car.dir_y - dy * car.dir_x
            
            # Determine if object is in front (positive forward component)
            if forward_component/abs(perpendicular_component) > 1.5 and forward_component <= self.FRONT_PROXIMITY_THRESHOLD:
                result['front_obstacle'] = True
            elif abs(perpendicular_component) <= self.SIDE_PROXIMITY_THRESHOLD:
                if perpendicular_component > 0:  # Object is to the left
                    result['left_obstacle'] = True
                elif perpendicular_component < 0:  # Object is to the right
                    result['right_obstacle'] = True
            
        return result
    
    def generate_pedestrian(self):
        """Generate a pedestrian at a random crosswalk."""
        if not self.world.crosswalks:
            return  # No crosswalks available
        
        # For now, we'll create a simple pedestrian since crosswalks aren't implemented yet
        # This is a placeholder implementation
        crosswalk = random.choice(self.world.crosswalks)
        
        # Calculate start and end points of crosswalk
        start_x = crosswalk[0][0]
        start_y = crosswalk[0][1]
        end_x = crosswalk[1][0]
        end_y = crosswalk[1][1]
        
        dir_x = end_x - start_x
        dir_y = end_y - start_y
        
        distance = math.sqrt(dir_x**2 + dir_y**2)
        travel_time = distance / Pedestrian.PEDESTRIAN_VELOCITY
        
        pedestrian = Pedestrian(start_x, start_y, dir_x, dir_y, self.current_time + travel_time)
        self.pedestrians.append(pedestrian)
    
    def get_obs(self):
        """
        Returns 3D object representations for all objects.
        Not implemented yet as specified.
        """
        pass
    
    def render_2d(self):
        """Render the simulation using pygame."""
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Calculate center of screen
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        scale = 2.0  # Scale factor for rendering
        
        # Render lanes as dark grey lines
        right_lane_center_radius = self.world.ROAD_RADIUS + self.world.LANE_WIDTH / 2
        right_lane_outer_radius = self.world.ROAD_RADIUS + self.world.LANE_WIDTH
        left_lane_center_radius = self.world.ROAD_RADIUS + 3 * self.world.LANE_WIDTH / 2
        left_lane_outer_radius = self.world.ROAD_RADIUS + 2 * self.world.LANE_WIDTH
        circle_width = int(self.world.LANE_WIDTH * scale)
        # Draw right lane
        pygame.draw.circle(self.screen, self.DARK_GREY, 
                          (center_x, center_y), 
                          int(right_lane_outer_radius * scale), 
                          circle_width)
        pygame.draw.circle(self.screen, self.LIGHT_GREY, 
                          (center_x, center_y), 
                          int(right_lane_center_radius * scale), 
                          1)
        
        # Draw left lane
        pygame.draw.circle(self.screen, self.DARK_GREY, 
                          (center_x, center_y), 
                          int(left_lane_outer_radius * scale), 
                          circle_width
                          )
        pygame.draw.circle(self.screen, self.LIGHT_GREY, 
                          (center_x, center_y), 
                          int(left_lane_center_radius * scale), 
                          1
                          )
        # Draw lane divider
        pygame.draw.circle(self.screen, self.VERY_LIGHT_GREY, 
                          (center_x, center_y), 
                          int(((left_lane_center_radius + right_lane_center_radius) / 2) * scale) + 2, 
                          3)
        
        for crosswalk in self.world.crosswalks:
            start_x = center_x + int(crosswalk[0][0] * scale)
            start_y = center_y - int(crosswalk[0][1] * scale) # Reverse y axis for correct directions
            end_x = center_x + int(crosswalk[1][0] * scale)
            end_y = center_y - int(crosswalk[1][1] * scale) # Reverse y axis for correct directions
            pygame.draw.line(self.screen, self.LIGHT_GREY, (start_x, start_y), (end_x, end_y), 10)
        
        # Render cars
        for car in self.cars:
            # Car position
            car_screen_x = center_x + int(car.x * scale)
            car_screen_y = center_y - int(car.y * scale) # Reverse y axis for correct directions
            
            # Draw car as green circle
            pygame.draw.circle(self.screen, self.GREEN, (car_screen_x, car_screen_y), 8)
            
            # Draw direction line (longer blue line)
            dir_end_x = car_screen_x + int(car.dir_x * 20)
            dir_end_y = car_screen_y - int(car.dir_y * 20)
            pygame.draw.line(self.screen, self.BLUE, (car_screen_x, car_screen_y), 
                           (dir_end_x, dir_end_y), 3)
            
            # Draw turn angle line (shorter blue line)
            # Calculate turn direction based on turn angle
            turn_dir_x = car.dir_x * math.cos(car.turn_angle) - car.dir_y * math.sin(car.turn_angle)
            turn_dir_y = car.dir_x * math.sin(car.turn_angle) + car.dir_y * math.cos(car.turn_angle)
            turn_end_x = car_screen_x + int(turn_dir_x * 10)
            turn_end_y = car_screen_y - int(turn_dir_y * 10)
            pygame.draw.line(self.screen, self.BLUE, (car_screen_x, car_screen_y), 
                           (turn_end_x, turn_end_y), 2)
        
        # Render pedestrians
        for pedestrian in self.pedestrians:
            ped_screen_x = center_x + int(pedestrian.x * scale)
            ped_screen_y = center_y - int(pedestrian.y * scale) # Reverse y axis for correct directions
            
            # Draw pedestrian as red circle
            pygame.draw.circle(self.screen, self.RED, (ped_screen_x, ped_screen_y), 5)

        # Update display
        pygame.display.flip()
    
    def quit(self):
        """Clean up pygame resources."""
        pygame.quit()
