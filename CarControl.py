import random
from enum import Enum
from World import World

class CommandState(Enum):
    """
    Represents the command states for car control.
    """
    STAY_IN_LANE = 1
    CHANGE_TO_RIGHT_LANE = 2
    CHANGE_TO_LEFT_LANE = 3

class CarControl:
    """
    Controls the behavior of a car based on command states and surrounding obstacles.
    """
    
    STEERING_GAIN = 0.1  # Proportional gain for steering
    MAX_ANGLE = 0.4
    def __init__(self, world: World):
        """
        Initializes the CarControl with a world instance.
        """
        self.world = world
        self.current_command_state = random.choice(list(CommandState))
        self.next_command_change_time = random.uniform(10, 15)
    
    def step(self, obstacles_info: dict, time_delta: float, current_time: float, car_info: dict) -> dict:
        """
        Calculates the target turn angle and velocity action for the car.
        
        Args:
            obstacles_info: Dictionary containing collision and proximity information from Simulator.detect_collisions_and_proximity.
            time_delta: Time step in seconds.
            current_time: Current simulation time.
            car_info: Dictionary containing car's lane and intersection information
            
        Returns:
            Dictionary containing:
            "target_turn_angle": angle
            "velocity_action": "brake" or "drive"
        """
        # Update command state if applicable
        if current_time >= self.next_command_change_time:
            self.current_command_state = random.choice(list(CommandState))
            self.next_command_change_time = current_time + random.uniform(10, 15)
            print(f"command={self.current_command_state}")
            
        target_displacement = 0.0
        
        # Determine target displacement based on command state and obstacles
        if self.current_command_state == CommandState.STAY_IN_LANE:
            target_displacement = 0.0
        elif self.current_command_state == CommandState.CHANGE_TO_RIGHT_LANE:
            if car_info['lane'] == "Right":
                target_displacement = 0.0 # Already in right lane
            elif car_info['lane'] == "Left" and not obstacles_info['right_obstacle']:
                # Aim for the center of the right lane
                target_displacement = self.world.LANE_WIDTH # Relative to left lane center, move right
            else: # Cannot change lane right due to obstacle or not in a lane
                target_displacement = 0.0 # Stay in current lane
        elif self.current_command_state == CommandState.CHANGE_TO_LEFT_LANE:
            if car_info['lane'] == "Left":
                target_displacement = 0.0 # Already in left lane
            elif car_info['lane'] == "Right" and not obstacles_info['left_obstacle']:
                # Aim for the center of the left lane
                target_displacement = -self.world.LANE_WIDTH # Relative to right lane center, move left
            else: # Cannot change lane left due to obstacle or not in a lane
                target_displacement = 0.0 # Stay in current lane

        # Calculate steering angle based on displacement error
        lane_position = car_info['lane_position']
        car_angle = car_info['angle_relative_to_road']
        if lane_position is not None:
            displacement_error = lane_position - target_displacement
            target_turn_angle = displacement_error * self.STEERING_GAIN
            target_turn_angle = (1 if target_turn_angle >= 0  else -1) * min(abs(target_turn_angle), self.MAX_ANGLE) # Constrain target angle
            corrected_turn_angle = target_turn_angle - car_angle # Account for where the car is already facing to avoid overcorrection and oscilations
            corrected_turn_angle = (1 if corrected_turn_angle >= 0  else -1) * min(abs(corrected_turn_angle), self.MAX_ANGLE)
            #print(f"d={displacement_error}, t={target_turn_angle}, car={car_info['angle_relative_to_road'] }, corrected={corrected_turn_angle}")
        else: # Car is not in a defined lane, try to get it back to center
            corrected_turn_angle = 0.0 # No steering if not in a lane
            # A more sophisticated approach would be to steer towards the nearest lane.
            # For simplicity, we assume the car will eventually find its way back if it strays.
        
        # Determine velocity action based on front obstacle
        velocity_action = "drive"
        if obstacles_info['front_obstacle']:
            velocity_action = "brake"

        return {
            "target_turn_angle": corrected_turn_angle,
            "velocity_action": velocity_action
        }
