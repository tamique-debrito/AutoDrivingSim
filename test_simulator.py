from imaplib import Commands
import time
import pygame
from CarControl import CommandState
from Simulator import Simulator

def test_simulator():
    """Test the simulator with a simple simulation loop."""
    
    # Create simulator
    sim = Simulator()
    
    # Add some cars at different positions
    sim.add_car(105, 0, 0, -1)
    sim.cars[0].speed = 8.0 
    
    sim.add_car(0, 115, 1, 0)
    sim.cars[1].speed = 6.0
    sim.cars[1].turn_angle = 0.1
    
    sim.add_car(-105, 0, 0, 1)
    sim.cars[2].speed = 10.0
    
    print("Simulator initialized with 3 cars")
    print("Press 'q' to quit, 'p' to generate pedestrian")
    
    # Simulation loop
    running = True
    clock = time.time()
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == ord('q'):
                    running = False
                elif event.key == ord('p'):
                    sim.generate_pedestrian()
                    print("Generated pedestrian")
        
        # Calculate time delta
        current_time = time.time()
        time_delta = current_time - clock
        clock = current_time
        
        # Cap time delta to prevent large jumps
        time_delta = min(time_delta, 0.1)
        
        # Step simulation
        sim.step(time_delta)
        
        # Render
        sim.render()
        
        # Small delay to control frame rate
        time.sleep(0.016)  # ~60 FPS
    
    # Clean up
    sim.quit()
    print("Simulation ended")

def test_car_movement():
    """Test car movement and physics."""
    sim = Simulator()
    
    # Add a car that will drive in a circle
    sim.add_car(100, 0, 0, -1)
    car = sim.cars[0]
    car.speed = 5.0
    car.controller.current_command_state = CommandState.STAY_IN_LANE
    car.turn_angle = 0.5  # Turn left
    
    print("Testing car movement - car should drive in a circle")
    print("Press 'q' to quit")
    
    running = True
    clock = time.time()
    
    while running:
        if 10 <= sim.current_time <= 20:
            if sim.current_time <= 10.1: print("command to switch to left")
            car.controller.current_command_state = CommandState.CHANGE_TO_LEFT_LANE
        elif 20 <= sim.current_time <= 30:
            if sim.current_time <= 20.1: print("command to switch to right")
            car.controller.current_command_state = CommandState.CHANGE_TO_RIGHT_LANE
        else:
            car.controller.current_command_state = CommandState.STAY_IN_LANE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == ord('q'):
                    running = False
        
        current_time = time.time()
        time_delta = current_time - clock
        clock = current_time
        time_delta = min(time_delta, 0.1)
        
        sim.step(time_delta)
        sim.render()
        time.sleep(0.016)
    
    sim.quit()
    print("Car movement test ended")

def test_pedestrian_generation():
    """Test pedestrian generation and movement."""
    sim = Simulator()
    
    print("Testing pedestrian generation")
    print("Press 'p' to generate pedestrian, 'q' to quit")
    
    running = True
    clock = time.time()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == ord('q'):
                    running = False
                elif event.key == ord('p'):
                    sim.generate_pedestrian()
                    print(f"Generated pedestrian. Total pedestrians: {len(sim.pedestrians)}")
        
        current_time = time.time()
        time_delta = current_time - clock
        clock = current_time
        time_delta = min(time_delta, 0.1)
        
        sim.step(time_delta)
        sim.render()
        time.sleep(0.016)
    
    sim.quit()
    print("Pedestrian test ended")

if __name__ == "__main__":
    print("Car Simulator Test")
    print("Choose a test:")
    print("1. Full simulator test")
    print("2. Car movement test")
    print("3. Pedestrian generation test")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        test_simulator()
    elif choice == "2":
        test_car_movement()
    elif choice == "3":
        test_pedestrian_generation()
    else:
        print("Invalid choice. Running full simulator test.")
        test_simulator() 