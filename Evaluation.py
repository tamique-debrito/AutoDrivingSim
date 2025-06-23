import pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pygame

import CarControl
from Dataset import DatasetExtractor
from Vision import SimpleCNN, CarDataset, ModelTrainer, StreamingCarDataset
from Simulator import Simulator, Car
from DataClasses import ObstaclesInfo, CarInfo
from CarControl import CarControl
from Render import Renderer, CarDrawInfo
from World import World

"""
A variety of evaluation functionality for both the simulator and the result of inserting an autonomous agent into the simulation
"""

class AutonomousSimulator(Simulator):
    def __init__(self, vision_model: SimpleCNN):
        super().__init__()
        self.vision_model = vision_model
        self.autonomous_car = None # Will store the actual autonomous car object
        self.renderer = Renderer() # Renderer for generating frames for the vision model
        self.vision_model.eval() # Set model to evaluation mode

    def set_autonomous_car(self, index = 0):
        self.autonomous_car = self.cars[index]

    def add_autonomous_car(self, x: float, y: float, dir_x: float, dir_y: float):
        """
        Add a car to the simulator and designate it as the autonomous one.
        """
        super().add_car(x, y, dir_x, dir_y)
        self.autonomous_car = self.cars[-1] # The last added car is the autonomous one

    def get_car_inputs(self, car: Car):
        """
        Overrides the base Simulator's get_car_inputs to use the vision model
        for the autonomous car, and ground truth for others.
        """
        if car == self.autonomous_car:
            # Get all draw infos from the simulator's current state
            cars_draw_info, pedestrians_draw_info, crosswalks_draw_info, road_draw_info = self.get_draw_infos()
            
            # Create CarDrawInfo for the autonomous car based on its current state
            autonomous_car_draw_info = CarDrawInfo(car.x, car.y, car.dir_x, car.dir_y)
            
            # Render the frame from the autonomous car's perspective
            frame = self.renderer.render_step(
                autonomous_car_draw_info, 
                cars_draw_info, 
                pedestrians_draw_info, 
                crosswalks_draw_info, 
                road_draw_info
            )
            
            # Process the frame for the vision model
            # Convert HWC (height, width, channels) to NCHW (batch, channels, height, width)
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0) # Add batch dimension

            # Pass through vision model
            with torch.no_grad():
                model_output = self.vision_model(frame_tensor).squeeze(0)

            # Interpret model output
            # ObstaclesInfo (4 logits)
            logits_output = model_output[:4]
            # Apply sigmoid and threshold to get boolean values
            booleans = (F.sigmoid(logits_output) > 0.5).float()
            
            obstacles_info = ObstaclesInfo(
                collision_risk=False, # Not used
                left_obstacle=bool(booleans[0].item()),
                right_obstacle=bool(booleans[1].item()),
                front_obstacle=bool(booleans[2].item())
            )

            # CarInfo (2 continuous values)
            continuous_output = model_output[4:]
            angle_relative_to_road_unnormalized = continuous_output[0].item()
            lane_position_unnormalized = continuous_output[1].item()
            angle_relative_to_road = angle_relative_to_road_unnormalized * (np.pi / 3)
            lane_position = lane_position_unnormalized * (World.LANE_WIDTH / 2)

            # Placeholder values for lane and distance_from_center as model doesn't output them
            car_info = CarInfo(
                lane_position=lane_position,
                lane="Right" if bool(booleans[3].item()) else "Left", # Placeholder
                distance_from_center=0.0, # Placeholder
                angle_relative_to_road=angle_relative_to_road
            )

            #print(f"vision inference: {repr(obstacles_info)}, {repr(car_info)}")
            
            return obstacles_info, car_info
        else:
            # For non-autonomous cars, use the superclass's method (ground truth)
            return super().get_car_inputs(car)
            
    def quit(self):
        """Clean up pygame resources for both simulator and renderer."""
        super().quit()
        if self.renderer:
            pass # Renderer does not have a separate quit method, pygame.quit() is handled by super().quit()

def evaluate_vision(model: SimpleCNN, num_samples: int = 100):
    print("Collecting evaluation data...")
    extractor = DatasetExtractor(num_samples, sim_time_delta=0.05, collection_interval=0.0) # don't skip any timesteps
    collected_data = extractor.run()
    print(f"Collected {len(collected_data)} samples for evaluation.")

    if not collected_data:
        print("No data collected for evaluation. Exiting.")
        return

    dataset = CarDataset(collected_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False) # Batch size of 1 for evaluating each sample

    model.eval() # Set model to evaluation mode
    
    losses = []
    accuracies = []

    criterion_logits = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()

    with torch.no_grad(): # Disable gradient calculation during evaluation
        for i, (frames, targets) in enumerate(dataloader):
            outputs = model(frames)

            outputs_logits = outputs[:, :4]
            targets_logits = targets[:, :4]
            outputs_mse = outputs[:, 4:]
            targets_mse = targets[:, 4:]

            loss_logits = criterion_logits(outputs_logits, targets_logits)
            loss_mse = criterion_mse(outputs_mse, targets_mse)

            loss = loss_logits + loss_mse
            losses.append(loss.item())

            predicted_logits = (torch.sigmoid(outputs_logits) > 0.5).float()
            correct_predictions = (predicted_logits == targets_logits).sum().item()
            total_predictions = targets_logits.numel()
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            accuracies.append(accuracy)

    print("Evaluation complete. Plotting results...")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Loss Over Samples")
    plt.xlabel("Sample Index")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title("Accuracy Over Samples (Logits)")
    plt.xlabel("Sample Index")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()


def test_vision_model(model: SimpleCNN):
    print("Starting autonomous car simulation...")
    which_car = 0
    
    # Initialize the autonomous simulator
    autonomous_sim = AutonomousSimulator(model)
    autonomous_sim.add_default_cars(5)

    time_delta = 0.05 # Simulation time step
    total_sim_time = 100 # Run for 100 seconds (adjust as needed for test duration)

    
    while autonomous_sim.current_time < 6:
        autonomous_sim.step(time_delta) # allow things to progress past initialization
        car_draw_infos, ped_draw_infos, walk_draw_infos, road_draw_info = autonomous_sim.get_draw_infos()
        autonomous_sim.renderer.render_step(car_draw_infos[which_car], car_draw_infos, ped_draw_infos, walk_draw_infos, road_draw_info)

    autonomous_sim.set_autonomous_car(which_car)

    print(f"Running simulation for {total_sim_time} seconds...")
    while autonomous_sim.current_time < total_sim_time:
        autonomous_sim.step(time_delta)
        pygame.time.wait(10)

    print("Autonomous car simulation complete.")
    autonomous_sim.quit()
    print("Simulation resources cleaned up.")


if __name__ == "__main__":
    eval_loss = False
    use_streaming_dataset = True
    num_training_samples = 2000

    collected_training_data = None
    if not use_streaming_dataset:
        try:
            # Attempt to load existing data
            train_data_path = "training_data.pkl"
            with open(train_data_path, "rb") as f:
                collected_training_data = pickle.load(f)
            print(f"Training data loaded from {train_data_path}")
        except FileNotFoundError:
            print(f"{train_data_path} not found. Collecting new training data...")
            extractor = DatasetExtractor(num_training_samples, sim_time_delta=0.05, collection_interval=0)
            collected_training_data = extractor.run() #, obstacle_sample_proportion=0.05)
            print(f"Collected {len(collected_training_data)} training samples.")

            # Save the newly collected training data
            with open(train_data_path, "wb") as f:
                pickle.dump(collected_training_data, f)
            print(f"Training data saved to {train_data_path}")

    if collected_training_data or use_streaming_dataset:
        print("Creating training dataset and dataloader...")
        if use_streaming_dataset:
            training_dataset = StreamingCarDataset(DatasetExtractor(num_training_samples, sim_time_delta=0.05, collection_interval=0))
        else:
            training_dataset = CarDataset(collected_training_data)
        training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=32, shuffle=True)
        print(f"Training dataset and dataloader created. (streaming={use_streaming_dataset})")

        print("Initializing model for training...")
        model_to_evaluate = SimpleCNN()
        print("Model initialized.")

        # Define model save path
        model_path = "simple_cnn_model.torch"

        # Attempt to load pre-trained model
        try:
            model_to_evaluate.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            CarControl.COMMAND_CHANGE_INTERVAL_LOWER = 3
            CarControl.COMMAND_CHANGE_INTERVAL_UPPER = 5
            print(f"Model file {model_path} not found. Training a new model...")
            print("Starting training...")
            trainer = ModelTrainer(model_to_evaluate, training_dataloader, learning_rate=0.001)
            for epoch in range(15):
                trainer.run_epoch(epoch)
                print(f"Training sim time: {training_dataset.data_extractor.sim.current_time} ({training_dataset.data_extractor.current_sample_count} samples)")
            print("Training complete.")
            torch.save(model_to_evaluate.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        CarControl.COMMAND_CHANGE_INTERVAL_LOWER = 10
        CarControl.COMMAND_CHANGE_INTERVAL_UPPER = 15
        # Now evaluate the trained model
        if eval_loss:
            evaluate_vision(model_to_evaluate, num_samples=500)

        # Test the vision model in a full simulation
        test_vision_model(model_to_evaluate)
    else:
        print("No training data collected. Cannot proceed with training and evaluation.")