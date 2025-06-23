import random
from time import sleep
from DataClasses import CarInfo, ObstaclesInfo
from Render import CarDrawInfo, Renderer
from Simulator import Car, Simulator
import numpy as np
import torch
from torch.utils.data import Dataset

from World import World


class DatasetExtractor(Dataset):
    """Extract a dataset from a running simulation"""
    def __init__(self, num_samples, sim_time_delta = 0.1, collection_interval = 0.5):
        self.sim = Simulator()
        self.renderer = Renderer()
        self._set_up_simulator_default()
        self.collection_interval = collection_interval
        self.time_delta = sim_time_delta
        self.num_samples = num_samples
        self.current_sample_count = 0
        self.next_collection_time = self.collection_interval # give simulator some time to get away from starting state

    def reset(self):
        self.sim.reset()
        self._set_up_simulator_default()
        self.next_collection_time = self.collection_interval

    def __len__(self):
        return self.num_samples

    def _set_up_simulator_default(self, number_of_cars=5):
        self.sim.add_default_cars(number_of_cars)

    TARGET_VECTOR_SIZE = 6
    def get_target_vector(self, car: Car):
        """
        Target vector elements:
        1. Obstacle at right: 0 if no, 1 if yes
        2. Obstacle at front: ^
        3. Obstable at left: ^
        4: Which lane: 0 if left, 1 if right
        5: Car angle relative to road: (-pi/3, pi/3) gets mapped to (-1, 1)
        6: Lane position: (-LANE_WIDTH/2, LANE_WIDTH/2) gets mapped to (-1, 1)
        """
        obstacles_info, car_info = self.sim.get_car_inputs(car)
        return np.array([
            1 if obstacles_info.right_obstacle else 0,
            1 if obstacles_info.front_obstacle else 0,
            1 if obstacles_info.left_obstacle else 0,
            1 if car_info.lane == "Right" else (0 if car_info.lane == "Left" else 0.5),
            np.clip(car_info.angle_relative_to_road / (np.pi / 3), -1, 1),
            np.clip(car_info.lane_position / (World.LANE_WIDTH / 2), -1, 1),
            ], dtype=np.float16)

    def _get_next_sample(self):
        """Steps the simulation and returns a single (frame, target) pair."""
        self.sim.step(self.time_delta)
        while self.sim.current_time < self.next_collection_time:
            self.sim.step(self.time_delta)

        
        self.next_collection_time = self.sim.current_time + self.collection_interval
        car_draw_infos, ped_draw_infos, walk_draw_infos, road_draw_info = self.sim.get_draw_infos()

        car = random.choice(self.sim.cars)
        car_draw_info = CarDrawInfo(car.x, car.y, car.dir_x, car.dir_y)
        frame = self.renderer.render_step(car_draw_info, car_draw_infos, ped_draw_infos, walk_draw_infos, road_draw_info)
        target = self.get_target_vector(car)
        
        self.current_sample_count += 1
        return frame, target

    def __getitem__(self, idx):
        return self._get_next_sample()

    def test_vis(self, time):
        while self.sim.current_time < time:
            self.sim.step(self.time_delta)
            car_draw_infos, ped_draw_infos, walk_draw_infos, road_draw_info = self.sim.get_draw_infos()
            self.renderer.render_step(car_draw_infos[0], car_draw_infos, ped_draw_infos, walk_draw_infos, road_draw_info)
            sleep(0.05)

    def run(self, obstacle_sample_proportion=None):
        samples = []
        obstacle_samples = 0 # If obstacle_sample_proportion is set, this is used to ensure a certain number of samples involve obstacles
        next_collection_time = self.collection_interval + 3 # give simulator some time to get away from starting state
        while len(samples) < self.num_samples:
            self.sim.step(self.time_delta)
            if self.sim.current_time >= next_collection_time:
                next_collection_time = self.sim.current_time + self.collection_interval
                car_draw_infos, ped_draw_infos, walk_draw_infos, road_draw_info = self.sim.get_draw_infos()
                for car in self.sim.cars:
                    car_draw_info = CarDrawInfo(car.x, car.y, car.dir_x, car.dir_y)
                    frame = self.renderer.render_step(car_draw_info, car_draw_infos, ped_draw_infos, walk_draw_infos, road_draw_info)
                    target = self.get_target_vector(car)
                    if obstacle_sample_proportion is not None and obstacle_samples < len(samples) * obstacle_sample_proportion:
                        # We don't have enough obstacle samples currently, so if the current sample doesnt have obstacles, ignore it
                        if not any(target[:3]):
                            continue # no obstacles
                    samples.append((frame, target))
        self.summarize_sample_distribution(samples)
        return samples[:self.num_samples]
    
    @staticmethod
    def summarize_sample_distribution(samples):
        logit_distribution = [[0, 0] for _ in range(4)] # true/false values for each logit
        for s in samples:
            t = s[1]
            for i in range(4):
                if t[i] > 0.5:
                    logit_distribution[i][0] += 1
                else:
                    logit_distribution[i][1] += 1
        for distribution, name in zip(logit_distribution, ["Right Obstacle", "Front Obstacle", "Left Obstacle", "Right/Left Lane"]):
            print(f"{name}: {distribution[0]} True, {distribution[1]} false")

if __name__ == "__main__":
    extractor = DatasetExtractor(1000) # num_samples for test_vis
    extractor.test_vis(10)
    print("collecting regular distribution")
    result = extractor.run()
    print("collecting filtered distribution")
    result = extractor.run(0.5)
    print(np.max(result[0][0]))
    sleep(1)
    print("showing extracted image...")
    import pygame
    image_surface = pygame.surfarray.make_surface(result[1][0])
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    screen.blit(image_surface, (0, 0))
    pygame.display.flip()
    sleep(10)