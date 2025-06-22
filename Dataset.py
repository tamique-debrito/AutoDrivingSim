from time import sleep
from Render import CarDrawInfo, Renderer
from Simulator import Car, Simulator
import numpy as np

from World import World


class DatasetExtractor:
    """Extract a dataset from a running simulation"""
    def __init__(self, sim_time_delta = 0.1, collection_interval = 0.5):
        self.sim = Simulator()
        self.renderer = Renderer()
        self._set_up_simulator_default()
        self.collection_interval = collection_interval
        self.time_delta = sim_time_delta

    def _set_up_simulator_default(self):
        self.sim.add_car(105, 0, 0, -1)
        self.sim.cars[0].speed = 8.0 
        
        self.sim.add_car(0, 115, 1, 0)
        self.sim.cars[1].speed = 6.0
        self.sim.cars[1].turn_angle = 0.1
        
        self.sim.add_car(-105, 0, 0, 1)
        self.sim.cars[2].speed = 10.0

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
            1 if obstacles_info['right_obstacle'] else 0,
            1 if obstacles_info['front_obstacle'] else 0,
            1 if obstacles_info['left_obstacle'] else 0,
            1 if car_info['lane'] == "Right" else (0 if car_info['lane'] == "Left" else 0.5),
            car_info['angle_relative_to_road'] / (np.pi / 3),
            car_info['lane_position'] / (World.LANE_WIDTH / 2),
            ], dtype=np.float16)

    def test_vis(self, time):
        while self.sim.current_time < time:
            self.sim.step(self.time_delta)
            car_draw_infos, ped_draw_infos, walk_draw_infos, road_draw_info = self.sim.get_draw_infos()
            self.renderer.render_step(car_draw_infos[0], car_draw_infos, ped_draw_infos, walk_draw_infos, road_draw_info)


    def run(self, samples_to_collect):
        samples = []
        next_collection_time = self.collection_interval + 5 # give simulator some time to get away from starting state
        while len(samples) < samples_to_collect:
            self.sim.step(self.time_delta)
            if self.sim.current_time >= next_collection_time:
                next_collection_time = self.sim.current_time + self.collection_interval
                car_draw_infos, ped_draw_infos, walk_draw_infos, road_draw_info = self.sim.get_draw_infos()
                for car in self.sim.cars:
                    car_draw_info = CarDrawInfo(car.x, car.y, car.dir_x, car.dir_y)
                    frame = self.renderer.render_step(car_draw_info, car_draw_infos, ped_draw_infos, walk_draw_infos, road_draw_info)
                    target = self.get_target_vector(car)
                    samples.append((frame, target))

        return samples[:samples_to_collect]

if __name__ == "__main__":
    extractor = DatasetExtractor()
    result = extractor.run(50)