import math
import pygame
from pygame.locals import *
from OpenGL.GL import glBegin, glEnd, glClear, glEnable, glTranslatef, glRotatef, glMatrixMode, glLoadIdentity, glLoadMatrixf, \
    glVertex3fv, glColor3fv, \
        GL_LINES, GL_QUADS, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, \
        GL_VERTEX_SHADER, GL_DEPTH_TEST, \
        GL_MODELVIEW, GL_PROJECTION
from OpenGL.GLU import gluPerspective, gluLookAt
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from dataclasses import dataclass
from pyglm import glm

WORLD_RADIUS = 30
GROUND_Z = 0
ROAD_ELEVATION = 0.05
ROAD_MARKING_ELEVATION = 0.1
ENTITY_BASE_ELEVATION = 0.1

CAR_HEIGHT = 0.5
CAR_TOP_ELEVATION = ENTITY_BASE_ELEVATION + CAR_HEIGHT
CAR_LENGTH = 1.0
CAR_WIDTH = 0.5
CAR_CHAMFER = 0.1
CAR_CAMERA_HEIGHT = CAR_HEIGHT + 0.2

PEDESTRIAN_HEIGHT = 0.5
PEDESTRIAN_WIDTH = 0.2
PEDESTRIAN_HALF_WIDTH = PEDESTRIAN_WIDTH / 2
PEDESTRIAN_TOP_ELEVATION = ENTITY_BASE_ELEVATION + PEDESTRIAN_HEIGHT

LANE_MARKER_HALF_WIDTH = 0.05

CROSSWALK_MARK_THICKNESS = 0.1
CROSSWALK_MARK_SPACING = 0.1
CROSSWALK_MARK_WIDTH = 0.5

@dataclass
class CarDrawInfo:
    x: float
    y: float
    dir_x: float
    dir_y: float

    def get_camera_params(self):
        return (
            (self.x, self.y, CAR_CAMERA_HEIGHT), # Camera position
            (self.x + self.dir_x, self.y + self.dir_y, CAR_CAMERA_HEIGHT), # Camera target
            (0, 0, 1) # Up vector
        )
@dataclass
class PedestrianDrawInfo:
    x: float
    y: float

@dataclass
class RoadDrawInfo:
    x: float
    y: float
    inner_lane_radius: float
    outer_lane_radius: float

@dataclass
class CrosswalkDrawInfo:
    start_x: float
    start_y: float
    end_x: float
    end_y: float

def draw_quads(vertices, surfaces, colors):
    glBegin(GL_QUADS)
    for surface, color in zip(surfaces, colors):
        for vertex_index in surface:
            vertex = vertices[vertex_index]
            glColor3fv(color)
            glVertex3fv(vertex)
    glEnd()

def draw_car(x, y, dir_x, dir_y):
        forward_dx, forward_dy = dir_x * CAR_LENGTH / 2, dir_y * CAR_LENGTH / 2
        side_dx, side_dy = dir_y * CAR_WIDTH / 2, -dir_x * CAR_WIDTH / 2 # Right-side vector
        chamfer_factor = CAR_CHAMFER / (CAR_WIDTH / 2)
        vertices= (
            (x + forward_dx + side_dx, y + forward_dy + side_dy, ENTITY_BASE_ELEVATION), #front right
            (x + forward_dx - side_dx, y + forward_dy - side_dy, ENTITY_BASE_ELEVATION), #front left
            (x - forward_dx - side_dx, y - forward_dy - side_dy, ENTITY_BASE_ELEVATION), #back left
            (x - forward_dx + side_dx, y - forward_dy + side_dy, ENTITY_BASE_ELEVATION), #back right
            (x + forward_dx * chamfer_factor + side_dx, y + forward_dy * chamfer_factor + side_dy, CAR_TOP_ELEVATION), #front right
            (x + forward_dx * chamfer_factor - side_dx, y + forward_dy * chamfer_factor - side_dy, CAR_TOP_ELEVATION), #front left
            (x - forward_dx - side_dx, y - forward_dy - side_dy, CAR_TOP_ELEVATION), #back left
            (x - forward_dx + side_dx, y - forward_dy + side_dy, CAR_TOP_ELEVATION), #back right
        )
        surfaces = (
            (0,1,2,3),
            (3,2,6,7),
            (6,7,4,5),
            (4,5,1,0),
            (1,5,7,2),
            (4,0,3,6)
        )

        colors = [(0.3, 0.3, 0.9) for s in surfaces]
        colors[3] = (0.9, 0.9, 0.9) # Windshield color

        draw_quads(vertices, surfaces, colors)

def draw_pedestrian(x, y):
        vertices= (
            (x + PEDESTRIAN_HALF_WIDTH, y - PEDESTRIAN_HALF_WIDTH, ENTITY_BASE_ELEVATION),
            (x + PEDESTRIAN_HALF_WIDTH, y + PEDESTRIAN_HALF_WIDTH, ENTITY_BASE_ELEVATION),
            (x - PEDESTRIAN_HALF_WIDTH, y + PEDESTRIAN_HALF_WIDTH, ENTITY_BASE_ELEVATION),
            (x - PEDESTRIAN_HALF_WIDTH, y - PEDESTRIAN_HALF_WIDTH, ENTITY_BASE_ELEVATION),
            (x + PEDESTRIAN_HALF_WIDTH, y - PEDESTRIAN_HALF_WIDTH, PEDESTRIAN_TOP_ELEVATION),
            (x + PEDESTRIAN_HALF_WIDTH, y + PEDESTRIAN_HALF_WIDTH, PEDESTRIAN_TOP_ELEVATION),
            (x - PEDESTRIAN_HALF_WIDTH, y + PEDESTRIAN_HALF_WIDTH, PEDESTRIAN_TOP_ELEVATION),
            (x - PEDESTRIAN_HALF_WIDTH, y - PEDESTRIAN_HALF_WIDTH, PEDESTRIAN_TOP_ELEVATION),
        )
        surfaces = (
            (0,1,2,3),
            (3,2,6,7),
            (6,7,4,5),
            (4,5,1,0),
            (1,5,7,2),
            (4,0,3,6)
        )

        colors = [(0.9, 0.3, 0.3) for s in surfaces]

        draw_quads(vertices, surfaces, colors)

def draw_annulus(x, y, inner_radius, outer_radius, elevation, color, divisions):
    vertices = []
    surfaces = []
    colors = []

    for i in range(divisions):
        angle1 = 2 * math.pi * i / divisions
        angle2 = 2 * math.pi * (i + 1) / divisions

        # Inner ring vertices
        ix1 = x + inner_radius * math.cos(angle1)
        iy1 = y + inner_radius * math.sin(angle1)
        ix2 = x + inner_radius * math.cos(angle2)
        iy2 = y + inner_radius * math.sin(angle2)

        # Outer ring vertices
        ox1 = x + outer_radius * math.cos(angle1)
        oy1 = y + outer_radius * math.sin(angle1)
        ox2 = x + outer_radius * math.cos(angle2)
        oy2 = y + outer_radius * math.sin(angle2)

        # Vertices for the current quad segment
        v_base = len(vertices)
        vertices.extend([
            (ix1, iy1, elevation),
            (ox1, oy1, elevation),
            (ox2, oy2, elevation),
            (ix2, iy2, elevation)
        ])
        surfaces.append((v_base, v_base + 1, v_base + 2, v_base + 3))
        colors.append(color)

    draw_quads(vertices, surfaces, colors)



def draw_crosswalk(start_x, start_y, end_x, end_y):
    dx = end_x - start_x
    dy = end_y - start_y
    length = math.sqrt(dx**2 + dy**2)

    if length == 0:
        return # Avoid division by zero

    dir_x = dx / length
    dir_y = dy / length

    # Perpendicular vector for width
    perp_x = -dir_y
    perp_y = dir_x

    current_pos = 0.0
    mark_color = (0.9, 0.9, 0.9) # White

    while current_pos < length:
        # Calculate the center of the current mark
        mark_center_x = start_x + dir_x * current_pos
        mark_center_y = start_y + dir_y * current_pos

        # Calculate vertices for the current mark
        half_thickness = CROSSWALK_MARK_THICKNESS / 2
        half_width = CROSSWALK_MARK_WIDTH / 2

        v1_x = mark_center_x - dir_x * half_thickness - perp_x * half_width
        v1_y = mark_center_y - dir_y * half_thickness - perp_y * half_width

        v2_x = mark_center_x + dir_x * half_thickness - perp_x * half_width
        v2_y = mark_center_y + dir_y * half_thickness - perp_y * half_width

        v3_x = mark_center_x + dir_x * half_thickness + perp_x * half_width
        v3_y = mark_center_y + dir_y * half_thickness + perp_y * half_width

        v4_x = mark_center_x - dir_x * half_thickness + perp_x * half_width
        v4_y = mark_center_y - dir_y * half_thickness + perp_y * half_width

        vertices = [
            (v1_x, v1_y, ROAD_MARKING_ELEVATION),
            (v2_x, v2_y, ROAD_MARKING_ELEVATION),
            (v3_x, v3_y, ROAD_MARKING_ELEVATION),
            (v4_x, v4_y, ROAD_MARKING_ELEVATION)
        ]
        surfaces = [(0, 1, 2, 3)]
        colors = [mark_color]
        draw_quads(vertices, surfaces, colors)

        current_pos += CROSSWALK_MARK_THICKNESS + CROSSWALK_MARK_SPACING

def draw_road(x, y, inner_lane_radius, outer_lane_radius):
    #ground
    draw_quads(
        [
            (WORLD_RADIUS, WORLD_RADIUS, GROUND_Z),
            (WORLD_RADIUS, -WORLD_RADIUS, GROUND_Z),
            (-WORLD_RADIUS, -WORLD_RADIUS, GROUND_Z),
            (-WORLD_RADIUS, WORLD_RADIUS, GROUND_Z),
        ],
        [(0, 1, 2, 3)],
        [(0.1, 0.6, 0.1)]
    )

    #road
    road_color = (0.3, 0.3, 0.3)
    draw_annulus(x, y, inner_lane_radius, outer_lane_radius, ROAD_ELEVATION, road_color, 64)

    #lane marking
    lane_marking_color = (0.9, 0.9, 0.9)
    middle_radius = (inner_lane_radius + outer_lane_radius) / 2
    lane_marking_inner_radius = middle_radius - LANE_MARKER_HALF_WIDTH
    lane_marking_outer_radius = middle_radius + LANE_MARKER_HALF_WIDTH
    draw_annulus(x, y, lane_marking_inner_radius, lane_marking_outer_radius, ROAD_MARKING_ELEVATION, lane_marking_color, 64)

def draw_all(cars: list[CarDrawInfo], pedestrians: list[PedestrianDrawInfo], crosswalks: list[CrosswalkDrawInfo], road: RoadDrawInfo):
    draw_road(road.x, road.y, road.inner_lane_radius, road.outer_lane_radius)
    for car in cars:
        draw_car(car.x, car.y, car.dir_x, car.dir_y)
    for pedestrian in pedestrians:
        draw_pedestrian(pedestrian.x, pedestrian.y)
    for crosswalk in crosswalks:
        draw_crosswalk(crosswalk.start_x, crosswalk.start_y, crosswalk.end_x, crosswalk.end_y)

class Renderer:
    def __init__(self) -> None:
        pygame.init()
        display = (800,600)
        self.screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        glEnable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_PROJECTION)
        persp = glm.perspective(45, (display[0]/display[1]), 0.1, 50.0)
        glLoadMatrixf([persp[i][j] for i in range(4) for j in range(4)])
    
    def render_step(self, ref_car: CarDrawInfo, cars: list[CarDrawInfo], pedestrians: list[PedestrianDrawInfo], crosswalks: list[CrosswalkDrawInfo], road: RoadDrawInfo): 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        self.set_camera(ref_car.get_camera_params())
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_all(cars, pedestrians, crosswalks, road)
        pygame.display.flip()
        return pygame.PixelArray(self.screen)

    def set_camera(self, look_at_args):
        glMatrixMode(GL_MODELVIEW)
        view_matrix = glm.lookAt(*look_at_args)
        glLoadMatrixf([view_matrix[i][j] for i in range(4) for j in range(4)])


def test_render():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glEnable(GL_DEPTH_TEST)
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    
    # glLoadIdentity()
    # glRotatef(90, 1, 0, 0)
    # glTranslatef(0.0, -7.0, 1)
    # glRotatef(180, 0, 1, 0)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    view_matrix = glm.lookAt(glm.vec3(0, -7, 1), glm.vec3(0, 0, 1), glm.vec3(0, 0, 1))
    glLoadMatrixf([view_matrix[i][j] for i in range(4) for j in range(4)])
    glMatrixMode(GL_PROJECTION)
    persp = glm.perspective(1, (display[0]/display[1]), 0.1, WORLD_RADIUS)
    glLoadMatrixf([persp[i][j] for i in range(4) for j in range(4)])
    glMatrixMode(GL_MODELVIEW)

    t = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        t += 0.05
        road = RoadDrawInfo(0, 0, 3.0, 6.0)
        
        cars = [
            CarDrawInfo(r * math.cos(t * v), r * math.sin(t * v), -math.sin(t * v), math.cos(t * v))
            for r, v in ((road.inner_lane_radius + CAR_WIDTH, 1.0), (road.outer_lane_radius - CAR_WIDTH, 0.2))
        ]
        pedestrians = [PedestrianDrawInfo(t - 2 * o, 0) for o in range(20)]
        crosswalks = [CrosswalkDrawInfo(2, 0, 10, 0)]
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        view_matrix = glm.lookAt(*cars[0].get_camera_params())
        glLoadMatrixf([view_matrix[i][j] for i in range(4) for j in range(4)])
        draw_all(cars, pedestrians, crosswalks, road)
        pygame.display.flip()
        pygame.time.wait(50)

if __name__ == "__main__":
    test_render()