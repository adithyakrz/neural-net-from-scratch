import pygame
import sys
import math

pygame.init()

WIDTH, HEIGHT = 900, 700

screen = pygame.display.set_mode((WIDTH,HEIGHT))

pygame.display.set_caption("Neural Network 3D Visualizer")

clock = pygame.time.Clock()
angle = 0.0
running = True
input_nodes = [(-150, -100, 100), (-150, 0, 100), (-150, 100, 100)]
output_nodes = [(150, -60, -100), (150, 60, -100)]

def rotate_y(x, y, z, angle):
  rx = x * math.cos(angle) - z * math.sin(angle)
  rz = x * math.sin(angle) + z * math.cos(angle)
  return rx, y, rz

def project_3d_to_2d(x,y,z,width,height,fov=400):
    scale = fov/(z+fov)
    screenx = int(x * scale + (width/2))
    screeny = int(y * scale + (height/2))
    return screenx,screeny, scale

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((14, 16, 18))
    proj_inputs = []

    for x,y,z in input_nodes:
        rx, ry, rz = rotate_y(x, y, z, angle)
        proj_inputs.append(project_3d_to_2d(rx, ry, rz, WIDTH, HEIGHT))

    proj_outputs = []
    for x,y,z in output_nodes:
        rx, ry, rz = rotate_y(x, y, z, angle)
        proj_outputs.append(project_3d_to_2d(rx, ry, rz, WIDTH, HEIGHT))

    for p1_x, p1_y, _ in proj_inputs:
      for p2_x, p2_y, _ in proj_outputs:
        pygame.draw.line(screen, (70, 80, 90), (p1_x, p1_y), (p2_x, p2_y), 2)
    for px, py, scale in proj_inputs:
        pygame.draw.circle(screen, (0, 200, 255), (px, py), int(16 * scale))

    for px, py, scale in proj_outputs:
        pygame.draw.circle(screen, (0, 255, 128), (px, py), int(16 * scale))

    angle += 0.02
    pygame.display.flip()
    clock.tick(60)


pygame.quit()
sys.exit()