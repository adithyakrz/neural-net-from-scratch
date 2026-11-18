import pygame
import sys
import math
import numpy as np
from network import activation, backward_prop, forward_sum, mse_loss, updateweight

pygame.init()
font = pygame.font.SysFont("consolas", 18)

X_train = np.array([[1, 2, 3], [0.1, 0.2, 0.3]])
Y_train = np.array([[1, 0], [0, 1]])
w = np.array([[0.2, 0.41], [0.11, 0.02], [0.001, 0.12]])
b = np.array([4.0, 4.0])

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

epoch = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((14, 16, 18))
    
    proj_inputs = []

    z = forward_sum(X_train, w, b)
    A = activation(z)
    loss = mse_loss(A, Y_train)
    dw, db = backward_prop(X_train, A, Y_train)
    w, b = updateweight(w, b, dw, db, learning_rate=0.1)
    epoch += 1

    for x,y,z in input_nodes:
        rx, ry, rz = rotate_y(x, y, z, angle)
        proj_inputs.append(project_3d_to_2d(rx, ry, rz, WIDTH, HEIGHT))

    proj_outputs = []
    for x,y,z in output_nodes:
        rx, ry, rz = rotate_y(x, y, z, angle)
        proj_outputs.append(project_3d_to_2d(rx, ry, rz, WIDTH, HEIGHT))

    for i, (p1_x, p1_y, _) in enumerate(proj_inputs):
      for j, (p2_x, p2_y, _) in enumerate(proj_outputs):
        weight_val = w[i][j]
        line_color = (0, 220, 120) if weight_val >= 0 else (255, 70, 85)
        thickness = max(1, min(8, int(abs(weight_val) * 3)))
        pygame.draw.line(
            screen, line_color, (p1_x, p1_y), (p2_x, p2_y), thickness
        )
    for px, py, scale in proj_inputs:
        pygame.draw.circle(screen, (0, 200, 255), (px, py), int(16 * scale))

    for j, (px, py, scale) in enumerate(proj_outputs):
      act_val = float(A[0][j])
      glow = int(40 + 215 * act_val)
      node_color = (0, glow, int(glow * 0.6))
      radius = int((12 + 8 * act_val) * scale)
      pygame.draw.circle(screen, node_color, (px, py), radius)

    angle += 0.01
    hud = font.render(
    f"Epoch: {epoch}  |  Loss: {loss:.5f}", True, (220, 225, 230))
    screen.blit(hud, (20, 20))
    pygame.display.flip()
    
    clock.tick(60)


pygame.quit()
sys.exit()