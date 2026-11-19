import pygame
import sys
import math
import numpy as np
from network import forward_mlp, backward_mlp, mse_loss, updateweight

pygame.init()
font = pygame.font.SysFont("consolas", 18)
small_font = pygame.font.SysFont("consolas", 15)

X_train = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float)
Y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=float)

def init_weights():
  w1 = np.random.uniform(-1.0, 1.0, (3, 4))
  b1 = np.zeros(4)
  w2 = np.random.uniform(-1.0, 1.0, (4, 2))
  b2 = np.zeros(2)
  return w1, b1, w2, b2

np.random.seed(42)
w1, b1, w2, b2 = init_weights() 

WIDTH, HEIGHT = 900, 700

screen = pygame.display.set_mode((WIDTH,HEIGHT))

pygame.display.set_caption("Neural Network 3D Visualizer")

clock = pygame.time.Clock()
angle = 0.0
running = True
epoch = 0
training_active = True
sample_idx = 1
input_nodes = [(-180, -100, 80), (-180, 0, 80), (-180, 100, 80)]
hidden_nodes = [(0, -135, 0), (0, -45, 0), (0, 45, 0), (0, 135, 0)]
output_nodes = [(180, -60, -80), (180, 60, -80)]

def rotate_y(x, y, z, angle):
  rx = x * math.cos(angle) - z * math.sin(angle)
  rz = x * math.sin(angle) + z * math.cos(angle)
  return rx, y, rz

def project_3d_to_2d(x,y,z,width,height,fov=400):
    scale = fov/(z+fov)
    screenx = int(x * scale + (width/2))
    screeny = int(y * scale + (height/2))
    return screenx,screeny, scale

def draw_synapses(layer_a, layer_b, weights):
  for i, (p1_x, p1_y, _) in enumerate(layer_a):
    for j, (p2_x, p2_y, _) in enumerate(layer_b):
      weight_val = weights[i][j]
      line_color = (0, 220, 120) if weight_val >= 0 else (255, 70, 85)
      thickness = max(1, min(8, int(abs(weight_val) * 3)))
      pygame.draw.line(
          screen, line_color, (p1_x, p1_y), (p2_x, p2_y), thickness
      )

a1, a2 = forward_mlp(X_train, w1, b1, w2, b2)
loss = mse_loss(a2, Y_train)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                training_active = not training_active
            elif event.key == pygame.K_r:
                w1, b1, w2, b2 = init_weights()
                epoch = 0
                a1, a2 = forward_mlp(X_train, w1, b1, w2, b2)
                loss = mse_loss(a2, Y_train)
            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                sample_idx = event.key - pygame.K_1

    mouse_dx, _ = pygame.mouse.get_rel()
    if pygame.mouse.get_pressed()[0]:
        angle += mouse_dx * 0.01
    else:
        angle += 0.008

    if training_active:
        for _ in range(5):
            a1, a2 = forward_mlp(X_train, w1, b1, w2, b2)
            loss = mse_loss(a2, Y_train)
            dw1, db1, dw2, db2 = backward_mlp(X_train, a1, a2, Y_train, w2)
            w1, b1 = updateweight(w1, b1, dw1, db1, learning_rate=0.5)
            w2, b2 = updateweight(w2, b2, dw2, db2, learning_rate=0.5)
            epoch += 1
    screen.fill((14, 16, 18))

    proj_inputs = [
        project_3d_to_2d(*rotate_y(x, y, z, angle), WIDTH, HEIGHT)
        for x, y, z in input_nodes
    ]
    proj_hidden = [
        project_3d_to_2d(*rotate_y(x, y, z, angle), WIDTH, HEIGHT)
        for x, y, z in hidden_nodes
    ]
    proj_outputs = [
        project_3d_to_2d(*rotate_y(x, y, z, angle), WIDTH, HEIGHT)
        for x, y, z in output_nodes
    ]
    draw_synapses(proj_inputs, proj_hidden, w1)
    draw_synapses(proj_hidden, proj_outputs, w2)
    
    for i, (px, py, scale) in enumerate(proj_inputs):
        in_val = float(X_train[sample_idx][i])
        color = (0, 210, 255) if in_val > 0.5 else (35, 65, 80)
        radius = int((16 if in_val > 0.5 else 11) * scale)
        pygame.draw.circle(screen, color, (px, py), radius)

    for h_idx, (px, py, scale) in enumerate(proj_hidden):
        act_val = float(a1[sample_idx][h_idx])
        glow = int(50 + 205 * act_val)
        pygame.draw.circle(
            screen, (glow, 90, 255), (px, py), int((12 + 7 * act_val) * scale)
        )

    for j, (px, py, scale) in enumerate(proj_outputs):
        act_val = float(a2[sample_idx][j])
        glow = int(40 + 215 * act_val)
        node_color = (0, glow, int(glow * 0.6))
        radius = int((12 + 8 * act_val) * scale)
        pygame.draw.circle(screen, node_color, (px, py), radius)

    in_bits = f"[{int(X_train[sample_idx][0])}, {int(X_train[sample_idx][1])}]"
    pred_class = int(np.argmax(a2[sample_idx]))
    status = "TRAINING" if training_active else "PAUSED"
    hud = font.render(
        f"Epoch: {epoch} | Loss: {loss:.5f} | Input {in_bits} -> Pred: {pred_class} [{status}]",
        True,
        (220, 225, 230),
    )
    controls = small_font.render(
        "Controls: [1-4] Switch XOR Input  |  [Space] Pause/Resume  |  [R] Reset  |  [Mouse Drag] Rotate",
        True,
        (130, 140, 150),
    )
    screen.blit(hud, (20, 20))
    screen.blit(controls, (20, HEIGHT - 32))

    pygame.display.flip()
    clock.tick(60)


pygame.quit()
sys.exit()