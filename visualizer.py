import pygame
import sys

pygame.init()

WIDTH, HEIGHT = 900, 700

screen = pygame.display.set_mode((WIDTH,HEIGHT))

pygame.display.set_caption("Neural Network 3D Visualizer")

clock = pygame.time.Clock()

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        screen.fill((14, 16, 18))
        pygame.display.flip()
        clock.tick(60)


pygame.quit()
sys.exit()