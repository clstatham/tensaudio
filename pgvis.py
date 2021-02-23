import torch
import numpy as np
import pygame
from pygame.locals import *
from global_constants import *

class PGVisualizer():
    def __init__(self):
        self.surface = None
        self.watched_vals = []
        self.val_history = []
        self.max_vals = 1+(TOTAL_PARAM_UPDATES*2)
        self.yscale = 0
        self.width = 0
        self.height = 0
        self.colors = []
        specratio = 255*6 / N_PARAMS
        step = int(specratio)
        red = 255
        green = 0
        blue = 0
        for i in range (0, 255*6+1, step):
            if i > 0 and i <= 255:
                blue += step
            elif i > 255 and i <= 255*2:
                red -= step
            elif i > 255*2 and i <= 255*3:
                green += step
            elif i > 255*3 and i <= 255*4:
                blue -= step
            elif i > 255*4 and i <= 255*5:
                red += step
            elif i > 255*5 and i <= 255*6:
                green -= step
            red = max(red, 0)
            red = min(red, 255)
            green = max(green, 0)
            green = min(green, 255)
            blue = max(blue, 0)
            blue = min(blue, 255)
            self.colors.append((red, green, blue))
    
    def to_pygame_coords(self, x, y):
        return x, self.height-y
    
    def init(self, width, height):
        pygame.init()
        self.width = width
        self.height = height
        self.yscale = height
        self.surface = pygame.display.set_mode((width, height))
    
    def reset(self):
        self.val_history = []

    def deinit(self):
        pygame.quit()

    def watch_val(self, val_lambda):
        self.watched_vals.append(val_lambda)

    def handle_events(self):
        for ev in pygame.event.get():
            if ev == QUIT:
                self.deinit()

    def update_vals(self):
        current_vals = []
        for getter in self.watched_vals:
            current_vals.append(next(getter))
        self.val_history.append(current_vals)
        if len(self.val_history) > self.max_vals:
            dif = len(self.val_history) - self.max_vals
            self.val_history = self.val_history[dif:self.max_vals]
            self.update_display()

    def draw_metrics(self, img, size):
        surf = pygame.image.fromstring(img, size, "RGB")
        self.surface.blit(surf, (0,0))
        pygame.display.flip()

    def update_display(self):
        self.surface.fill((50,50,50))
        
        if len(self.val_history) > 5:
            x = 0
            points = torch.Tensor([])
            for iteration in self.val_history:
                this_iter = []
                for val in iteration:
                    this_iter.append([x, (val*2)+100])
                points = torch.cat((points, torch.Tensor(this_iter)))
                x += self.width/len(self.val_history)
            points = points.permute(1, 0, 2)
            i = 0
            for line in points:
                pgline = []
                for point in line:
                    x = point[0].item()
                    y = point[1].item()
                    pgline.append(self.to_pygame_coords(x, y))
                pygame.draw.lines(self.surface, self.colors[i], False, pgline, 2)
                i += 1
            
        pygame.display.flip()

G_vis = None
def init_gvis(width, height):
    G_vis = PGVisualizer()
    G_vis.init(width, height)
    return G_vis
