import pygame
import numpy as np
import math
from definitions import *
class Wrangler:
    def __init__(self):
        self.pos = np.array([400, 50], dtype=float)
        self.speed = 4
        self.color = (255, 0, 0)  # Default red.
        self.line_color = (0, 255, 0)  # Guiding line color.
        self.line_length = 100
        self.fly_over = False
        self.oscillation_amplitude = 20    # Maximum horizontal offset (in pixels)
        self.oscillation_frequency = 0.005  # Frequency factor (adjust as needed)

    def update_position(self, cattle_group, target_pos, farthest_cattle, active_centroid, screen):
        

        close_to_cow = any(np.linalg.norm(self.pos - c.pos) < 50 for c in cattle_group)
        close_to_cluster = np.linalg.norm(self.pos - active_centroid) < 100

        if close_to_cow or close_to_cluster:
            self.color = COLOR_FLY_HIGH  # Light pink (indicating proximity).
            self.fly_over = True

        else:
            self.color = (255, 0, 0)  # Default red.
            self.fly_over = False


        # If any cow is stressed, back away (fly high) and wait.
        if any(c.stressed for c in cattle_group) and self.fly_over == False:
            self.color = COLOR_FLY_HIGH  # Light pink.
            # Back away by moving upward.
            self.pos[1] -= self.speed/2
            return

        # Otherwise, use normal guidance.
        if not cattle_group:
            return
        
        

        herd_velocity = np.mean([c.vel for c in cattle_group], axis=0)
        herd_speed = np.linalg.norm(herd_velocity)
        herd_direction = herd_velocity / herd_speed if herd_speed > 0 else np.zeros(2)
        goal_direction = target_pos - self.pos
        if np.linalg.norm(goal_direction) > 0:
            goal_direction /= np.linalg.norm(goal_direction)

        # If the targeted cow (farthest) is moving fast enough in the right direction,
        # or the herd overall is, pause movement.
        target_speed = np.linalg.norm(farthest_cattle.vel)
        if target_speed > WRANGLER_RELEASE_THRESHOLD and np.dot(farthest_cattle.vel, goal_direction) > 0.8:
            self.color = COLOR_FLY_HIGH  # Drone flies high.
            return
        if herd_speed > 0.5 and np.dot(herd_direction, goal_direction) > 0.8:
            self.color = COLOR_FLY_HIGH
            return

        # Calculate the approach direction based on target and active centroid.
        direction_to_target = target_pos - farthest_cattle.pos
        direction_to_centroid = active_centroid - farthest_cattle.pos
        if np.linalg.norm(direction_to_target) > 0:
            direction_to_target /= np.linalg.norm(direction_to_target)
        if np.linalg.norm(direction_to_centroid) > 0:
            direction_to_centroid /= np.linalg.norm(direction_to_centroid)
        approach_direction = direction_to_target + direction_to_centroid
        if np.linalg.norm(approach_direction) > 0:
            approach_direction /= np.linalg.norm(approach_direction)

        # Set the desired position (base) for the wrangler.
        base_desired_position = farthest_cattle.pos - approach_direction * 80

        # Now add an oscillatory horizontal offset.
        # We'll use the current pygame time (in milliseconds) to compute the sine oscillation.
        time_ms = pygame.time.get_ticks()
        oscillation_offset = self.oscillation_amplitude * math.sin(time_ms * self.oscillation_frequency)
        
        # Assuming that "horizontal" corresponds to the x-direction,
        # we add the offset to the desired x-position.
        desired_position = base_desired_position.copy()
        desired_position[0] += oscillation_offset

        # Compute the vector toward the desired position and move the wrangler.
        direction_to_desired = desired_position - self.pos
        distance_to_desired = np.linalg.norm(direction_to_desired)
        if distance_to_desired > 0:
            move_vector = direction_to_desired / distance_to_desired * min(self.speed, distance_to_desired)
            self.pos += move_vector

        # Draw a guiding line (optional)
        line_end = farthest_cattle.pos + approach_direction * self.line_length
        self.draw_guiding_line(farthest_cattle.pos, line_end, screen)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.pos.astype(int), 10)

    def draw_guiding_line(self, start_pos, end_pos, screen):
        pygame.draw.line(screen, self.line_color, start_pos.astype(int), end_pos.astype(int), 2)