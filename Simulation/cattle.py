import pygame
import numpy as np
import random
from definitions import *

class Cattle:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.random.uniform(-1, 1, 2)
        self.acc = np.zeros(2)
        self.push_radius = 50.0  # (Unused here, but available for fine-tuning.)
        # Stress-related attributes.
        self.stressed = False
        self.stress_timer = 0

    def apply_force(self, force):
        self.acc += force

    def boid_behavior(self, cattle_group):
        separation = self.separation(cattle_group)
        alignment = self.alignment(cattle_group)
        cohesion = self.cohesion(cattle_group)
        self.apply_force(SEPARATION_WEIGHT * separation)
        self.apply_force(ALIGNMENT_WEIGHT * alignment)
        self.apply_force(COHESION_WEIGHT * cohesion)

    def separation(self, cattle_group):
        steer = np.zeros(2)
        count = 0
        for other in cattle_group:
            distance = np.linalg.norm(self.pos - other.pos)
            if 0 < distance < SEPARATION_RADIUS:
                diff = self.pos - other.pos
                diff /= distance
                steer += diff
                count += 1
        if count > 0:
            steer /= count
        return self.limit(steer, MAX_FORCE)

    def alignment(self, cattle_group):
        avg_vel = np.zeros(2)
        count = 0
        for other in cattle_group:
            distance = np.linalg.norm(self.pos - other.pos)
            if 0 < distance < ALIGNMENT_RADIUS:
                avg_vel += other.vel
                count += 1
        if count > 0:
            avg_vel /= count
            avg_vel = self.limit(avg_vel, MAX_SPEED)
        steer = avg_vel - self.vel
        return self.limit(steer, MAX_FORCE)

    def cohesion(self, cattle_group):
        center_of_mass = np.zeros(2)
        count = 0
        for other in cattle_group:
            distance = np.linalg.norm(self.pos - other.pos)
            if 0 < distance < COHESION_RADIUS:
                center_of_mass += other.pos
                count += 1
        if count > 0:
            center_of_mass /= count
            desired = center_of_mass - self.pos
            desired = self.limit(desired, MAX_SPEED)
            steer = desired - self.vel
            return self.limit(steer, MAX_FORCE)
        return np.zeros(2)

    def limit(self, vector, max_value):
        magnitude = np.linalg.norm(vector)
        if magnitude > max_value:
            return vector / magnitude * max_value
        return vector
    

    def move(self, wrangler, cattle_group, clusters, fence_x=1000, fence_y=900):
        # Apply flocking behavior.
        self.boid_behavior([c for c in cattle_group if c != self])
        
        # Count down stress if already stressed.
        if self.stressed:
            self.stress_timer -= 1
            if self.stress_timer <= 0:
                self.stressed = False

        # Only apply drone pressure if the wrangler is not "flying high" (i.e. in light pink).
        if wrangler.color != COLOR_FLY_HIGH:
            distance_to_wrangler = np.linalg.norm(self.pos - wrangler.pos)
            if distance_to_wrangler < PRESSURE_PUSH_RADIUS:
                # Chance to trigger stress if not already stressed.
                if not self.stressed and random.random() < STRESS_PROBABILITY and distance_to_wrangler < PRESSURE_PUSH_RADIUS / 1.25:
                    self.stressed = True
                    self.stress_timer = random.randint(*STRESS_DURATION_RANGE)
                # Only apply pressure if not stressed.
                if not self.stressed:
                    direction = self.pos - wrangler.pos
                    if np.linalg.norm(direction) > 0:
                        direction /= np.linalg.norm(direction)
                    pressure_strength = (PRESSURE_PUSH_RADIUS - distance_to_wrangler) / PRESSURE_PUSH_RADIUS
                    random_factor = random.uniform(0.1, 0.3)
                    self.apply_force(direction * pressure_strength * random_factor)
                    # Release pressure if moving fast enough.
                    if np.linalg.norm(self.vel) > CATTLE_SPEED_THRESHOLD:
                        self.vel *= 0.95
        
        # Muster toward the nearest cluster if available.
        if clusters:
            closest_cluster = min(clusters, key=lambda cluster: np.linalg.norm(self.pos - cluster))
            direction_to_cluster = closest_cluster - self.pos
            self.apply_force(self.limit(direction_to_cluster, MAX_FORCE))

        # Fence avoidance.
        if self.pos[1] < fence_y:
            if fence_x - 20 < self.pos[0] < fence_x + 20:
                if self.pos[0] < fence_x:
                    self.apply_force(np.array([-1, 0]) * 50 * MAX_FORCE)
                else:
                    self.apply_force(np.array([1, 0]) * 50 * MAX_FORCE)

        # Add a bit of randomness.
        self.acc += np.random.uniform(-0.05, 0.05, 2)

        # Update velocity and position.
        self.vel += self.acc
        self.vel = self.limit(self.vel, MAX_SPEED)
        self.pos += self.vel
        self.acc *= 0

        # Stay within the screen.
        self.pos[0] = np.clip(self.pos[0], 0, SCREEN_WIDTH)
        self.pos[1] = np.clip(self.pos[1], 0, SCREEN_HEIGHT)

    def draw(self, screen):
        color = COLOR_STRESSED if self.stressed else COLOR_NORMAL
        pygame.draw.circle(screen, color, self.pos.astype(int), 5)
