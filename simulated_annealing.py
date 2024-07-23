import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from metrics import calculate_mse, calculate_ssim
from image_utils import get_mean_image_color


class SimulatedAnnealing:
    def __init__(
        self,
        target,
        superpixels,
        superpixels_sizes,
        max_iterations=3000,
        initial_temperature=3000,
        objective_function=calculate_mse,
    ):
        self.target = target
        self.superpixels = superpixels
        self.superpixels_sizes = superpixels_sizes
        self.max_iterations = max_iterations
        self.iterations_per_size, self.superpixels_groups = (
            self.calculate_iterations_per_size(
                self.superpixels_sizes, self.max_iterations
            )
        )

        self.initial_temperature = initial_temperature
        self.cooling_factor = 0.9999
        self.objective_function = objective_function
        self.initial = self.get_initial_state()

    def generate_random_successor(self, current, iteration):
        position = [
            np.random.randint(0, self.target.shape[0]),
            np.random.randint(0, self.target.shape[1]),
        ]
        color_in_target = self.target[position[0], position[1]]

        superpixels_group = self.calculate_current_superpixels_group(iteration)

        mean_colors_array = [superpixel.mean_color for superpixel in superpixels_group]
        distances = np.sqrt(np.sum((color_in_target - mean_colors_array) ** 2, axis=1))
        closest_color = np.argmin(distances)

        random_superpixel = superpixels_group[closest_color]
        successor = self.put_superpixel_in_image(current, random_superpixel, position)

        return successor

    def temperature_change(self, current_temperature):
        return current_temperature * self.cooling_factor

    def simulated_annealing(self):
        current = self.initial
        current_value = self.objective_function(self.target, current)
        current_temperature = self.initial_temperature

        n_superpixels = 0
        iterations = 0
        while iterations < self.max_iterations:
            current_temperature = self.temperature_change(current_temperature)

            successor = self.generate_random_successor(current, iterations)
            successor_value = self.objective_function(self.target, current)
            delta_value = successor_value - current_value

            if delta_value < 0:
                current = successor
                current_value = successor_value
                n_superpixels += 1
            else:
                acceptance_probability = np.exp(-delta_value / current_temperature)
                if np.random.random() < acceptance_probability:
                    current = successor
                    current_value = successor_value
                    n_superpixels += 1

            iterations += 1
            yield iterations, current, current_value, current_temperature, n_superpixels

        return current

    def put_superpixel_in_image(self, image, superpixel, position, alpha=0.5):
        height, width, _ = image.shape
        new_image = np.copy(image)

        filter_mask = np.argwhere(superpixel.mask == 1)
        filter_mask_image = np.copy(filter_mask)

        filter_mask_image[:, 0] += position[0]
        filter_mask_image[:, 1] += position[1]

        mask = (filter_mask_image[:, 0] < height) & (filter_mask_image[:, 1] < width)

        filter_mask = filter_mask[mask]
        filter_mask_image = filter_mask_image[mask]

        blended_superpixel = (1 - alpha) * new_image[
            filter_mask_image[:, 0], filter_mask_image[:, 1]
        ] + (alpha) * superpixel.mean_color # (superpixel.superpixel[filter_mask[:, 0], filter_mask[:, 1]])

        new_image[filter_mask_image[:, 0], filter_mask_image[:, 1]] = blended_superpixel

        return new_image

    def get_initial_state(self):
        mean_color = get_mean_image_color(self.target)
        return np.full(self.target.shape, mean_color)

    def calculate_iterations_per_size(self, superpixels_sizes, max_iterations):
        weights = list(range(1, len(superpixels_sizes) + 1))
        total_weight = sum(weights)
        iterations_per_size = []

        for i in range(len(superpixels_sizes)):
            iterations_per_size.append(
                int((weights[i] / total_weight) * max_iterations)
            )

        superpixels_groups = []
        for size in superpixels_sizes:
            group = [
                superpixel
                for superpixel in self.superpixels
                if superpixel.size_category == size
            ]
            superpixels_groups.append(group)

        return iterations_per_size, superpixels_groups

    def calculate_current_superpixels_group(self, iteration):
        index = self.find_group_index(iteration)
        if index != None:
            return self.superpixels_groups[index]
        return self.superpixels_groups[-1]

    def find_group_index(self, iteration):
        cumulative_sum = 0
        for index, iter_count in enumerate(self.iterations_per_size):
            cumulative_sum += iter_count
            if iteration < cumulative_sum:
                return index
        return None
