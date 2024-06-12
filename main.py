from image_utils import load_and_scale_images
from superpixels import get_superpixels
from simulated_annealing import SimulatedAnnealing
from metrics import calculate_mse, calculate_ssim
from skimage import io


def run():
    target_image_path = "images_test/beatles.jpg"
    selected_images_path = ["images_test/desperate_man.jpeg", "images_test/pearl.jpeg"]
    output_reconstruction_path = "reconstruction/beatles_reconstruction.png"

    target, selected_images = load_and_scale_images(
        target_image_path, selected_images_path, resize_factor=10
    )

    segments_list = [500, 2500, 5000, 10000]
    print("obtenido superpixels")
    superpixels = get_superpixels(selected_images, segments_list)
    simulated_annealing = SimulatedAnnealing(
        target,
        superpixels,
        segments_list,
        max_iterations=15000,
        initial_temperature=15000,
        objective_function=calculate_mse,
    )
    print("corriendo recocido simulado")
    initial = simulated_annealing.get_initial_state()
    best_solution = simulated_annealing.simulated_annealing(initial)

    io.imsave(output_reconstruction_path, best_solution)


if __name__ == "__main__":
    run()
