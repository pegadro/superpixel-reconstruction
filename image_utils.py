from PIL import Image
import numpy as np


def load_and_scale_images(target_image_path, selected_images_paths, resize_factor=2):
    target = Image.open(target_image_path)
    target = target.resize(
        (target.size[0] // resize_factor, target.size[1] // resize_factor)
    )
    target = np.array(target)

    selected_images = []
    for path in selected_images_paths:
        image = Image.open(path)
        width = target.shape[1]

        factor = (
            image.size[0] // width if image.size[0] >= width else width // image.size[0]
        )

        image = image.resize(
            (
                width,
                (
                    image.size[1] // factor
                    if image.size[0] >= width
                    else image.size[1] * factor
                ),
            )
        )

        image = np.array(image)
        selected_images.append(selected_images)

    return target, selected_images


def get_mean_image_color(image):
    return np.mean(image, axis=(0, 1)).astype(np.uint8)
