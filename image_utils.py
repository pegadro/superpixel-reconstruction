from PIL import Image
import numpy as np
from PyQt5.QtGui import QImage, QPixmap

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
        selected_images.append(image)

    return target, selected_images


def numpy_array_to_qpixmap(image):
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # OpenCV usa BGR, Qt usa RGB, así que necesitamos cambiar el orden de los canales
    # if channel == 3:
    #     image = image[:, :, ::-1]
    
    # Crear QImage desde el numpy array
    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    # Convertir QImage a QPixmap
    pixmap = QPixmap.fromImage(q_image)
    
    return pixmap


def get_mean_image_color(image):
    return np.mean(image, axis=(0, 1)).astype(np.uint8)
