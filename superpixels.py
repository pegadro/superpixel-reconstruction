import numpy as np
from PIL import Image
from skimage.segmentation import slic


class Superpixel:
    def __init__(self, superpixel, mask, mean_color=None, size_category=None):
        self.superpixel = superpixel
        self.mask = mask
        self.mean_color = mean_color
        self.size_category = size_category


def get_superpixels(images_paths, n_segments_list, target):
    superpixels = []

    for path in images_paths:
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

        for n_segments in n_segments_list:
            segments = slic(
                image, n_segments=n_segments, compactness=50, sigma=1, start_label=1
            )
            unique_segments = np.unique(segments)

            for i, segment_label in enumerate(unique_segments):
                mask = segments == segment_label
                coords = np.argwhere(mask)

                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0) + 1

                superpixel = image[y0:y1, x0:x1]
                superpixel_mask = mask[y0:y1, x0:x1]

                pixels = superpixel[superpixel_mask]
                mean_color = np.mean(pixels, axis=0).astype(np.int_)

                superpixels.append(
                    Superpixel(
                        superpixel,
                        superpixel_mask,
                        mean_color=mean_color,
                        size_category=n_segments,
                    )
                )

    return superpixels
