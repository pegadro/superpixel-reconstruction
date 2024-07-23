from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QFileDialog
from PyQt5.QtCore import QTimer
from ui.ui_components import create_left_panel, create_central_panel, create_right_panel
from image_utils import load_and_scale_images, numpy_array_to_qpixmap
from superpixels import get_superpixels
from simulated_annealing import SimulatedAnnealing
from ui.simulated_annealing_wrapper import SimulatedAnnealingWrapper
import numpy as np
import cv2
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from skimage import io

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_variables()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Superpixel Reconstruction")
        self.setGeometry(100, 100, 1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.left_panel = create_left_panel(self)
        main_layout.addLayout(self.left_panel)

        self.central_panel = create_central_panel(self)
        main_layout.addLayout(self.central_panel, 2)

        self.right_panel = create_right_panel(self)
        main_layout.addLayout(self.right_panel)

    def init_variables(self):
        self.reconstruction_pixmap = None
        self.target_image_path = ""
        self.reconstruction_image_paths = []
        self.target_image_dimensions = None
        self.reconstruction_image_dimensions = None
        self.resize_factor = 3
        self.max_iterations = 10000
        self.initial_temperature = 10000
        self.segments_options = [
            [250, 500, 1000],
            [500, 1000, 1500],
            [1000, 1500, 2000],
            [1500, 2000, 2500]
        ]
        self.segments_list = self.segments_options[0]
        self.final_reconstruction = None
        self.sa_wrapper = None

    def change_target_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Target Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_name:
            self.target_image_path = file_name
            target_pixmap = QPixmap(file_name)
            scaled_pixmap = target_pixmap.scaled(
                self.target_image_container.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.target_image_label.setPixmap(scaled_pixmap)
            self.target_image_label.setAlignment(Qt.AlignCenter)

            self.target_image_dimensions = (
                target_pixmap.width(),
                target_pixmap.height(),
            )
            self.update_target_dimensions_label()
            self.update_reconstruction_dimensions()
            self.update_reconstruction_dimensions_label()

    def choose_reconstruction_images(self):
        file_names, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images for Reconstruction",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
        )

        self.reconstruction_image_paths.extend(file_names)

        for file_name in file_names:
            pixmap = QPixmap(file_name)
            self.image_carousel.add_image(pixmap)

    def clear_image_carousel(self):
        self.image_carousel.clear_images()
        self.reconstruction_image_paths.clear()

    def start_reconstruction(self):
        if self.target_image_path and self.reconstruction_image_paths:
            self.update_progress("Extracting superpixels")

            self.target, self.selected_images = self.load_and_preprocess_images()
            self.superpixels = []
            self.current_image_index = 0
            self.current_segment_index = 0

            QTimer.singleShot(0, self.process_next_superpixel)

    def process_next_superpixel(self):
        if self.current_image_index < len(self.selected_images):
            image = self.selected_images[self.current_image_index]

            if self.current_segment_index < len(self.segments_list):
                n_segments = self.segments_list[self.current_segment_index]
                im_superpixels, im_boundaries = get_superpixels(image, n_segments)

                im_pixmap_boundaries = numpy_array_to_qpixmap(im_boundaries)
                self.update_reconstruction_image(im_pixmap_boundaries)

                self.superpixels.extend(im_superpixels)

                self.current_segment_index += 1
                
                QTimer.singleShot(100, self.process_next_superpixel)
            else:
                self.current_image_index += 1
                self.current_segment_index = 0
                QTimer.singleShot(100, self.process_next_superpixel)
        else:
            self.update_progress("Reconstructing...")
            self.start_simulated_annealing()

    def start_simulated_annealing(self):
        sa_instance = SimulatedAnnealing(
            self.target,
            self.superpixels,
            self.segments_list,
            max_iterations=self.max_iterations,
            initial_temperature=self.initial_temperature,
        )

        self.sa_wrapper = SimulatedAnnealingWrapper(sa_instance)
        self.sa_wrapper.iteration_complete.connect(self.update_reconstruction_progress)
        self.sa_wrapper.reconstruction_complete.connect(self.finalize_reconstruction)

        QTimer.singleShot(0, self.sa_wrapper.run_iteration)

    def update_reconstruction_image(self, pixmap):
        self.reconstruction_pixmap = pixmap
        if pixmap and not pixmap.isNull():
            max_width = self.image_container.width() * 0.8
            max_height = self.image_container.height() * 0.8

            scaled_pixmap = pixmap.scaled(
                int(max_width),
                int(max_height),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.adjustSize()

    def finalize_reconstruction(self, final_reconstruction):
        self.update_progress("Reconstruction completed")
        self.final_reconstruction = final_reconstruction
        pixmap = numpy_array_to_qpixmap(final_reconstruction)

        self.update_reconstruction_image(pixmap)
        self.start_button.setEnabled(True)
        self.save_image_button.setEnabled(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.reconstruction_pixmap and self.reconstruction_pixmap:
            self.update_reconstruction_image(self.reconstruction_pixmap)

    def save_reconstructed_image(self):
        if hasattr(self, 'final_reconstruction') and isinstance(self.final_reconstruction, np.ndarray):
            file_name, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Save Reconstructed Image",
                "",
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)",
            )
            if file_name:
                io.imsave(file_name, self.final_reconstruction)
        else:
            print("No image to save")

    def update_progress(self, message):
        self.progress_label.setText(message)

    def update_iterations(self, value):
        self.iterations_label.setText(str(value))

    def update_superpixels(self, value):
        self.superpixels_label.setText(str(value))

    def update_similarity_score(self, value):
        self.similarity_score_label.setText(f"{value:.2f}")

    def update_reconstruction_progress(self, iterations, current_image, current_value, current_temperature, n_superpixels):
        self.iterations_label.setText(str(iterations))
        self.similarity_score_label.setText(f"{current_value:.2f}")
        self.current_temperature_label.setText(f"{current_temperature:.2f}")
        self.superpixels_label.setText(str(n_superpixels))

        pixmap = numpy_array_to_qpixmap(current_image)
        self.update_reconstruction_image(pixmap)

    def update_current_temperature(self, value):
        self.current_temperature_label.setText(f"{value:.2f}")

    def update_target_dimensions_label(self):
        if self.target_image_dimensions:
            width, height = self.target_image_dimensions
            self.target_dimensions_label.setText(f"{width}x{height}")
        else:
            self.target_dimensions_label.setText("N/A")

    def update_reconstruction_dimensions_label(self):
        if self.reconstruction_image_dimensions:
            width, height = self.reconstruction_image_dimensions
            self.reconstruction_dimensions_label.setText(f"{width}x{height}")
        else:
            self.reconstruction_dimensions_label.setText("N/A")

    def update_reconstruction_dimensions(self):
        if self.target_image_dimensions:
            self.reconstruction_image_dimensions = (
                self.target_image_dimensions[0] // self.resize_factor,
                self.target_image_dimensions[1] // self.resize_factor,
            )

    def on_resize_factor_changed(self, value):
        self.resize_factor = value
        self.update_reconstruction_dimensions()
        self.update_reconstruction_dimensions_label()

    def on_max_iterations_changed(self, value):
        self.max_iterations = value

    def on_initial_temperature_changed(self, value):
        self.initial_temperature = value

    def on_segments_changed(self):
        index = self.segments_button_group.checkedId()
        if index != -1:
            self.segments_list = self.segments_options[index]

    def load_and_preprocess_images(self):
        target, selected_images = load_and_scale_images(
            self.target_image_path, self.reconstruction_image_paths, self.resize_factor
        )
        return target, selected_images