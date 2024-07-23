from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QFormLayout,
    QSizePolicy,
    QSpinBox,
    QButtonGroup,
    QRadioButton,
    QScrollArea,
    QWidget,
)
from PyQt5.QtCore import Qt
from ui.image_carousel import ImageCarousel


def create_left_panel(main_window):
    left_panel = QVBoxLayout()
    left_panel.setSpacing(20)

    # Stats group
    stats_group = QGroupBox("Stats")
    stats_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    stats_layout = QFormLayout(stats_group)
    stats_layout.setLabelAlignment(Qt.AlignLeft)
    stats_layout.setFormAlignment(Qt.AlignLeft)

    main_window.iterations_label = QLabel("0")
    main_window.superpixels_label = QLabel("0")
    main_window.similarity_score_label = QLabel("0.0")
    main_window.current_temperature_label = QLabel("0.0")
    main_window.target_dimensions_label = QLabel("N/A")
    main_window.reconstruction_dimensions_label = QLabel("N/A")

    stats_layout.addRow("Iterations:", main_window.iterations_label)
    stats_layout.addRow("Superpixels:", main_window.superpixels_label)
    stats_layout.addRow("Current value:", main_window.similarity_score_label)
    stats_layout.addRow("Current temperature:", main_window.current_temperature_label)
    stats_layout.addRow("Target dimensions:", main_window.target_dimensions_label)
    stats_layout.addRow(
        "Reconstruction dimensions:", main_window.reconstruction_dimensions_label
    )

    stats_group.setMinimumWidth(290)

    left_panel.addWidget(stats_group)

    image_group = QGroupBox("Image Loading")
    image_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    image_layout = QVBoxLayout(image_group)
    image_layout.setSpacing(10)

    target_image_label = QLabel("Target Image:")
    image_layout.addWidget(target_image_label)

    main_window.target_image_container = QWidget()
    main_window.target_image_container.setFixedSize(200, 200)
    main_window.target_image_container.setStyleSheet("background-color: #2a2a2a;")

    target_image_layout = QVBoxLayout(main_window.target_image_container)
    target_image_layout.setAlignment(Qt.AlignCenter)

    main_window.target_image_label = QLabel()
    main_window.target_image_label.setAlignment(Qt.AlignCenter)
    target_image_layout.addWidget(main_window.target_image_label)

    image_layout.addWidget(main_window.target_image_label)

    change_target_btn = QPushButton("Change Target Image")
    change_target_btn.clicked.connect(main_window.change_target_image)
    image_layout.addWidget(change_target_btn)

    carousel_label = QLabel("Images for Reconstruction:")
    image_layout.addWidget(carousel_label)

    main_window.image_carousel = ImageCarousel()
    image_layout.addWidget(main_window.image_carousel)

    choose_images_btn = QPushButton("Choose Images for Reconstruction")
    choose_images_btn.clicked.connect(main_window.choose_reconstruction_images)
    image_layout.addWidget(choose_images_btn)

    clear_carousel_btn = QPushButton("Clear Image Selection")
    clear_carousel_btn.clicked.connect(main_window.clear_image_carousel)
    image_layout.addWidget(clear_carousel_btn)

    left_panel.addWidget(image_group)

    left_panel.addStretch(1)

    return left_panel


def create_central_panel(main_window):
    central_panel = QVBoxLayout()
    central_panel.setSpacing(10)

    main_window.image_container = QWidget()
    main_window.image_container.setStyleSheet("background-color: #2a2a2a;")
    main_window.image_container_layout = QVBoxLayout(main_window.image_container)
    main_window.image_container_layout.setAlignment(Qt.AlignCenter)

    main_window.image_label = QLabel()
    main_window.image_label.setAlignment(Qt.AlignCenter)
    main_window.image_label.setStyleSheet("background-color: transparent;")
    main_window.image_container_layout.addWidget(main_window.image_label)

    scroll_area = QScrollArea()
    scroll_area.setWidget(main_window.image_container)
    scroll_area.setWidgetResizable(True)
    scroll_area.setAlignment(Qt.AlignCenter)

    central_panel.addWidget(scroll_area, 1)

    main_window.progress_label = QLabel("")
    main_window.progress_label.setAlignment(Qt.AlignCenter)
    main_window.progress_label.setStyleSheet("font-size: 14pt;")
    central_panel.addWidget(main_window.progress_label)

    return central_panel


def create_right_panel(main_window):
    right_panel = QVBoxLayout()
    right_panel.setAlignment(Qt.AlignTop)
    right_panel.setSpacing(20)

    parameters_group = QGroupBox("Parameters")
    parameters_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    parameters_layout = QFormLayout(parameters_group)
    parameters_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

    main_window.resize_factor_spinbox = QSpinBox()
    main_window.resize_factor_spinbox.setRange(1, 20)
    main_window.resize_factor_spinbox.setValue(main_window.resize_factor)
    main_window.resize_factor_spinbox.valueChanged.connect(main_window.on_resize_factor_changed)
    parameters_layout.addRow("Resize factor:", main_window.resize_factor_spinbox)

    main_window.max_iterations_spinbox = QSpinBox()
    main_window.max_iterations_spinbox.setRange(1, 100000)
    main_window.max_iterations_spinbox.setValue(main_window.max_iterations)
    main_window.max_iterations_spinbox.valueChanged.connect(main_window.on_max_iterations_changed)
    parameters_layout.addRow("Max iterations:", main_window.max_iterations_spinbox)

    main_window.initial_temperature_spinbox = QSpinBox()
    main_window.initial_temperature_spinbox.setRange(1, 100000)
    main_window.initial_temperature_spinbox.setValue(main_window.initial_temperature)
    main_window.initial_temperature_spinbox.valueChanged.connect(
        main_window.on_initial_temperature_changed
    )
    parameters_layout.addRow("Initial temperature:", main_window.initial_temperature_spinbox)

    segments_group = QGroupBox("Segments List")
    segments_layout = QVBoxLayout()
    main_window.segments_button_group = QButtonGroup(main_window)

    for i, segments in enumerate(main_window.segments_options):
        radio_button = QRadioButton(f"{segments}")
        main_window.segments_button_group.addButton(radio_button, i)
        segments_layout.addWidget(radio_button)

    main_window.segments_button_group.buttons()[0].setChecked(True)
    main_window.segments_button_group.buttonClicked.connect(main_window.on_segments_changed)
    segments_group.setLayout(segments_layout)

    parameters_layout.addRow(segments_group)

    right_panel.addWidget(parameters_group)

    running_saving_group = QGroupBox("Running and Saving Options")
    running_saving_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    running_saving_layout = QVBoxLayout(running_saving_group)

    main_window.start_button = QPushButton("Start")
    main_window.start_button.clicked.connect(main_window.start_reconstruction)

    running_saving_layout.addWidget(main_window.start_button)

    main_window.save_image_button = QPushButton("Save Reconstructed Image")
    main_window.save_image_button.clicked.connect(main_window.save_reconstructed_image)
    main_window.save_image_button.setEnabled(False)
    running_saving_layout.addWidget(main_window.save_image_button)

    right_panel.addWidget(running_saving_group)

    right_panel.addStretch(1)

    return right_panel
