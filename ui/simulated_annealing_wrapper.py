from PyQt5.QtCore import QObject, pyqtSignal, QTimer

class SimulatedAnnealingWrapper(QObject):
    iteration_complete = pyqtSignal(int, object, float, float, int)
    reconstruction_complete = pyqtSignal(object)

    def __init__(self, simulated_annealing_instance):
        super().__init__()
        self.sa = simulated_annealing_instance
        self.generator = self.sa.simulated_annealing()

    def run_iteration(self):
        # Run a single iteration of the simulated annealing algorithm
        try:
            iterations, current, current_value, current_temperature, n_superpixels = next(self.generator)
            self.iteration_complete.emit(iterations, current, current_value, current_temperature, n_superpixels)
            QTimer.singleShot(0, self.run_iteration)
        except StopIteration as e:
            self.reconstruction_complete.emit(e.value)