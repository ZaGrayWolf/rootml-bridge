from abc import ABC, abstractmethod


class BaseTrainer(ABC):

    @abstractmethod
    def train(self, data_path, config, out_dir):
        pass
