from abc import ABC, abstractmethod
from typing import List


class TrainingPipeline(ABC):
    
    @abstractmethod
    def train(self) -> None:
        pass
