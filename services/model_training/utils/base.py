"""
This module defines the `TrainingPipeline` abstract base class (ABC) for training pipelines.

Any subclass of `TrainingPipeline` must implement the `train` method, which defines the 
training process for a specific machine learning model.
"""

from abc import ABC, abstractmethod


class TrainingPipeline(ABC):
    """
    Abstract base class for training pipelines.

    Subclasses must implement the `train` method, which encapsulates the model training process.
    """

    @abstractmethod
    def train(self) -> None:
        """
        Trains the model. 

        This method must be implemented by subclasses to define the specific training logic.
        """
        pass
