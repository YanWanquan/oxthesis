# --- --- ---
# data_loader_interface.py
# Sven Giegerich / 03.05.2021
# --- --- ---

from enum import IntEnum
from abc import ABC, abstractmethod

class DataTypes(IntEnum):

    TRAIN = 1
    VAL = 2
    TEST = 3

    @staticmethod
    def get_string_name():
        return {
            DataTypes.TRAIN: "train",
            DataTypes.VAL: "validation",
            DataTypes.TEST: "test"
        }


class DataLoaderInterface(ABC):

    @abstractmethod
    def lorem(self):
        pass
