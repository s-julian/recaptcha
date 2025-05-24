from abc import ABC, abstractmethod

class AutomatorInterface(ABC):
    @abstractmethod
    def launch(self): pass

    @abstractmethod
    def find_captcha_box(self): pass

    @abstractmethod
    def click_box(self): pass
    
    @abstractmethod
    def move_mouse_to(self, x: int, y: int): pass

    @abstractmethod
    def move_to_tile(self, row: int, col: int): pass

    @abstractmethod
    def follow_path(self, path: list): pass

    @abstractmethod
    def refresh(self): pass

    @abstractmethod
    def close(self): pass