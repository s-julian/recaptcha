from abc import ABC, abstractmethod

class AutomatorInterface(ABC):
    @abstractmethod
    def launch(self, url:str): pass

    @abstractmethod
    def go_to_url(self, url:str): pass

    @abstractmethod
    def click_at(self): pass

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
    def close_browser(self): pass

    @abstractmethod
    def copy_token(self) -> str: pass

    @abstractmethod
    def challenge_triggered(self) -> bool: pass