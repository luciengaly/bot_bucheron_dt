import time
import random

import pyautogui
import cv2
import mss

from constant import *
from matcher import TemplateMatcher


class ScreenController:
    def __init__(self, matcher: TemplateMatcher, path: list, current_pos: tuple, direction: int = 1) -> None:
        self.path = path
        self.matcher = matcher
        self.current_pos = current_pos
        self.direction = direction
        
    def move(self, dir: str) -> None:
        pyautogui.press(dir)
        self.current_pos = self.next_pos
    
    def click(self, coord: tuple) -> None: 
        pyautogui.click(coord[0], coord[1])
        
    def compute_dir(self, current_pos: tuple, next_pos: tuple) -> str:
        if next_pos[0] > current_pos[0]:
            return "right"
        if next_pos[0] < current_pos[0]:
            return "left"
        if next_pos[1] > current_pos[1]:
            return "down"
        if next_pos[1] < current_pos[1]:
            return "up"
        
    def navigate(self) -> None:
        current_idx = self.path.index(self.current_pos)
        next_idx = current_idx + self.direction
        self.next_pos = self.path[next_idx] if next_idx < len(self.path) else self.path[next_idx % len(self.path)]
        dir = self.compute_dir(self.current_pos, self.next_pos)
        print(f"Move from {self.current_pos} to {self.next_pos}")
        self.move(dir)
        self.wait(MAX_TIME_CHGMT_MAP_S)
        self.click(((WIDTH - LEFT) // 2, (HEIGHT - TOP) // 2))
        
    def wait(self, time_s: float) -> None:
        noise = random.random() * time_s * 0.2
        time.sleep(time_s + noise)
        
    def select_trees(self, centers: list) -> None:
        for i, center in enumerate(centers):
            self.click(center)
            if i == 0:
                self.wait(3.5)
            else: 
                self.wait(0.1)       
        
    def run(self):
        with mss.mss() as sct:
            while True:
                self.click(((WIDTH - LEFT) // 2, (HEIGHT - TOP) // 2))
                screen_capture = sct.shot(output="current_screen.png")
                screen = cv2.imread(screen_capture)
                centers = self.matcher.search(screen)
                cv2.imwrite("annotated_screen.png", screen)
                self.select_trees(centers)
                self.navigate()
                self.wait(4.5*len(centers))