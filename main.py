from constant import *
from controller import ScreenController
from matcher import TemplateMatcher


def main():
    """Main function"""
    folder_path = r"data\raw"
    matcher = TemplateMatcher(folder_path)
    current_pos = (5, 17)
    direction = 1
    
    controller = ScreenController(matcher, SHORT_PATH_2, current_pos, direction=direction)
    controller.run()

if __name__ == "__main__":
    main()