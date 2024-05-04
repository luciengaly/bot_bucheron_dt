import os

import numpy as np
import cv2

from imutils.object_detection import non_max_suppression


class TemplateMatcher: 
    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path
        self.thresholds = {
            "chataigner_1.png": 0.50,
            "chataigner_2.png": 0.70,
            "chataigner_3.png": 0.74,
            "chataigner_4.png": 0.95,
            "chataigner_5.png": 0.95,
            "chataigner_6.png": 0.95,
            "chataigner_7.png": 0.95,
            "chataigner_8.png": 0.95,
            "chataigner_9.png": 0.98,
            "chataigner_10.png": 0.95,
            "chataigner_11.png": 0.95,
            "chataigner_12.png": 0.95,
            "chataigner_13.png": 0.95,
            "chataigner_14.png": 0.95,
            "chene_1.png": 0.47,
            "chene_2.png": 0.71,
            "chene_3.png": 0.95,
            "chene_4.png": 0.95,
            "chene_5.png": 0.95,
            "chene_6.png": 0.90,
            "chene_7.png": 0.90,
            "chene_8.png": 0.90,
            "frene_1.png": 0.85,
            "frene_2.png": 0.85,
            "frene_3.png": 0.95,
            "frene_4.png": 0.95,
            "frene_5.png": 0.95,
            "frene_6.png": 0.95,
            "frene_7.png": 0.95,
            "frene_8.png": 0.95,
            "frene_9.png": 0.95,
            "frene_10.png": 0.95,
            "frene_11.png": 0.85,
            "frene_12.png": 0.95,
            "noyer_1.png": 0.47,
            "noyer_2.png": 0.95,
            "noyer_3.png": 0.95,
            "noyer_4.png": 0.95,
            "noyer_5.png": 0.95,
            "noyer_6.png": 0.95,
            "merisier_1.png": 0.80,
            "erable_1.png": 0.75,
            "erable_2.png": 0.95,
            "erable_3.png": 0.95,
            "erable_4.png": 0.95,
            "erable_5.png": 0.95,
            "erable_6.png": 0.95,
            "erable_7.png": 0.95,
            "erable_8.png": 0.95,
            "erable_9.png": 0.95,
        }
        self.templates = []
        self.load_templates()
        
    def load_templates(self) -> None:
        print("Loading templates...")
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                path = os.path.join(self.folder_path, filename)
                image = cv2.imread(path)
                if image is not None:
                    threshold = self.thresholds[filename]
                    self.templates.append((filename, image, threshold))
                    print(f"Loaded template {filename} with threshold {threshold}")
                else:
                    print(f"Failed to load image {filename}")
                    
    def match_templates(self, screen):
        print("Matching templates...")
        matches = []
        for filename, template, threshold in self.templates:
            res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                top_left = pt
                bottom_right = (
                    top_left[0] + template.shape[1],
                    top_left[1] + template.shape[0],
                )
                score = res[pt[1], pt[0]]
                matches.append((filename, top_left, bottom_right, score))
        print(f"Found {len(matches)} potential matches")
        return matches

    def apply_nms(self, matches):
        print("Applying Non-Maximum Suppression...")
        filtered_matches = []
        for filename in set(match[0] for match in matches):
            boxes = np.array(
                [
                    (x[1][0], x[1][1], x[2][0], x[2][1])
                    for x in matches
                    if x[0] == filename
                ]
            )
            scores = np.array([x[3] for x in matches if x[0] == filename])
            picked = non_max_suppression(boxes, probs=scores, overlapThresh=0.8)
            filtered_matches.extend(
                [
                    (filename, (p[0], p[1]), (p[2], p[3]), s)
                    for p, s in zip(picked, scores)
                ]
            )
        print(f"{len(filtered_matches)} matches after NMS")
        return filtered_matches
    
    def draw_matches(self, screen, matches: list):
        for filename, top_left, bottom_right, score in matches:
            cv2.rectangle(screen, top_left, bottom_right, (0, 0, 255), 1)
            label = f"{filename} ({score:.2f})"
            cv2.putText(
                screen,
                label,
                (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
    
    def convert_matches_to_centers(self, matches: list) -> list:
        centers = []
        for filename, top_left, bottom_right, score in matches:
            center_x, center_y = (
                top_left[0] + self.templates[0][1].shape[1] // 2,
                top_left[1] + self.templates[0][1].shape[0] // 2,
            )
            print(f"Matches at ({center_x}, {center_y}) on {filename} with score {score}")
            centers.append((center_x, center_y))
        return centers
    
    def search(self, screen) -> list:
        matches = self.match_templates(screen)
        matches = self.apply_nms(matches)
        self.draw_matches(screen, matches)
        centers = self.convert_matches_to_centers(matches)
        return centers
