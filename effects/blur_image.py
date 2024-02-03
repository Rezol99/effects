import numpy as np
import cv2


def execute(image: np.ndarray, params: dict) -> np.ndarray:
    intensity = params.get("intensity", 0)
    if intensity > 0:
        out = cv2.blur(image, (intensity, intensity)) # type: ignore
        return out
    return image