import numpy as np
import cv2


def execute(image: np.ndarray, params: dict) -> np.ndarray:
    amount = params.get("amount", 0)
    if amount > 0:
        out = cv2.blur(image, (amount, amount)) # type: ignore
        return out
    return image