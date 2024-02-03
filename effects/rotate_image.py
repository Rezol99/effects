import numpy as np
import cv2

# TODO: type ignoreをなくす

def _rotate_image_x(mat: np.ndarray, angle: float) -> np.ndarray:
    rotation_vector = np.array([angle, 0, 0])
    R, _ = cv2.Rodrigues(rotation_vector) # type: ignore
    return np.dot(mat, R)


def _rotate_image_y(mat: np.ndarray, angle: float) -> np.ndarray:
    rotation_vector = np.array([0, angle, 0])
    R, _ = cv2.Rodrigues(rotation_vector) # type: ignore
    return np.dot(mat, R)


def _rotate_image_z(mat: np.ndarray, angle: float) -> np.ndarray:
    rotation_vector = np.array([0, 0, angle])
    R, _ = cv2.Rodrigues(rotation_vector) # type: ignore
    return np.dot(mat, R)

def rotate_image_xyz(image: np.ndarray, params: dict) -> np.ndarray:
    angle_x = params.get("x", 0)
    angle_y = params.get("y", 0)
    angle_z = params.get("z", 0)

    # ラジアンに変換
    angle_x = np.deg2rad(angle_x)
    angle_y = np.deg2rad(angle_y)
    angle_z = np.deg2rad(angle_z)

    # 回転行列を計算
    mat = np.eye(3)
    mat = _rotate_image_x(mat, angle_x)
    mat = _rotate_image_y(mat, angle_y)
    mat = _rotate_image_z(mat, angle_z)

    height, width = image.shape[:2]
    center_x = width / 2
    center_y = height / 2

    mat[0, 2] = center_x - center_x * mat[0, 0] - center_y * mat[0, 1]
    mat[1, 2] = center_y - center_x * mat[1, 0] - center_y * mat[1, 1]
    mat = mat[:2, :]

    if image.shape[2] < 4:  # RGB画像の場合
        out_img = cv2.warpAffine(image, mat, (width, height)) # type: ignore
    else:  # アルファチャンネルを含む画像の場合
        img = image
        alpha_channel = img[:, :, 3]
        rgb_channels = img[:, :, :3]

        # RGBチャンネルだけで変換を計算
        planar_img = cv2.merge( # type: ignore
            [rgb_channels[:, :, 0], rgb_channels[:, :, 1], rgb_channels[:, :, 2]]
        )
        rotated_img = cv2.warpAffine(planar_img, mat, (width, height)) # type: ignore

        # アルファチャンネルだけで変換を計算
        alpha_img = cv2.merge([alpha_channel, alpha_channel, alpha_channel]) # type: ignore
        rotated_alpha = cv2.warpAffine(alpha_img, mat, (width, height)) # type: ignore

        # RGBチャンネルとアルファチャネルをマージ
        out_img = cv2.merge( # type: ignore
            [
                rotated_img[:, :, 0],
                rotated_img[:, :, 1],
                rotated_img[:, :, 2],
                rotated_alpha[:, :, 0],
            ]
        )

    return out_img
