import numpy as np

def move_image(image: np.ndarray, params: dict) -> np.ndarray:
    """
    画像を移動させる
    [in]  image: 画像
    [in]  params: パラメータ
    [out] 移動させた画像
    """
    # パラメータを取得
    x = params.get("x", 0)
    y = params.get("y", 0)

    # 移動させる
    height, width = image.shape[:2]
    mat = np.float32([[1, 0, x], [0, 1, y]]) # type: ignore
    out_img = cv2.warpAffine(image, mat, (width, height)) # type: ignore

    return out_img
