import cv2
import numpy as np
import math

# HSV bounds for blue-arrow detection
arrow_minHSV = np.array([100, 100, 50])
arrow_maxHSV = np.array([140, 255, 255])

def detect_arrow(cnt, frame, hsv_frame):
    """
    Given a contour, the BGR frame, and its HSV version,
    returns one of "arrow (right|down|left|up)" if a blue arrow is detected,
    otherwise returns None.
    """
    # Only consider roughly quadrilateral-to-octagonal shapes
    peri = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    if area < 500 or len(cnt) < 4:
        return None
    epsilon = 0.03 * peri if area > 1000 else 0.05 * peri
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if not (4 <= len(approx) <= 8):
        return None

    # Mask for blue region and clean it up
    maskHSV = cv2.inRange(hsv_frame, arrow_minHSV, arrow_maxHSV)
    kernel = np.ones((5, 5), np.uint8)
    maskHSV = cv2.morphologyEx(maskHSV, cv2.MORPH_CLOSE, kernel)
    maskHSV = cv2.morphologyEx(maskHSV, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Crop to contour bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    x, y = max(x - 10, 0), max(y - 10, 0)
    w = min(w + 20, frame.shape[1] - x)
    h = min(h + 20, frame.shape[0] - y)
    region = maskHSV[y:y + h, x:x + w]
    if region.size == 0:
        return None

    # Find the two strongest corners in the blurred mask
    blur = cv2.GaussianBlur(region, (9, 9), 0)
    corners = cv2.goodFeaturesToTrack(blur, maxCorners=2, qualityLevel=0.7, minDistance=15)
    if corners is None or len(corners) < 2:
        return None

    # Compute the mid-point of the two corners
    corners = np.int0(corners).reshape(-1, 2)
    (x0, y0), (x1, y1) = corners + [x, y]
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2

    # Compute center of the contour
    (cx, cy), _ = cv2.minEnclosingCircle(cnt)

    # Compute angle from center to midpoint
    angle = math.degrees(math.atan2(my - cy, mx - cx))

    # Classify direction
    if -45 <= angle < 45:
        return "arrow right"
    elif 45 <= angle < 135:
        return "arrow down"
    elif angle >= 135 or angle <= -135:
        return "arrow left"
    else:
        return "arrow up"
