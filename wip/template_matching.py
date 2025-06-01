import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pyautogui

time.sleep(3)
# Step 1: Capture mouse position and screenshot
mouse_pos = pyautogui.position()
screenshot = pyautogui.screenshot()
img = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2GRAY)

# Step 2: Get dimensions
screenshot_width, screenshot_height = screenshot.size
screen_width, screen_height = pyautogui.size()

# Step 3: Check if scaling is present
print(
    f"screenshot_width={screenshot_width}, screenshot_height={screenshot_height},"  # noqa: E501
    f"screen_width={screen_width}, screen_height={screen_height}"
)
if (screenshot_width, screenshot_height) != (screen_width, screen_height):
    # Scaling is active, calculate scale factors
    scale_x = screenshot_width / screen_width
    scale_y = screenshot_height / screen_height
    mouse_scaled = (int(mouse_pos[0] * scale_x), int(mouse_pos[1] * scale_y))
    print(f"Scaling detected. Mouse scaled: {mouse_scaled}")
else:
    # No scaling
    mouse_scaled = (mouse_pos[0], mouse_pos[1])
    print(f"No scaling. Mouse unchanged: {mouse_scaled}")

# Step 4: perform template matching
img2 = img.copy()
template = cv.imread("data/target.png", cv.IMREAD_GRAYSCALE)
assert template is not None, (
    "file could not be read, check with os.path.exists()"
)
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = [
    # "TM_CCOEFF",
    # "TM_CCORR",  # do not
    # "TM_SQDIFF",
    # "TM_CCOEFF_NORMED",
    "TM_CCORR_NORMED",
    "TM_SQDIFF_NORMED",
]

for meth in methods:
    img = img2.copy()
    method = getattr(cv, meth)

    # Apply template Matching
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    print(
        f"{meth}: min_val={min_val}, max_val={max_val},"
        f" min_loc={min_loc}, max_loc={max_loc}"
    )
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # draw square around entire screen
    side_len = max(screenshot_width, screenshot_height)
    print("side_len", side_len)
    square_top_left = (0, 0)
    square_bottom_right = (side_len, side_len)
    # grayscale identification
    # cv.rectangle(img, square_top_left, square_bottom_right, 0, 10)
    # # draw rectangle around target
    # cv.rectangle(img, top_left, bottom_right, 0, 10)
    # # draw circle centered on mouse
    # cv.circle(img, mouse_scaled, radius=15, color=0, thickness=5)
    # color identification
    img_rgb = cv.cvtColor(img2.copy(), cv.COLOR_GRAY2RGB)
    cv.rectangle(img_rgb, square_top_left, square_bottom_right, (0, 0, 0), 10)
    cv.rectangle(img_rgb, top_left, bottom_right, (255, 0, 0), 10)
    cv.circle(
        img_rgb, mouse_scaled, radius=15, color=(0, 255, 0), thickness=5
    )

    # show result
    plt.subplot(121), plt.imshow(res, cmap="gray")
    plt.title("Matching Result"), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(img, cmap="gray")  # grayscale identification
    plt.subplot(122), plt.imshow(img_rgb)  # color identification
    plt.title(f"Detected Points: {meth}"), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    import datetime as dt
    import os

    file_dir = os.getcwd()
    data_dir = "data"
    test_dir = "test"
    datetime = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    filename = f"template_match_{meth}_{datetime}.png"
    file_path = os.path.join(file_dir, data_dir, test_dir, filename)
    plt.savefig(file_path)
    plt.show()

    print(
        f">> target region (top_left, size):, top_left={top_left}, width={w}, height={h}"  # noqa: E501
    )
    print(f">> mouse region (scaled) location: {mouse_scaled}")
