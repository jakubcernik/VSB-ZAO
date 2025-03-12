import cv2, pyautogui, time, numpy as np

template = cv2.imread('template.png', 0)
if template is None:
    raise FileNotFoundError("File not found!")

template_h, template_w = template.shape
screen_w, screen_h = pyautogui.size()

def move_and_click(x, y):
    pyautogui.moveTo(x, y)
    pyautogui.click()

while True:
    screenshot = np.array(pyautogui.screenshot().convert('L'))
    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val > 0.5:
        center_x, center_y = max_loc[0] + template_w // 2, max_loc[1] + template_h // 2
        scale_x, scale_y = screen_w / screenshot.shape[1], screen_h / screenshot.shape[0]
        target_x, target_y = int(center_x * scale_x), int(center_y * scale_y)
        move_and_click(target_x, target_y)
        time.sleep(0.3)