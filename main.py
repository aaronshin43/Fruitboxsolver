import pyautogui
import time

time.sleep(2)
#Screenshot current display after 2 seconds
screenshot = pyautogui.screenshot(region=(599, 222, 1334, 867))  # Define game window area
screenshot.save("game_screen.png")