import pyautogui
import time

time.sleep(2)
#Screenshot current display after 2 seconds
#print(pyautogui.position())
screenshot = pyautogui.screenshot(region=(725, 352, 1046, 615))  # Define game window area
screenshot.save("game_screen.png")