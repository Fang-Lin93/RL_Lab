import keyboard
from consts import KEYBOARD_MAP


async def keyboard_act():
    for key, v in KEYBOARD_MAP.items():
        if keyboard.is_pressed(key):
            print(f'You Pressed {key}')
            return v


if __name__ == '__main__':
    while True:
        action = keyboard_act()
        print(action)