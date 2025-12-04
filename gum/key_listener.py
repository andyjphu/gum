# /Users/andyphu/nlp/user-workflow-modeling/gum/gum/key_listener.py
from pynput import keyboard

def get_key_listener(manual_observer):
    def on_press(key):
        try:
            if key.char == 'p':
                print("[KeyLogger] found that p was pressed")
                # manual_observer.toggle_recording() #TODO: I disabled this for now
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    return listener
