# Activate your virtual psychopy environment in PowerShell first, e.g.:
# & "C:\Users\enric\OneDrive - Università degli Studi di Padova\Documents\.virtualenvs\psychopy-env\Scripts\Activate.ps1"

from psychopy import visual, core, event
import random
import pandas as pd

# ------------------------------------------------------------
# Window and stimuli
# ------------------------------------------------------------

win = visual.Window(size=(800, 600), color="white")
stim = visual.TextStim(win, height=0.12)

# ------------------------------------------------------------
# Define trials
# ------------------------------------------------------------

trials = [
    ("RED", "red"), ("RED", "green"), ("RED", "blue"),
    ("GREEN", "red"), ("GREEN", "green"), ("GREEN", "blue"),
    ("BLUE", "red"), ("BLUE", "green"), ("BLUE", "blue")
] * 2

random.shuffle(trials)

# ------------------------------------------------------------
# Response mapping
# ------------------------------------------------------------

keys = {
    "r": "red",
    "g": "green",
    "b": "blue"
}

# ------------------------------------------------------------
# Run experiment
# ------------------------------------------------------------

clock = core.Clock()
results = []

for i, (word, color) in enumerate(trials, start=1):

    # Fixation cross
    stim.text = "+"
    stim.color = "black"
    stim.draw()
    win.flip()
    core.wait(0.3)

    # Stroop stimulus
    stim.text = word
    stim.color = color
    stim.draw()
    win.flip()

    clock.reset()
    key, rt = event.waitKeys(
        keyList=["r", "g", "b", "escape"],
        timeStamped=clock
    )[0]

    if key == "escape":
        break

    chosen_color = keys[key]
    correct = chosen_color == color

    results.append({
        "trial": i,
        "word": word,
        "ink_color": color,
        "key": key,
        "response": chosen_color,
        "correct": correct,
        "rt": round(rt, 3)
    })

    # Accuracy feedback
    stim.text = "Correct!" if correct else "Wrong!"
    stim.color = "black"
    stim.draw()
    win.flip()
    core.wait(0.4)

    # Short interval
    win.flip()
    core.wait(0.3)

# ------------------------------------------------------------
# Close window and save data
# ------------------------------------------------------------

win.close()

results = pd.DataFrame(results)
results.to_csv("stroop_results.csv", index=False)

