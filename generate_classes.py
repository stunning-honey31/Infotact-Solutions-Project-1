import os
import json

train_dir = "data/train" 

classes = sorted(os.listdir(train_dir))

with open("classes.json", "w") as f:
    json.dump(classes, f, indent=4)

print("classes.json created:", classes)