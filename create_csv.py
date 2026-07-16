import os
import pandas as pd

image_folder = "dataset/train"
images = [f.replace(".png", "") for f in os.listdir(image_folder) if f.endswith(".png")]

grades = []
for i, img in enumerate(images):
    grades.append(i % 5)

df = pd.DataFrame({"id_code": images, "diagnosis": grades})
df.to_csv("dataset/train.csv", index=False)

print(f"Created train.csv with {len(df)} entries")
print(df["diagnosis"].value_counts().sort_index())
