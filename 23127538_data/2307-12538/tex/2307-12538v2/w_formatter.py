import os
import subprocess

# folder path
dir_path = "."

# list to store files
res = []

# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(path)

print(res)

for i_file in res:
    if i_file.endswith("tex"):
        subprocess.run(["latexindent", "-w", i_file])
