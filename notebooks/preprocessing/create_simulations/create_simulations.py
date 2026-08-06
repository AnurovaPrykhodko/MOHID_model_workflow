"""
Module to automate creation of simulation configuration files 
with varying parameters to facilitate sensitivity analysis and calibration. 

- Copies files from template simulation to the rest.

(add?)
- choice to change parameters for respective simulation.

Author: Karolina Anurova-Prykhodko
"""

import re
import shutil
from pathlib import Path

data_dir = Path(r"C:\Users\karoa\mohid_runs\tenerife\tenerife2D\data")

# when a new simulation is created in MOHID, it does not copy
# the template simulation configuration files.
# the new files have suffixes _<index> for respective simulation
# Match filenames whose stem ends with _<number>, capturing the base and the number
suffix_pattern = re.compile(r"^(.*)_(\d+)$")

# First pass: build a set of base names that have a _1 source file available
sources = {}  # base_name -> Path to the _1 file
for f in data_dir.iterdir():
    if not f.is_file():
        continue
    m = suffix_pattern.match(f.stem)
    if m and m.group(2) == "1":
        sources[(m.group(1), f.suffix)] = f

# Second pass: for every file with a numeric suffix other than _1, replace it
for f in data_dir.iterdir():
    if not f.is_file():
        continue
    m = suffix_pattern.match(f.stem)
    if not m:
        continue
    base, num = m.group(1), m.group(2)
    if num == "1":
        continue  # skip the source files themselves
    
    source_file = sources.get((base, f.suffix))
    if source_file is None:
        print(f"Skipped (no _1 source found): {f.name}")
        continue
    
    shutil.copy2(source_file, f)
    print(f"Copied: {source_file.name} -> {f.name}")

print("Template simulation files copied.")