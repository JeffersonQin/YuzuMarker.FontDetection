import subprocess

total_mission = 64
min_mission = 1
max_mission = 64

for i in range(min_mission, max_mission + 1):
    subprocess.Popen(
        [
            "python",
            "font_ds_generate_script.py",
            str(i),
            str(total_mission),
        ]
    )
