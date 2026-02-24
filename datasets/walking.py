import os

# folder_path = "1_Deroussi_and_Norre"
# folder_path = "2_Ham"
# folder_path = "3_Homayouni_and_Fontes-Brandimarte"
# folder_path = "4_Homayouni_and_Fontes-Fattahi"
folder_path = "5_Dauzere"
is_proc_folder = False

groups = {}

with os.scandir(folder_path) as entries:
    for entry in entries:
        if is_proc_folder:
            if entry.name != "trucks_data":
                with os.scandir(entry.path) as files:
                    for file in files:
                        with open(file.path, 'r', encoding='utf-8') as f:
                            num_jobs, num_machines, num_trucks = map(int, f.readline().split())
                            if f"{num_jobs} {num_machines} {num_trucks}" in groups:
                                groups[f"{num_jobs} {num_machines} {num_trucks}"].append(file.path)
                            else:
                                groups[f"{num_jobs} {num_machines} {num_trucks}"] = [file.path]
        else:
            if entry.name == "trucks_data":
                with os.scandir(entry.path) as files:
                    for file in files:
                        with open(file.path, 'r', encoding='utf-8') as f:
                            num_machines = len(list(map(float, f.readline().split()))) - 1
                            groups[str(num_machines)] = file.path

keys = sorted(list(groups.keys()), key=lambda x: list(map(int, x.split())))
for key in keys:
    print(key, groups[key])
