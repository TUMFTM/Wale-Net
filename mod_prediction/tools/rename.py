import os

directory = "data/sc_test/"
file_list = os.listdir(directory)
file_list.sort()

for i in file_list:
    no = int(i.split(".", 1)[0])
    new_name = str(no - 1).zfill(8) + ".png"
    print(new_name, end="\r")
    os.rename(os.path.join(directory, i), os.path.join(directory, new_name))
