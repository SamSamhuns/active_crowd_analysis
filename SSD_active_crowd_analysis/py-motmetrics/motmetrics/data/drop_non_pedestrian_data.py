import os
import tempfile

if __name__ == "__main__":
    rootdir = "MOT16"

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            full_path = os.path.join(subdir, file)
            extn = full_path.split(".")[-1]
            if extn == "txt":
                temp = tempfile.NamedTemporaryFile(mode="r+")

                with open(full_path, "r") as fr:
                    for line in fr:
                        splitted = line.strip().split(",")
                        # if splitted[7] == '1':
                        if float(splitted[-1]) > 0.7:
                            temp.write(line)
                temp.seek(0)
                with open(full_path, "w") as fw:
                    for line in temp:
                        fw.write(line)

                temp.close()
