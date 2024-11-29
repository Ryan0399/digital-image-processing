import os


def generate_file_list(directory, output_file):
    with open(output_file, "w") as f:
        for root, _, files in os.walk(directory):
            for file in sorted(files):
                if file.endswith(".jpg"):
                    f.write(os.path.join(root, file) + "\n")


# 生成 train_list.txt
generate_file_list(
    r"03_PlaywithGANs\Pix2Pix\datasets\facades\train",
    r"03_PlaywithGANs\Pix2Pix\train_list.txt",
)

# 生成 val_list.txt
generate_file_list(
    r"03_PlaywithGANs\Pix2Pix\datasets\facades\val",
    r"03_PlaywithGANs\Pix2Pix\val_list.txt",
)
