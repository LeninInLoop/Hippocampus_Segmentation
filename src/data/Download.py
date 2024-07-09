import tarfile
import gdown
import os


def download_dataset(dataset_dir="dataset"):
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'https://drive.google.com/uc?id=1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C&export=download'
    output_tar = os.path.join(dataset_dir, 'Task04_Hippocampus.tar')
    if not os.path.exists(output_tar):
        gdown.download(url, output_tar, quiet=False)
    else:
        print("Output tar {} already exists!".format(output_tar))

    tar = tarfile.open(output_tar)
    tar.extractall(path=dataset_dir)
    tar.close()
