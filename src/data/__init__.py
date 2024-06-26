from .DataSetLoader import HippocampusDataset

data_dir = r'C:\Users\Adib\PycharmProjects\Hippocampus_Segmentation\dataset'
json_file = r'C:\Users\Adib\PycharmProjects\Hippocampus_Segmentation\dataset\dataset.json'

train_loader, val_loader, test_loader = HippocampusDataset.get_data_loaders(data_dir, json_file)