import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.data import Dataset
import os

class DataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

        self.prepare_data()
        self.setup()

    def prepare_data(self, data_path=None) -> None:
        self.data = CNN_DM_Dataset()


    def setup(self, stage=None) -> None:
        # split data
        train, val, test = random_split(
            self.data,
            [287113, 13368, 11490],
        )

        if stage == "fit" or stage is None:
            self.train_set = train
            self.val_set = val

        if stage == "test":
            self.test_set = test

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=2)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=2)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=2)


class CNN_DM_Dataset(Dataset):
    def __init__(self, dir='data/processed/tokenized/'):
        self.input_ids = torch.tensor([])
        self.attention_mask = torch.tensor([])
        self.labels = torch.tensor([])
        for file in os.listdir('data/processed/tokenized/summary'):
            summary = torch.load('data/processed/tokenized/summary/' + file)
            text = torch.load('data/processed/tokenized/text/' + file)
            self.input_ids = torch.concat((self.input_ids, text['input_ids']))
            self.attention_mask = torch.concat((self.attention_mask, text['attention_mask']))
            self.labels = torch.concat((self.labels, summary['input_ids']))

    def __getitem__(self, index: int):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.input_ids)

if __name__ == '__main__':
    DataModule()
