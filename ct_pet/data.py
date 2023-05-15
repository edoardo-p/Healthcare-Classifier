import torch


class CTData(torch.utils.data.Dataset):
    def __init__(self, images, masks):
        super().__init__()
        self.images = images
        self.masks = masks

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.images[idx]).unsqueeze(0),
            torch.tensor(self.masks[idx]).unsqueeze(0),
        )
