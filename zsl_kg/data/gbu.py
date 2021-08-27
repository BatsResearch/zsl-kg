import os.path as osp

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class GBU(Dataset):
    def __init__(self, path, indices_list, labels, stage="train"):
        """This is a dataloader for the processed good bad ugly dataset.
        Assuming that the images have been cropped and save as 0-indexed
        files in the path. Assuming the indices list is also 0-indexed.
        Args:
            path (str): path of the processed dataset
            indices_list (list): list of image indices to be list
            labels (list): contains integer labels for the images
            stage (str, optional): loads a different transforms if test. Defaults to 'train'.
        """
        self.data = []
        for i in range(len(indices_list)):
            image_file = str(indices_list[i]) + ".jpg"
            self.data.append((osp.join(path, image_file), labels[i]))

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if stage == "train":
            self.transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        if stage == "test":
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i]
        image = Image.open(path).convert("RGB")
        image = self.transforms(image)
        if (
            image.shape[0] != 3
            or image.shape[1] != 224
            or image.shape[2] != 224
        ):
            print("you should delete this guy:", path)
        return image, label
