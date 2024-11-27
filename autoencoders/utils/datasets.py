import os
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
            limit (int, optional): Maximum number of images to load for a sanity check.
        """
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) 
                             if fname.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if limit is not None:
            self.image_paths = self.image_paths[:limit]  # Limit the number of images
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label