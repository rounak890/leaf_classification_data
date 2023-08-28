from pathlib import Path
from PIL import Image
import torch
import torchvision
from torchvision import transforms as T
import pandas as pd
from torch.utils.data import Dataset,DataLoader

train_csv = pd.read_csv('data\\train.csv',usecols=["id","species"])
test_csv = pd.read_csv('data\\test.csv',usecols=["id","species"])
x = train_csv.head()
# print(x)

# for i,j in x.iterrows():
#     print(j["id"],j["species"])

k = 1
path = f"C:/Users/RG/Desktop/leaf_classification/data/images/{k}.jpg"
# img_path = Path(path)
img = Image.open(path)
# print(img)

transform = T.Compose([
    T.Resize(size = (216,216)),
    T.ToTensor()
])

class leaf_dataset(Dataset):
    def __init__(self,csv):
        self.ids = []
        self.species = []
        for i,j in csv.iterrows():
            z = j["id"]
            path = f"C:/Users/RG/Desktop/leaf_classification/data/images/{z}.jpg"
            self.ids.append(transform(Image.open(path)))
            self.species.append(j["species"])

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,id):
        return self.ids[id],self.species[id]
    
train_data = leaf_dataset(x)
print(next(iter(train_data)))


# print(transform(img))


# so now data is finally transformed now we can focus on building model and copying this part to colab