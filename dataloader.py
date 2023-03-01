import joblib
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import glob,os
import torch

class WSI(Dataset):
    def __init__(self, path, type):
        self.names = glob.glob(os.path.join(path,type,"*"))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        data=joblib.load(os.path.join(self.names[idx],"embeddings.joblib"))
        embeddings=torch.Tensor(data["embedding"][:]).squeeze(1)
        label=torch.Tensor([int(self.names[idx].split("_")[-1])])
        # label=torch.LongTensor([int(self.names[idx].split("_")[-1])])
        name= self.names[idx].split(os.sep)[-1]
        y=torch.Tensor(data["y"][:])
        x=torch.Tensor(data["x"][:])
        return embeddings,label,name, x, y

    def num_pat(self):
        return len(set([x.split('/')[10].split('_')[0] for x in self.names]))

    def num_class(self, cl:int):
        return len([x.split('_')[-1] for x in self.names if int(x.split('_')[-1]) == cl])

    # def coord(self):
    #     return (self.x, self.y)

# train_dataset=WSI("/mnt/beegfs/work/H2020DeciderFicarra/decider/feats/pfi_45_180/x5","train")
# train_dataloader= DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

# for slide in train_dataloader:
#     img,label,name=slide
#     print(img.shape,name)


# test_dataset=WSI("/mnt/beegfs/work/H2020DeciderFicarra/decider/feats/pfi_45_180/x5","test")
# test_dataloader= DataLoader(train_dataset, batch_size=1,shuffle=True, num_workers=0)

# for slide in test_dataloader:
#     img,label,name=slide
#     print(img.shape,name)