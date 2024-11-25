from argparse import ArgumentParser
import os
import numpy as np
from PIL import Image
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mmseg.apis import init_model
import joblib

class InferDataset(Dataset):
    def __init__(self, img_root, mask_root):
        self.img_root = img_root
        self.mask_root = mask_root
        self.filenames = os.listdir(self.img_root)
        self.filenames.sort()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def __getitem__(self, item):
        filename = self.filenames[item]
        img = Image.open(os.path.join(self.img_root, filename)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_root, filename.replace('.jpg', '.png')))
        return self.transform(img), torch.from_numpy(np.array(mask)).long(), filename
    
    def __len__(self):
        return len(self.filenames)
    

def calculate_real_image_clusters():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    
    parser.add_argument('--real-img-path', type=str, required=True)
    parser.add_argument('--real-mask-path', type=str, required=True)
    
    args = parser.parse_args()
    
    model = init_model(args.config, checkpoint=args.checkpoint, device=args.device)
    model.eval()
    
    dataset_real = InferDataset(img_root=args.real_img_path, mask_root=args.real_mask_path)
    trainloader_real = DataLoader(dataset_real, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    csv_dir = 'csv_files_real/'
    os.makedirs(csv_dir, exist_ok=True)
    
    real_class_losses = {i: [] for i in range(1, 6)}
    for i, (img, mask, _) in enumerate(tqdm(trainloader_real)):
        img, mask = img.cuda(), mask.cuda()
        classes = torch.unique(mask).tolist()
    
        with torch.no_grad():
            preds = model.predict(img)
            preds = torch.cat([pred.seg_logits.data.unsqueeze(0) for pred in preds])
    
        loss = criterion(preds, mask - 1)
    
        for class_ in classes:
            if class_ == 0:
                continue
            real_class_losses[class_].extend(loss[mask == class_].cpu().numpy())
    
    real_class_clusters = {}
    for class_id in range(1, 6):
        loss_values = np.array(real_class_losses[class_id]).reshape(-1, 1)
        #n_clusters = min(10, len(loss_values) // 100)
        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters).fit(loss_values)
        real_class_clusters[class_id] = kmeans
    
        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(loss_values)), loss_values, c=kmeans.labels_, cmap='viridis', s=2)
        plt.title(f'Class {class_id} Real Image Clustering Analysis')
        plt.xlabel('Index')
        plt.ylabel('Loss Value')
        plt.colorbar(label='Cluster Label')
        plt.savefig(os.path.join(csv_dir, f'class_{class_id}_real_clustering.png'))
        plt.close()
    
    joblib.dump(real_class_clusters, 'real_class_clusters.pkl')
    print('Real image clusters saved to real_class_clusters.pkl')

if __name__ == '__main__':
    calculate_real_image_clusters()
