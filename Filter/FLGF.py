from argparse import ArgumentParser
import os
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from mmseg.models.utils import resize
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

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--real-img-path', type=str, required=True)
    parser.add_argument('--real-mask-path', type=str, required=True)
    parser.add_argument('--syn-img-path', type=str, required=True)
    parser.add_argument('--syn-mask-path', type=str, required=True)
    parser.add_argument('--filtered-mask-path', type=str, required=True)
    parser.add_argument('--tolerance-margin-loss', type=float, default=1.35)
    parser.add_argument('--tolerance-margin-kmeans', type=float, default=1.7)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--cluster-viz-path', type=str, required=True)
    args = parser.parse_args()
    
    model = init_model(args.config, checkpoint=args.checkpoint, device=args.device)
    model.eval()
    
    dataset_real = InferDataset(img_root=args.real_img_path, mask_root=args.real_mask_path)
    trainloader_real = DataLoader(dataset_real, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    
    dataset_syn = InferDataset(img_root=args.syn_img_path, mask_root=args.syn_mask_path)
    trainloader_syn = DataLoader(dataset_syn, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    
    os.makedirs(args.filtered_mask_path, exist_ok=True)
    os.makedirs(args.cluster_viz_path, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    
    class_wise_mean_loss = [(0, 0) for _ in range(5)]
    
    class_wise_mean_loss = [(0, 0) for _ in range(5)]
    real_class_clusters = joblib.load('real_class_clusters.pkl')
    real_cluster_means = {}


    for class_, kmeans1 in real_class_clusters.items():
        cluster_means = [np.mean(kmeans1.cluster_centers_[i]) for i in range(kmeans1.n_clusters)]
        sorted_indices = np.argsort(cluster_means)
        real_cluster_means[class_] = sorted_indices
    
    
    # Calculate the class-wise mean loss on real images
    for i, (img, mask, _) in enumerate(tqdm(trainloader_real)):
        img, mask = img.cuda(), mask.cuda()
        classes = torch.unique(mask).tolist()
        
        with torch.no_grad():
            img_metas = [{'img_shape': img.shape[2:], 'ori_shape': img.shape[2:], 'pad_shape': img.shape[2:], 'scale_factor': 1.0}]
            preds = model.encode_decode(img, img_metas)
            preds = F.interpolate(preds, size=mask.shape[-2:], mode='bilinear', align_corners=False)

        loss = criterion(preds, mask - 1)
        
        for class_ in classes:
            if class_ == 0:
                continue
            pixel_num, loss_sum = class_wise_mean_loss[class_ - 1]
            class_wise_mean_loss[class_ - 1] = (pixel_num + torch.sum(mask == class_).item(), loss_sum + torch.sum(loss[mask == class_]).item())
    
    class_wise_mean_loss = [loss_sum / (pixel_num + 1e-5) for pixel_num, loss_sum in class_wise_mean_loss]
   # print('Class-wise mean loss:')
   # print(class_wise_mean_loss)
    
    # Filter out noisy synthetic pixels using both clustering and class-wise loss
    for i, (img, mask, filenames) in enumerate(tqdm(trainloader_syn)):
        img, mask = img.cuda(), mask.cuda()

        classes = torch.unique(mask).tolist()
        mask_filtered = mask.clone()

        with torch.no_grad():
        
            preds1 = model.predict(img)
            preds1 = torch.cat([pred.seg_logits.data.unsqueeze(0) for pred in preds1])
            img_metas = [{'img_shape': img.shape[2:], 'ori_shape': img.shape[2:], 'pad_shape': img.shape[2:], 'scale_factor': 1.0}]
            preds = model.encode_decode(img, img_metas)
            preds = F.interpolate(preds, size=mask.shape[-2:], mode='bilinear', align_corners=False)
            
            # Extract features for clustering
            features = model.extract_feat(img)
            outs = []
            for idx in range(len(features)):
                x = features[idx]
                conv = model.decode_head.convs[idx]
                outs.append(resize(input=conv(x), size=features[0].shape[2:], mode='bilinear', align_corners=False))
            
            out_tensor = model.decode_head.fusion_conv(torch.cat(outs, dim=1))
            out_tensor_upsampled = resize(input=out_tensor, size=(mask.size(1), mask.size(2)), mode='bilinear', align_corners=False)

        # Perform K-means clustering
        features_np = out_tensor_upsampled.permute(0, 2, 3, 1).cpu().numpy().reshape(-1, out_tensor.shape[1])
        
        # Create mask for non-background pixels
        non_background_mask = (mask.cpu().numpy() != 1).flatten()
        
        # Ensure the length of non-background mask matches features_np
        if len(non_background_mask) != features_np.shape[0]:
            raise ValueError("Mismatch between feature dimensions and mask dimensions.")
        
        lesion_features_np = features_np[non_background_mask]
        
        if lesion_features_np.shape[0] == 0:
            raise ValueError("No non-background pixels found for clustering.")
        
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(lesion_features_np)
        cluster_labels = np.zeros(features_np.shape[0], dtype=np.int32)
        cluster_labels[non_background_mask] = kmeans.labels_
        labels = cluster_labels.reshape(out_tensor_upsampled.shape[2], out_tensor_upsampled.shape[3])
        
        # Visualize clustering results
        cluster_viz = Image.fromarray(labels.astype(np.uint8) * (255 // args.n_clusters))
        cluster_viz.save(os.path.join(args.cluster_viz_path, filenames[0].replace('.jpg', '_cluster.png')))
        
        # Calculate average distance of each cluster center
        distances = {i: [] for i in range(args.n_clusters)}
        for x in range(out_tensor_upsampled.size(2)):
            for y in range(out_tensor_upsampled.size(3)):
                if mask[0, x, y] == 1:
                    continue

                feature_vector = out_tensor_upsampled[0, :, x, y].cpu().numpy()
                cluster_center = kmeans.cluster_centers_[labels[x, y]]
                distance = np.linalg.norm(feature_vector - cluster_center)
                distances[labels[x, y]].append(distance)

        avg_distances = {i: np.mean(dists) if len(dists) > 0 else 0 for i, dists in distances.items()}
        print("Cluster average distances:")
        print(avg_distances)
        loss = criterion(preds1, mask - 1)

        # Filter pixels
        for class_ in classes:
                
            if class_ == 0 or class_ not in real_cluster_means:
                continue
            
            kmeans1 = real_class_clusters[class_]
            loss_values = loss[mask == class_].cpu().numpy().reshape(-1, 1)
            cluster_labels1 = kmeans1.predict(loss_values)
            sorted_indices = real_cluster_means[class_]
            cluster_means = np.array([np.mean(loss_values[cluster_labels1 == i]) for i in range(kmeans1.n_clusters)])
            
            high_loss_clusters = np.where(cluster_means > class_wise_mean_loss[class_ - 1])[0]

            cluster_labels_tensor = torch.tensor(cluster_labels1, device=mask.device)
            mask_indices = (mask == class_)
            filtered_region = torch.zeros_like(mask, dtype=torch.bool)
            filtered_region[mask_indices] = torch.tensor(np.isin(cluster_labels1, high_loss_clusters), device=mask.device).squeeze()

            coords = torch.nonzero(mask == class_, as_tuple=True)
            if class_ == 1:
                # For class 1, only use class-wise mean loss for filtering
                for x, y in zip(coords[1].tolist(), coords[2].tolist()):
                    if filtered_region[0, x, y].item():
                        mask_filtered[0, x, y] = 0
            else:
                # For other classes, use both clustering and class-wise mean loss
                for x, y in zip(coords[1].tolist(), coords[2].tolist()):
                    feature_vector = out_tensor_upsampled[0, :, x, y].cpu().numpy()
                    cluster_label = labels[x, y]
                    distance = np.linalg.norm(feature_vector - kmeans.cluster_centers_[cluster_label])
                    
                    threshold_dist = args.tolerance_margin_kmeans * avg_distances[cluster_label]
  #                  threshold_loss = args.tolerance_margin_loss * class_wise_mean_loss[class_ - 1]
                    
                    if distance > threshold_dist or filtered_region[0, x, y].item():
                        mask_filtered[0, x, y] = 0
                
                

        mask_filtered = mask_filtered.cpu().numpy().astype(np.uint8)
        mask_filtered_img = Image.fromarray(mask_filtered[0])
        mask_filtered_img.save(os.path.join(args.filtered_mask_path, filenames[0].replace('.jpg', '.png')))


if __name__ == '__main__':
    main()


