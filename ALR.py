import os
import numpy as np
import cv2
import random
from scipy.ndimage import label

def get_connected_components(label_img, lesion_value):
    binary_img = np.where(label_img == lesion_value, 1, 0).astype(np.uint8)
    labeled_img, num_features = label(binary_img)
    components = [np.column_stack(np.where(labeled_img == i)) for i in range(1, num_features + 1)]
    return components

def calculate_ratios(merged_label):
    total_pixels = np.sum(merged_label > 1)
    hard_exudates_ratio = np.sum(merged_label == 60) / total_pixels if total_pixels > 0 else 0
    hemorrhages_ratio = np.sum(merged_label == 120) / total_pixels if total_pixels > 0 else 0
    microaneurysms_ratio = np.sum(merged_label == 180) / total_pixels if total_pixels > 0 else 0
    soft_exudates_ratio = np.sum(merged_label == 240) / total_pixels if total_pixels > 0 else 0
    return hard_exudates_ratio, hemorrhages_ratio, microaneurysms_ratio, soft_exudates_ratio

def select_components(components, num_pixels):
    selected_pixels = []
    for component in components:
        if len(selected_pixels) + len(component) <= num_pixels:
            selected_pixels.extend(component)
        else:
            remaining_pixels = num_pixels - len(selected_pixels)
            if remaining_pixels > 0:
                selected_pixels.extend(component[:remaining_pixels])
            break
    return selected_pixels

def merge_labels(hard_exudates_label, hemorrhages_label, microaneurysms_label, soft_exudates_label, mask):#Modify category ratio
    target_ratios = {
        60: 0.285,  
        120: 0.354,  
        180: 0.037,
        240: 0.068,  
    }

    merged_label = mask.copy()

    hard_exudates_ratio, hemorrhages_ratio, microaneurysms_ratio, soft_exudates_ratio = calculate_ratios(merged_label)

    hard_exudates_components = get_connected_components(hard_exudates_label, 60)
    hemorrhages_components = get_connected_components(hemorrhages_label, 120)
    microaneurysms_components = get_connected_components(microaneurysms_label, 180)
    soft_exudates_components = get_connected_components(soft_exudates_label, 240)

    total_pixels = np.sum((mask == 10) | (mask == 30))

    if hard_exudates_ratio < target_ratios[60]:
        required_pixels = int(total_pixels * target_ratios[60] - np.sum(merged_label == 60))
        selected_pixels = select_components(hard_exudates_components, required_pixels)
        for coord in selected_pixels:
            if merged_label[coord[0], coord[1]] == 10:  # Ensure it's background before adding
                merged_label[coord[0], coord[1]] = 60

    if hemorrhages_ratio < target_ratios[120]:
        required_pixels = int(total_pixels * target_ratios[120] - np.sum(merged_label == 120))
        selected_pixels = select_components(hemorrhages_components, required_pixels)
        for coord in selected_pixels:
            if merged_label[coord[0], coord[1]] == 10:  # Ensure it's background before adding
                merged_label[coord[0], coord[1]] = 120

    if microaneurysms_ratio < target_ratios[180]:
        required_pixels = int(total_pixels * target_ratios[180] - np.sum(merged_label == 180))
        selected_pixels = select_components(microaneurysms_components, required_pixels)
        for coord in selected_pixels:
            if merged_label[coord[0], coord[1]] == 10:  # Ensure it's background before adding
                merged_label[coord[0], coord[1]] = 180

    if soft_exudates_ratio < target_ratios[240]:
        required_pixels = int(total_pixels * target_ratios[240] - np.sum(merged_label == 240))
        selected_pixels = select_components(soft_exudates_components, required_pixels)
        for coord in selected_pixels:
            if merged_label[coord[0], coord[1]] == 10:  # Ensure it's background before adding
                merged_label[coord[0], coord[1]] = 240

    return merged_label

def process_labels(hard_exudates_folder, hemorrhages_folder, microaneurysms_folder, soft_exudates_folder, mask_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    hard_exudates_files = os.listdir(hard_exudates_folder)
    hemorrhages_files = os.listdir(hemorrhages_folder)
    microaneurysms_files = os.listdir(microaneurysms_folder)
    soft_exudates_files = os.listdir(soft_exudates_folder)

    for mask_file in os.listdir(mask_path):

        hard_exudates_file = random.choice(hard_exudates_files)
        hemorrhages_file = random.choice(hemorrhages_files)
        microaneurysms_file = random.choice(microaneurysms_files)
        soft_exudates_file = random.choice(soft_exudates_files)

        hard_exudates_label = cv2.imread(os.path.join(hard_exudates_folder, hard_exudates_file), cv2.IMREAD_GRAYSCALE)
        hemorrhages_label = cv2.imread(os.path.join(hemorrhages_folder, hemorrhages_file), cv2.IMREAD_GRAYSCALE)
        microaneurysms_label = cv2.imread(os.path.join(microaneurysms_folder, microaneurysms_file), cv2.IMREAD_GRAYSCALE)
        soft_exudates_label = cv2.imread(os.path.join(soft_exudates_folder, soft_exudates_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_path, mask_file), cv2.IMREAD_GRAYSCALE)

        merged_label = merge_labels(hard_exudates_label, hemorrhages_label, microaneurysms_label, soft_exudates_label, mask)

        output_path = os.path.join(output_folder, f'{os.path.splitext(mask_file)[0]}.png')
        cv2.imwrite(output_path, merged_label)

        print(f'Saved: {output_path}')

def main():
    hard_exudates_folder = '/home/data/fupl/FreestyleNet/synlabel/train_ex_ddr/'
    hemorrhages_folder = '/home/data/fupl/FreestyleNet/synlabel/train_HE_ddr/'
    microaneurysms_folder = '/home/data/fupl/FreestyleNet/synlabel/train_MA_ddr/'
    soft_exudates_folder = '/home/data/fupl/FreestyleNet/synlabel/train_SE_ddr/'
    mask_path = '/home/data/fupl/FreestyleNet/synlabel/validation_ddr/validation/'  
    output_folder = '/home/data/fupl/FreestyleNet/synlabel/0904_synlabel_ddr/'

    process_labels(hard_exudates_folder, hemorrhages_folder, microaneurysms_folder, soft_exudates_folder, mask_path, output_folder)

if __name__ == '__main__':
    main()
