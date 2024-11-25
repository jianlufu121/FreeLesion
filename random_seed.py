import os

def generate_files_with_seeds(txt_file_path, output_directory):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 打开 .txt 文件并读取所有行
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    # 遍历每一行，即每个图片的路径
    for line in lines:
        # 去除换行符并获取图片文件名
        image_path = line.strip()
        image_name = os.path.basename(image_path)

        # 构建输出目录中图片的路径
        output_image_path = os.path.join(output_directory, image_name)

        # 生成 31 个新文件，每个文件名后缀从 41 到 62 递增
        for i in range(10):
           
            new_filename = f"{image_name.split('.')[0]}_seed_{i+41}.{image_name.split('.')[1]}"

            # 构建新文件的完整路径
            new_file_path = os.path.join(output_directory, new_filename)

            # 复制图片到新文件名
            os.system(f"cp {image_path} {new_file_path}")

# 调用函数并传入 .txt 文件路径和输出目录路径
generate_files_with_seeds('/home/data/fupl/SAFM/dataset/ddr_512/synann.txt', 'home/data/fupl/SAFM/dataset/ddr_512/annotations/training_ann')

