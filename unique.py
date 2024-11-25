from PIL import Image

def get_unique_pixels(png_file):
    # 打开 PNG 图像文件
    img = Image.open(png_file)
    
    # 获取图像的所有像素值
    pixels = list(img.getdata())
    
    # 去除重复的像素值，得到唯一像素值
    unique_pixels = list(set(pixels))
    
    return unique_pixels

if __name__ == "__main__":
    png_file = "IDRiD_04_seed_49.png"  # 替换为你要处理的 PNG 图像文件路径
    unique_pixels = get_unique_pixels(png_file)
    
    print("Unique pixels in the image:")
    print(unique_pixels)

