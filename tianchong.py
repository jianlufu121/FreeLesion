from PIL import Image

# 打开原始图像
original_image = Image.open("IDRiD_55.jpg")

# 创建一个白色背景的新图像
new_image = Image.new("RGB", (768,768), (0, 0, 0))

# 计算将原始图像放置在新图像中的位置
x_offset = (512 - original_image.width) // 2
y_offset = (512 - original_image.height) // 2

# 将原始图像粘贴到新图像中
new_image.paste(original_image, (x_offset, y_offset))
new_image.save("IDRiD_55_padd.jpg")
resized_image = new_image.resize((512, 512))

# 保存填充后的图像
resized_image.save("IDRiD_55_paddingresize.jpg")

