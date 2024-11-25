import numpy as np
from PIL import Image
file_path="/home/data/fupl/FreestyleNet/IDRiD_01.png"
img=Image.open(file_path)
img_array=np.array(img)
#img_array=img_array+1
img_array=img_array*60
new_img = Image.fromarray(img_array)
new_img.save(file_path)

print("处理完成，已覆盖原始.png文件")
