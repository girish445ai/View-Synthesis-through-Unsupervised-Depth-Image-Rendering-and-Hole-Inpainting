# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# depth_npy = np.load('/data2/AdaMPI-Inpaint/warpback/data_depth_2/train/source/1.npy')
# print(depth_npy.shape)
# depth_png = Image.fromarray(depth_npy.squeeze())

# plt.imshow(depth_png, cmap="viridis")
# # plt.axis('off')  
# # depth_png.save('1_depth.png')
# output_path = os.path.join('/data2/pconv_files_new/sample_images/', '1_depth.png')
# plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  
# # plt.close()  

from PIL import Image
import numpy as np
import os

depth_npy = np.load('/data2/AdaMPI-Inpaint/warpback/data_depth_2/train/source/5.npy')
print(depth_npy.shape)
depth_npy = depth_npy.squeeze().astype(np.uint8)

depth_png = Image.fromarray(depth_npy.squeeze(),mode='L')

# Save the image as a PNG file directly using Pillow
output_path = os.path.join('/data2/pconv_files_new/sample_images/', '5_stereo_depth.png')
depth_png.save(output_path)

# Optionally, display the image using Pillow's Image.show() method
# depth_png.show()
