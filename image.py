import numpy as np
import matplotlib.pyplot as plt
rgb_imgs = np.load("rgb.npy")
gen_thermal_imgs=np.load("fake_thermal.npy")
thermal_imgs=np.load("original_thermal.npy")

plt.figure(figsize=(10,10))
plt.subplot(131)
plt.imshow(rgb_imgs[0][:,:,0])
plt.title("RGB")
plt.subplot(132)
plt.imshow(thermal_imgs[0][:,:,0],cmap="hot")
plt.title("Original Thermal Image")
plt.subplot(133)
plt.imshow(gen_thermal_imgs[0][:,:,0],cmap="hot")
plt.title("Generated Thermal Image")

# plt.title("Generated Thermal Image")
plt.savefig("pix2pix_results.png")
