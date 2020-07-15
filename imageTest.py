import numpy
import cv2
import os
from skimage.measure import compare_ssim
import argparse
import imutils
from glob import glob
import numpy as np

rgb_dataset_folder = "cycle/"
rgb_images_list=[image_name for image_name in os.listdir(rgb_dataset_folder) if image_name.endswith("png")]

# # load the two input images
# imageA = cv2.imread(args["first"])
# imageB = cv2.imread(args["second"])
# # convert the images to grayscale
# grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
# grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)


rgb_dataset_folder = "FLIR_Dataset/training/RGB/"
thermal_foder = "FLIR_Dataset/training/thermal_8_bit/"



testforder = "cycle/"
rgb_images_list=[image_name for image_name in os.listdir(testforder) if image_name.endswith("png")]



#MSE compare
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err



# Hash image
from PIL import Image
import imagehash
# cutoff = 5

# if hash0 - hash1 < cutoff:
#   print('images are similar')
# else:
#   print('images are not similar')


# compare code
from skimage.measure import compare_ssim
import argparse
import imutils
from glob import glob

print(rgb_images_list)
scoreArr = []
hashArr = []
mseArr = []
for i in range(0,len(rgb_images_list)):
    img = cv2.imread((rgb_dataset_folder+ rgb_images_list[i]).replace("png","jpg"))
    print((rgb_dataset_folder+ rgb_images_list[i]).replace("png","jpg"))
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    GrayImage = cv2.resize(GrayImage, (640,512))
    # print("imgtest/" + rgb_images_list[i])

    modelImage = cv2.imread(testforder+ rgb_images_list[i])
    modelImage = cv2.resize(modelImage, (640,512))
    thermalImage = cv2.imread(thermal_foder+ rgb_images_list[i].replace("png","jpeg"))
    GrayImage = np.concatenate((GrayImage,)*3, axis=-1)
    # cv2.imwrite("imgtest/" + rgb_images_list[i], GrayImage)
    # if i > 10:
    #     break 

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    # print(GrayImage.shape)
    # print(thermalImage.shape)
    # (score, diff) = compare_ssim(thermalImage, GrayImage, full=True)
    # diff = (diff * 255).astype("uint8")
    # print("SSIM gray: {}".format(score))
    print(thermalImage.shape)
    GrayImage = (GrayImage.reshape(thermalImage.shape))
    print(GrayImage.shape)
    (score, diff) = compare_ssim(thermalImage, GrayImage, full=True, multichannel=True)
    diff = (diff * 255).astype("uint8")


    hash0 = imagehash.average_hash(Image.open(testforder + rgb_images_list[i])) 
    hash1 = imagehash.average_hash(Image.open(thermal_foder+ rgb_images_list[i].replace("png","jpeg"))) 

    MseError = mse(modelImage,thermalImage)
    distance =  hash0 - hash1


    print("SSIM error: {}".format(score))
    scoreArr.append(score)

    print("MSE error: {}".format(MseError))
    mseArr.append(MseError)

    print("Hash error: {}".format(distance))
    hashArr.append(distance)




def Average(lst): 
    return sum(lst) / len(lst) 


print("_______________________________________")
print("FINAL RESULT")
print("_______________________________________")
print("Avg SSMI Score: "+ str(Average(scoreArr)))

print("Avg MSE Score: "+ str(Average(mseArr)))

print("Avg Hash Score: "+ str(Average(hashArr)))
