# -*- coding: utf-8 -*-

import sys
import scipy
import skimage.io
import os
import cv2
import random
import os
import scipy
import cv2
import keras
import datetime
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from glob import glob
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.config.experimental.set_virtual_device_configuration(physical_devices[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])



SEED=1234
np.random.seed(SEED)

def drawImg(imgArr,url):
    img=(np.array(imgArr)+1)*127.5
    cv2.imwrite(url.replace("jpg","jpeg"), img) 



class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128),
                 rgb_dataset_folder="FLIR_Dataset/training/RGB",
                 thermal_dataset_folder="FLIR_Dataset/training/thermal_8_bit"):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.rgb_dataset_folder=rgb_dataset_folder
        self.thermal_dataset_folder=thermal_dataset_folder
        self.rgb_images_list=[image_name for image_name in os.listdir(self.rgb_dataset_folder) if image_name.endswith("jpg")]

    def load_target(self,thermal_ext=".tiff"):
        rgb_imgs = []
        random_rgb_image_name_list=np.array(self.rgb_images_list).tolist()
        # print("test 123124 ")
        i = 0
        list_new  = []
        for rgb_img_name in random_rgb_image_name_list:
            # print(i)
            i = i + 1
            if i <= 0: continue
            if i > 1500: break
            # print(i)
            rgb_img_path= os.path.join(self.rgb_dataset_folder,rgb_img_name)
            rgb_img=self.imread(rgb_img_path)
            rgb_img = cv2.resize(rgb_img, self.img_res)
            rgb_imgs.append(rgb_img)
            list_new.append(rgb_img_name)
        rgb_imgs=np.array(rgb_imgs)/127.5-1
        return list_new, rgb_imgs

    def load_rgb(self,num_imgs=50,thermal_ext=".tiff"):
        rgb_imgs = []
        random_rgb_image_name_list=np.random.choice(self.rgb_images_list,size=num_imgs).tolist()
        for rgb_img_name in random_rgb_image_name_list:
            rgb_img_path= os.path.join(self.rgb_dataset_folder,rgb_img_name)
            rgb_img=self.imread(rgb_img_path)
            rgb_img = cv2.resize(rgb_img, self.img_res)
            rgb_imgs.append(rgb_img)
        rgb_imgs=np.array(rgb_imgs)/127.5-1
        return random_rgb_image_name_list, rgb_imgs

    def load_samples(self,num_imgs=5,thermal_ext=".tiff"):
        rgb_imgs,thermal_imgs=[],[]
        random_rgb_image_name_list=np.random.choice(self.rgb_images_list,size=num_imgs).tolist()
        for rgb_img_name in random_rgb_image_name_list:
            rgb_img_path= os.path.join(self.rgb_dataset_folder,rgb_img_name)
            thermal_img_path=os.path.join(self.thermal_dataset_folder,rgb_img_name.split(".")[0]+thermal_ext)
            rgb_img=self.imread(rgb_img_path)
            thermal_img=self.thermal_imread(thermal_img_path)
            rgb_img = cv2.resize(rgb_img, self.img_res)
            thermal_img = cv2.resize(thermal_img, self.img_res)
            rgb_imgs.append(rgb_img)
            thermal_imgs.append(thermal_img)
        rgb_imgs=np.array(rgb_imgs)/127.5-1
        thermal_imgs=np.array(thermal_imgs)[:,:,:,np.newaxis]/127.5-1
        return rgb_imgs, thermal_imgs

    def load_batch(self, batch_size=1, is_testing=False,thermal_ext=".tiff"):
        

        self.n_batches = int(len(self.rgb_images_list) / batch_size)

        for i in range(self.n_batches-1):
            batch = self.rgb_images_list[i*batch_size:(i+1)*batch_size]
            rgb_imgs, thermal_imgs = [], []
            for img_name in batch:
                rgb_img = self.imread(os.path.join(self.rgb_dataset_folder,img_name))
                thermal_img= self.thermal_imread(os.path.join(self.thermal_dataset_folder,img_name.split(".")[0]+thermal_ext))
                h, w, _ = rgb_img.shape
            
                rgb_img = cv2.resize(rgb_img, self.img_res)
                thermal_img = cv2.resize(thermal_img, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        rgb_img = np.fliplr(rgb_img)
                        thermal_img = np.fliplr(thermal_img)

                rgb_imgs.append(rgb_img)
                thermal_imgs.append(thermal_img)

            rgb_imgs = np.array(rgb_imgs)/127.5 - 1.
            thermal_imgs = np.array(thermal_imgs)[:,:,:,np.newaxis]/127.5 - 1.

            yield rgb_imgs, thermal_imgs


    def imread(self, path):
        try:
            img= cv2.imread(path)
            img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            return img
        except:
            print(path)
        
    def thermal_imread(self,img_path):
        thermal_img_path= img_path
        thermal_img= skimage.io.imread(thermal_img_path)
        return thermal_img


class Pix2Pix():
    def __init__(self,img_rows=512,
                img_cols=640,
                channels=3,
                 thermal_channels=1,
                 dataset_name="flir_rgbdas"
                ):
        # Input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.thermal_channels=thermal_channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.thermal_img_shape=(self.img_rows,self.img_cols,self.thermal_channels)

        # Configure data loader
        self.dataset_name = dataset_name
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_cols, self.img_rows),thermal_dataset_folder="FLIR_Dataset/training/thermal_8_bit")


        # Calculate output shape of D (PatchGAN)
        patch = 128 #from the paper
        # self.disc_patch = (patch, patch, 1)
        self.disc_patch = (128, 160, 1)
        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 128


        optimizer = Adam(0.0002,0.5,0.999)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_thermal = Input(shape=self.thermal_img_shape)
        img_rgb = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_thermal = self.generator(img_rgb)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([img_rgb,fake_thermal])

        self.combined = Model(inputs=[img_rgb, img_thermal], outputs=[valid, fake_thermal])
        self.combined.compile(loss=["mse","mae"],
                              loss_weights=[1,100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.thermal_channels, kernel_size=4, strides=1, padding='same', activation='tanh',name="thermal")(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True,strides=2):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_rgb = Input(shape=self.img_shape)
        img_thermal = Input(shape=self.thermal_img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_rgb, img_thermal])

        d1 = d_layer(combined_imgs, self.df, bn=False,strides=2) #128
        d2 = d_layer(d1, self.df*2,strides=2) #64
        # d3 = d_layer(d2, self.df*4,strides=1) #128
        # d4 = d_layer(d3, self.df*8,strides=1) #128
        # d5=  d_layer(d4, self.df*8)
        

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same',name="validity")(d2)
        print(validity.shape)

        return Model([img_rgb, img_thermal], validity)

    def train(self, epochs, batch_size=4, sample_interval=1):

        start_time = datetime.datetime.now()

        print("disc psth")
        print(self.disc_patch)
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        print("__VAILD____")
        # print(valid)
        print(valid.shape)
        print("___FAKED____")
        # print(fake)
        print(fake.shape)

        for epoch in range(epochs):
            for batch_i, (imgs_rgb, imgs_thermal) in enumerate(self.data_loader.load_batch(batch_size,thermal_ext=".jpeg")):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_thermal = self.generator.predict(imgs_rgb)

                # Train the discriminators (original images = real / generated = Fake)

                print("RGB")
                # print(imgs_rgb)
                print(imgs_rgb.shape)
                print("Thermal")
                print(imgs_thermal)
                print(imgs_thermal.shape)

                print("Thermal_FAKE")
                print(fake_thermal)
                print(fake_thermal.shape)

                d_loss_real = self.discriminator.train_on_batch([imgs_rgb, imgs_thermal], valid)
                d_loss_fake = self.discriminator.train_on_batch([imgs_rgb, fake_thermal], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # priont
                # print("Loss")
                # print(d_loss_real)
                # print(d_loss_fake)
                # print(d_loss)
                


                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_rgb, imgs_thermal], [valid, imgs_thermal])
                # print("g_loss")
                # print(g_loss)
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch,batch_i,10)

    def sample_images(self, epoch,batch_i, num_images=5):
        # target_folder='images_2020_04_13_2nd_Arch/{}/{}'.format(epoch,batch_i)
        # if not os.path.exists(target_folder):
            # os.makedirs(target_folder, exist_ok=True)
        print("______________________________________")
        print(epoch)
        print("____________________________________________")
        r, c = num_images, 3

        imgs_rgb, imgs_thermal = self.data_loader.load_samples(num_images,thermal_ext=".jpeg")
        fake_thermal = self.generator.predict(imgs_rgb)
        # np.save(target_folder+"/rgb.npy",imgs_rgb)
        # np.save(target_folder+"/original_thermal.npy",imgs_thermal)
        # np.save(target_folder+"/fake_thermal.npy",fake_thermal)
        imgs_thermal=0.5*imgs_thermal+0.5
        imgs_rgb=0.5*imgs_rgb+0.5
        fake_thermal=0.5*fake_thermal+0.5
        titles = ['Condition','Original', 'Generated']
        if epoch%2 == 0:
            self.generator.save_weights("2layer/p2pmodel_e"+str(epoch)+".h5".format(epoch,batch_i))
        self.generator.save_weights("2layer/p2pmodel.h5".format(epoch,batch_i))

"""Okay! This class needs a lot of explanation!

### The class construction

The class takes 5 arguments to initiate
- img_rows, img_cols, channels, RGB image shape
- thermal_channels. The thermal images have same height and width , But only one channel
- dataset_name="flir_rgbdas"

### Methods

The class has 4 methods

#### ```build_generator```

As the name suggests, it builds the generator. The generator is ```U-net``` structure. ```U-net``` is very much used for image segmentation. If you notice the results, you will find, the initial outputs are more like image segmentations!

```build_discriminator```

As the name suggests, it builds the discriminator of the network. The difference from the traditional GAN network is that, it doesnt give output 0 or 1 or "fake" or "not fake"! Instead, It analyzes an $NxN$ patch of the image and then says if that patch is real or fake! Thats why , authors named it ***PatchGAN***

***What is the best value for ```N```?*** It depends on trial and error. For my experiments, I found out that ```64x64``` works good!

***How to determine patch?***

In line ```23-26```

# Calculate output shape of D (PatchGAN)
patch = 64 #from the paper
self.disc_patch = (patch, patch, 1)

Also in line ```126-127```

d1 = d_layer(combined_imgs, self.df, bn=False,strides=2) #128
d2 = d_layer(d1, self.df*2,strides=2) #64

To try different sizes of patches, you need to play with different values of strides and paddings

```train```

As usual , train method means training the model!
"""


gan = Pix2Pix()
# gan.train(epochs=50,batch_size=4,sample_interval=500)
# quit()

gan.generator.load_weights('3layer/p2pmodel_e0.h5')
# gan.generator.load_weights('3layer/p2pmodel_e48.h5')


# MyDataLoader = DataLoader(dataset_name = "flir",img_res =(640,512),rgb_dataset_folder="data")
# 

MyDataLoader = DataLoader(dataset_name = "flir",img_res =(640,512))
# save ima
ListImg,testImg = MyDataLoader.load_rgb()

print(len(ListImg))
i = 0
while i < len(ListImg):
    try:
        thermal = gan.generator.predict([testImg[i:i+15]])
        for j in range(0, 15):
            print(i+j)
            drawImg(thermal[j],'2output/' + ListImg[i+j])
            # matplotlib.image.imsave('2output/' + ListImg[i+j].replace("jpg","jpeg"), thermal[j][:,:,0],cmap="gist_gray")
# 


            # matplotlib.image.imsave('2output/' + ListImg[i+j], thermal[j][:,:,0])
            # matplotlib.image.imsave('2output/' + ListImg[i+j], thermal[j][:,:,0],cmap="gist_gray")
            # Visulizing img and thermal img
            # plt.figure(figsize=(20,20))
        #     plt.subplot(131)
        #     plt.imshow(testImg[i+j][:,:,0])
        #     plt.title("RGB")

        #     plt.subplot(132)
        #     plt.imshow(thermal[i+j][:,:,0])

        #     plt.title("Generated Thermal Image")

        #     plt.savefig("output/"+ListImg[i+j])
    
        i = i +15
        # break
    except:
        break

rgb_dataset_folder = "FLIR_Dataset/training/RGB/"
thermal_foder = "FLIR_Dataset/training/thermal_8_bit/"



testforder = "2output/"
rgb_images_list=[image_name for image_name in os.listdir(testforder) if image_name.endswith("jpeg")]





#some other function 

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
    img = cv2.imread((rgb_dataset_folder+ rgb_images_list[i]).replace("jpeg","jpg"))
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    GrayImage = cv2.resize(GrayImage, (640,512))
    print("imgtest/" + rgb_images_list[i])

    modelImage = cv2.imread(testforder+ rgb_images_list[i])
    thermalImage = cv2.imread(thermal_foder+ rgb_images_list[i])
    GrayImage = np.concatenate((GrayImage,)*3, axis=-1)
    cv2.imwrite("imgtest/" + rgb_images_list[i], GrayImage)
    # if i > 10:
    #     break 

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    # print(GrayImage.shape)
    # print(thermalImage.shape)
    # (score, diff) = compare_ssim(thermalImage, GrayImage, full=True)
    # diff = (diff * 255).astype("uint8")
    # print("SSIM gray: {}".format(score))

    (score, diff) = compare_ssim(modelImage, thermalImage, full=True, multichannel=True)
    diff = (diff * 255).astype("uint8")


    hash0 = imagehash.average_hash(Image.open("imgtest/" + rgb_images_list[i])) 
    hash1 = imagehash.average_hash(Image.open(thermal_foder+ rgb_images_list[i])) 

    MseError = mse(np.reshape(GrayImage, (512,640,3)),thermalImage)
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

