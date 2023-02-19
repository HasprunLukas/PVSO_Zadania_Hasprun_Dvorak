import time

from ximea import xiapi
import cv2
import os
import numpy as np
### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
def save_image(path, current_photo_num):
    cv2.imwrite(path + "photo" + str(current_photo_num) + ".jpg", image)
    print("photo" + str(current_photo_num) + ".jpg created")

def create_mosaic(images):
    im_tile = concat_tile([[images[0], images[1]],
                               [images[2], images[-1]]])
    cv2.imwrite("mosaic.jpg", im_tile)
    print("mosaic.jpg created")
    cv2.imshow("im_tile", im_tile)

#create instance for first connected camera
cam = xiapi.Camera()



#start communication
#to open specific device, use:
#cam.open_device_by_SN('41305651')
#(open by serial number)
print('Opening first camera...')
cam.open_device()

#settings
cam.set_exposure(10000)
cam.set_param("imgdataformat","XI_RGB32")
cam.set_param("auto_wb",1)

print('Exposure was set to %i us' %cam.get_exposure())

#create instance of Image to store image data and metadata
img = xiapi.Image()

#start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()

key = cv2.waitKey()
max_num_of_photos = 4
current_photo_num = 1
path = 'images/'
images_list = []
while key != ord('q'):
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image,(240,240))
    cv2.imshow("test", image)
    if key == 32 and current_photo_num <= 4:
        save_image(path, current_photo_num)
        images_list.append(image)
        current_photo_num += 1
        if current_photo_num == 5:
            create_mosaic(images_list)
        # time.sleep(1)
    if key == ord('1'):
        kernel = np.ones((3,3), np.float32)/9
        cv2.filter2D()
    key = cv2.waitKey('mosaic.jpg',-1,kernel)



# for i in range(10):
#     #get data and pass them from camera to img
#     cam.get_image(img)
#     image = img.get_image_data_numpy()
#     cv2.imshow("test", image)
#     cv2.waitKey()
#     #get raw data from camera
#     #for Python2.x function returns string
#     #for Python3.x function returns bytes
#     data_raw = img.get_image_data_raw()
#
#     #transform data to list
#     data = list(data_raw)
#
#     #print image data and metadata
#     print('Image number: ' + str(i))
#     print('Image width (pixels):  ' + str(img.width))
#     print('Image height (pixels): ' + str(img.height))
#     print('First 10 pixels: ' + str(data[:10]))
#     print('\n')

#stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

#stop communication
cam.close_device()

print('Done.')