import os
import numpy as np
import random
from PIL import Image
import skimage.io as io
import sys
from unetModel import *
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(0)


def saveImage(dir,results,ids):
	for i,e in enumerate(results):
		img = e[:,:,0]
		true_label = read_image(ids[i],'data/original_labels/')
		true_label = true_label[:,:,0]
		pred_label = img
		# Uncomment to see IOU
		# pred_label = img >= 0.5
		# print(true_label.shape,pred_label.shape)
		# IOU = (true_label == pred_label).sum()/(256*256)
		# print(ids[i],IOU)
		io.imsave(os.path.join(dir,"%s_predict.png"%ids[i]),(pred_label*255).astype('uint8'))

def read_image(id,dir):
	img = Image.open(dir + id)
	img = img.resize((256,256), Image.ANTIALIAS)

	x = np.asarray(img,dtype=np.float32)
	return np.reshape(x/255,[256,256,1])

		
def get_test_data(ids,img_dir):
	for id in ids:
		img = read_image(id,img_dir)
		yield np.reshape(img,[1,256,256,1])


def predict(net) :

	img_dir = str(sys.argv[1])

	# img_list = (f[:-1] for f in os.listdir(img_dir))

	img_list = [f for f in os.listdir(img_dir) if re.match(r'.*\.png', f)]
	ids = img_list

	results = net.predict_generator(get_test_data(ids,img_dir),len(ids),verbose=1)
	# print(results.shape,results)
	npresult = np.asarray(results)  

	saveImage("data/result",results,ids)


if __name__ == '__main__' :

	net = load_model('unet_1223pm.hdf5')

	predict(net)