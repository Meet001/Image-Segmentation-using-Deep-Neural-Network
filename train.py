import os
import sys
import random
import numpy as np
from PIL import Image
from unetModel import *
from keras import backend as keras
from keras.callbacks import ModelCheckpoint


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(int(sys.argv[1]))



def read(id,prefix,dir):

	x = np.asarray(Image.open(dir + prefix + id),dtype=np.float32)
	x = (x-x.min())/(x.max()-x.min())
	return np.reshape(x,[256,256,1])


def get_data(ids, img_dir, lbl_dir,bs):
	i = 0
	img = []
	mask = []
	random.shuffle(ids)
	for id in ids:
		img.append(read(id, 'image_original_' ,img_dir))
		mask.append(read(id, '_groundtruth_(1)_image_' ,lbl_dir))
		i = (i + 1)%bs
		if  i == 0 :
			yield (np.array(img),np.array(mask))
			img = [] 
			mask = []
			i = 0


def get_test_data(ids,img_dir):
	img = None
	for id in ids:
		img = read(id,'image_original_',img_dir)
		yield np.reshape(img,[1,256,256,1])
		img = None

def train(net,n_epochs=50,batch_size=2,train_percent=0.9,save_ckpt=True) :

	img_dir = 'data/train_images/' 
	lbl_dir = 'data/train_labels/' 

	img_list = [f[15:] for f in os.listdir(img_dir)]
	ids = img_list
	bs = 2
	train_ids = ids[:int(train_percent*len(ids))]
	test_ids  = ids[int(train_percent*len(ids)):]

	model_checkpoint = ModelCheckpoint('unet_modified_dice.hdf5', monitor='loss',verbose=1)
	net.fit_generator(get_data(train_ids,img_dir,lbl_dir,bs),steps_per_epoch=50,epochs=1,callbacks=[model_checkpoint])
	
if __name__ == '__main__' :
	# net = unet()
	net = load_model('unet_modified_dice.hdf5')
	train(net)