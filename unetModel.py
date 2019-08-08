from keras.models import *
from keras.layers import *

def conv_Relu(num_filters,layer_input):
	first_conv_out = Conv2D(num_filters , 3, activation = 'relu', padding = 'same')(layer_input)
	second_conv_out = Conv2D(num_filters, 3, activation = 'relu', padding = 'same')(first_conv_out)
  
	return second_conv_out


def scaleUp_merge_conv(num_filters,conc_half,layer_input):

	upscale = UpSampling2D(size = (2,2))(layer_input)
	scale_conv_out = Conv2D(num_filters, 2, activation = 'relu', padding = 'same')(upscale)
	mergeLayer = concatenate([conc_half,upscale], axis = 3)

	up_conv = conv_Relu(num_filters,mergeLayer)

	return up_conv


def unet(input_size = (256,256,1)):

	inputs = Input(input_size)

	down_one = conv_Relu(64,inputs)
	max_pool_one = MaxPooling2D()(down_one)

	down_two = conv_Relu(128,max_pool_one)
	max_pool_two = MaxPooling2D()(down_two)

	down_three = conv_Relu(256,max_pool_two)
	max_pool_three = MaxPooling2D()(down_three)

	down_four = conv_Relu(512,max_pool_three)
	threshold_one = Dropout(0.5)(down_four)
	max_pool_four = MaxPooling2D()(threshold_one)

	side = conv_Relu(1024,max_pool_four)
	threshold_two = Dropout(0.5)(side)

	up_four = scaleUp_merge_conv(512,threshold_one,threshold_two)

	up_three = scaleUp_merge_conv(256,down_three,up_four)

	up_two = scaleUp_merge_conv(128,down_two,up_three)

	up_one = scaleUp_merge_conv(64,down_one,up_two)

	final_classifier = Conv2D(2, 3, activation = 'relu', padding = 'same')  ( up_one )
  
	net_output = Conv2D(1, 1, activation = 'sigmoid') ( final_classifier )

	model = Model(input = inputs, output = net_output)
 
	model.compile(loss = dice_coef_loss, optimizer = keras.optimisers.Adam(lr = 1e-4), metrics = ['accuracy'] )   

	
	
	return model


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
    

















