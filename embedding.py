import cv2
import numpy as np
import mxnet as mx
from skimage import transform as trans


class Embedding:
	def __init__(self, name, cuda=0, batch_size=1):
		print('loading', name)
		self.name = name
		path = f'model/{name}/model'
		context = mx.cpu() if cuda == -1 else mx.gpu(cuda)
		sym, arg_params, aux_params = mx.model.load_checkpoint(path, 0)
		all_layers = sym.get_internals()
		sym = all_layers['fc1_output']
		model = mx.mod.Module(symbol=sym, context=context, label_names=None)
		model.bind(for_training=False, data_shapes=[('data', (batch_size, 3, 112, 112))])
		model.set_params(arg_params, aux_params)
		self.model = model
		src = np.array([
			[30.2946, 51.6963],
			[65.5318, 51.5014],
			[48.0252, 71.7366],
			[33.5493, 92.3655],
			[62.7299, 92.2041]], dtype=np.float32)
		src[:,0] += 8.0
		self.src = src


	def preprocess(self, image_name, landmarks=None, scale=0.5):
		## fit face to image with size 112px using landmarks
		landmarks = landmarks.reshape(5,2)
		img = cv2.imread(image_name)
		tform = trans.SimilarityTransform()
		tform.estimate(landmarks, self.src)
		M = tform.params[0:2, :]
		img = cv2.warpAffine(img, M, (112, 112))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# img_flip = np.fliplr(img)
		# img_flip = np.transpose(img_flip,(2,0,1))
		# cv2.imshow('img', img)
		# cv2.waitKey(0)
		img = np.transpose(img, (2, 0, 1)) # 3 x 112 x 112, RGB
		return np.array(img)


	def extract(self, images_batch):
		data = mx.nd.array(images_batch)
		data_batch = mx.io.DataBatch(data=(data,))
		self.model.forward(data_batch, is_train=False)
		features = self.model.get_outputs()[0].asnumpy()
		return features.astype(np.float32)