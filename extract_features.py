from sklearn.preprocessing import LabelEncoder
from keras.applications import resnet_v2
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from imutils import paths
import numpy as np
import pickle
import os,argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default="data/", help='Root directory of data')
args = parser.parse_args()

print("[INFO] loading network...")
model = resnet_v2(weights="imagenet", include_top=False)
le = None

for split in ("data", "dummy"):
	print("[INFO] processing '{} split'...".format(split))
	p = os.path.sep.join([args.root, split])
	imagePaths = list(paths.list_images(p))

	labels = [p.split(os.path.sep)[-2] for p in imagePaths]

	if le is None:
		le = LabelEncoder()
		le.fit(labels)

	csvPath = os.path.sep.join(["resnet_v2_output",
		"{}.csv".format(split)])
	csv = open(csvPath, "w")

	for (b, i) in enumerate(range(0, len(imagePaths), 32)):
		print("[INFO] processing batch {}/{}".format(b + 1,
			int(np.ceil(len(imagePaths) / float(32)))))
		batchPaths = imagePaths[i:i + 32]
		batchLabels = le.transform(labels[i:i + 32])
		batchImages = []

		for imagePath in batchPaths:

			image = load_img(imagePath, target_size=(224, 224))
			image = img_to_array(image)
			image = np.expand_dims(image, axis=0)
			image = imagenet_utils.preprocess_input(image)
			batchImages.append(image)
		batchImages = np.vstack(batchImages)
		features = model.predict(batchImages, batch_size=32)
		features = features.reshape((features.shape[0], 7*7*2048))
        
		for (label, vec) in zip(batchLabels, features):
			vec = ",".join([str(v) for v in vec])
			csv.write("{},{}\n".format(label, vec))
	csv.close()
f = open("resnet_v2_output/le.cpickle", "wb")
f.write(pickle.dumps(le))
f.close()

