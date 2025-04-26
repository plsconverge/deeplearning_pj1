<<<<<<< HEAD
# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# model = nn.models.Model_MLP()
model = nn.models.Model_CNN()
model.load_model(r'.\saved_models\best_model_CNN.pkl')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        # test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

# logits = model(test_imgs)

# mats = []
# mats.append(model.layers[0].params['W'])
# mats.append(model.layers[2].params['W'])

np.random.seed(42)
mats=[]
# convolution layer 1
# for i in range(4):
#         ind1 = np.random.randint(0, 31)
#         mats.append(model.layers[0].params['W'][ind1, 0])
# convolution layer 2
# for i in range(4):
#         ind1 = np.random.randint(0, 63)
#         ind2 = np.random.randint(0, 31)
#         mats.append(model.layers[3].params['W'][ind1, ind2])
# convolution layer 3
for i in range(4):
        ind1 = np.random.randint(0, 127)
        ind2 = np.random.randint(0, 63)
        mats.append(model.layers[6].params['W'][ind1, ind2])

# _, axes = plt.subplots(30, 20)
# _.set_tight_layout(1)
# axes = axes.reshape(-1)
# for i in range(600):
#         axes[i].matshow(mats[0].T[i].reshape(28,28))
#         axes[i].set_xticks([])
#         axes[i].set_yticks([])

# plt.figure()
# plt.matshow(mats[1])
# plt.xticks([])
# plt.yticks([])
# plt.show()

fig, axes = plt.subplots(2, 2)
axes = axes.reshape(-1)

for i in range(4):
        im = axes[i].imshow(mats[i])
        fig.colorbar(im, ax=axes[i])
        plt.axis('off')

plt.tight_layout()
plt.show()
=======
# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# model = nn.models.Model_MLP()
model = nn.models.Model_CNN()
model.load_model(r'.\saved_models\best_model_CNN.pkl')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        # test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

# logits = model(test_imgs)

# mats = []
# mats.append(model.layers[0].params['W'])
# mats.append(model.layers[2].params['W'])

np.random.seed(42)
mats=[]
# convolution layer 1
# for i in range(4):
#         ind1 = np.random.randint(0, 31)
#         mats.append(model.layers[0].params['W'][ind1, 0])
# convolution layer 2
# for i in range(4):
#         ind1 = np.random.randint(0, 63)
#         ind2 = np.random.randint(0, 31)
#         mats.append(model.layers[3].params['W'][ind1, ind2])
# convolution layer 3
for i in range(4):
        ind1 = np.random.randint(0, 127)
        ind2 = np.random.randint(0, 63)
        mats.append(model.layers[6].params['W'][ind1, ind2])

# _, axes = plt.subplots(30, 20)
# _.set_tight_layout(1)
# axes = axes.reshape(-1)
# for i in range(600):
#         axes[i].matshow(mats[0].T[i].reshape(28,28))
#         axes[i].set_xticks([])
#         axes[i].set_yticks([])

# plt.figure()
# plt.matshow(mats[1])
# plt.xticks([])
# plt.yticks([])
# plt.show()

fig, axes = plt.subplots(2, 2)
axes = axes.reshape(-1)

for i in range(4):
        im = axes[i].imshow(mats[i])
        fig.colorbar(im, ax=axes[i])
        plt.axis('off')

plt.tight_layout()
plt.show()
>>>>>>> 1a49665a8b316f1b9059ecc40c923cb579fa7760
