# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import shift, rotate
from tqdm import tqdm

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        # train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# # augmentation
# indices_aug = np.random.choice(len(train_imgs), int(len(train_imgs) * 0.35), replace=False)
# augment_imgs = []
# augment_labs = []
#
# print('Doing data augmentation...')
# for ind in tqdm(indices_aug):
#         img_aug = train_imgs[ind].reshape(28, 28)
#         img_aug = rotate(img_aug, np.random.uniform(-90, 90), reshape=False, mode='nearest')
#         x_trans = np.random.randint(-3, 3)
#         y_trans = np.random.randint(-3, 3)
#         if x_trans or y_trans:
#                 img_aug = shift(img_aug, (x_trans, y_trans), mode='nearest')
#         img_aug = np.clip(img_aug, 0, 255)
#         augment_imgs.append(img_aug.reshape(-1))
#         # augment_imgs.append(img_aug.reshape(1, 28, 28))
#         augment_labs.append(train_labs[ind])
#
# augment_imgs = np.array(augment_imgs)
# augment_labs = np.array(augment_labs)
#
# train_imgs = np.concatenate((train_imgs, augment_imgs), axis=0)
# train_labs = np.concatenate((train_labs, augment_labs), axis=0)

# choose 10000 samples from train set as validation set.
# idx = np.random.permutation(np.arange(num))
idx = np.random.permutation(np.arange(train_imgs.shape[0]))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# model
linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 200, 10], 'ReLU', [1e-4, 1e-4, 1e-4])
# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 100, 100, 10], 'ReLU', [1e-4, 1e-4, 1e-4])
# cnn_model = nn.models.Model_CNN(3, [(1, 16, 5), 2, (16, 32, 5), 2, (32, 64, 4), 10])
# cnn_model = nn.models.Model_CNN(3, [(1, 32, 5), 2, (32, 64, 5), 2, (64, 128, 4), 10])
# cnn_model = nn.models.Model_CNN_withDropout(f1, 32, 5), 2, (32, 64, 5), 2, (64, 128, 4), 64, 10])

# SGD optimizer
optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
# optimizer = nn.optimizer.SGD(init_lr=0.003, model=cnn_model)
# Momentum optimizer
# optimizer = nn.optimizer.MomentGD(init_lr=0.005, model=linear_model, mu=0.9)
# optimizer = nn.optimizer.MomentGD(init_lr=0.003, model=cnn_model, mu=0.9)

# scheduler
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)

# loss function
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
# loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max()+1)

# runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
# runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
# runner = nn.runner.Runner_EarlyStopping(cnn_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
runner = nn.runner.Runner_EarlyStopping(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

# runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=200, save_dir=r'./best_models')
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=200, save_dir=r'./best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()