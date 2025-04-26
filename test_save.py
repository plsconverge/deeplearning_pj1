import mynn as nn
import numpy as np
from struct import unpack
import gzip
import os
import json
import pickle

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    # test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

model_new = nn.models.Model_MLP()
# model_new = nn.models.Model_CNN()
model_new.load_model(r'.\best_models\best_model.pkl')

logits_new = model_new(test_imgs)
score_new = nn.metric.accuracy(logits_new, test_labs)
# print(nn.metric.accuracy(logits, test_labs))
print(f'Accuracy: {score_new: .4f}')

save_dir = r'.\saved_models'
score_path = os.path.join(save_dir, 'model_score_MLP.json')
# score_path = os.path.join(save_dir, 'model_score_CNN.json')
model_path = os.path.join(save_dir, 'best_model_MLP.pkl')
# model_path = os.path.join(save_dir, 'best_model_CNN.pkl')
if not os.path.exists(score_path):
    with open(score_path, 'w', encoding='utf-8') as f:
        json.dump({'score': score_new}, f)
    model_new.save_model(model_path)
    # information
    print('No best model has been saved before.')
    print(f'New model saved. Accuracy Now: {score_new: .4f}')
else:
    with open(score_path, 'r', encoding='utf-8') as f:
        score_old = json.load(f)
        score_old = score_old['score']
    if score_new <= score_old:
        print(f'New model failed. Accuracy Remained: {score_old: .4f}')
    else:
        with open(score_path, 'w', encoding='utf-8') as f:
            json.dump({'score': score_new}, f)
        # with open(model_path, 'wb') as f:
        #     pickle.dump(model_new, f)
        model_new.save_model(model_path)
        print(f'New model saved. Accuracy Updated: {score_old: .4f} --> {score_new: .4f}')
