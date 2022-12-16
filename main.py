import os
import shutil

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm


# Fungsi untuk ekstraksi fitur dari gambar

def extract_features(directory):
    model = InceptionV3(weights="imagenet", include_top=False)
    features = []
    img_names = []
    for i in tqdm(directory):
        fname = 'dataset' + '/' + i
        img = image.load_img(fname, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        flat = feature.flatten()
        features.append(flat)
        img_names.append(i)
    return features, img_names


# direktori gambar

img_path = os.listdir('dataset')
img_features, img_names = extract_features(img_path)

# Melakukan clustering dengan K-Means
k = 2
cluster = KMeans(k)
cluster.fit(img_features)

# Menyimpan hasil dataset
image_cluster = pd.DataFrame(img_names, columns=['image'])
image_cluster['cluster_id'] = cluster.labels_

# print dataframe
print(image_cluster)

# Menyimpan copy hasil dataset ke dalam folder
os.mkdir('cluster_0')
os.mkdir('cluster_1')
for i in range(len(image_cluster)):
    if image_cluster['cluster_id'][i] == 0:
        shutil.copy('dataset/' + image_cluster['image'][i], 'cluster_0')
    else:
        shutil.copy('dataset/' + image_cluster['image'][i], 'cluster_1')
