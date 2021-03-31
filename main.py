import os
import sys 
import io 

from PIL import Image
import numpy as np

import time
import streamlit as st 

def initialize_K_centroids(X, K):
    m = len(X)
    return X[np.random.choice(m, K, replace=False),:]

def find_closest_centroids(X, centroids):
    m = len(X)
    c = np.zeros(m)
    for i in range(m):
        distances = np.linalg.norm(X[i]-centroids, axis=1)

        c[i] = np.argmin(distances)

    return c

def compute_means(X, idx, K):
    _, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        examples = X[np.where(idx==k)]
        mean = [np.mean(column) for column in examples.T]
        centroids[k] = mean
    return centroids

def find_k_means(X, K, max_iters=10):
    centroids = initialize_K_centroids(X, K)
    previous_centroids = centroids

    for _ in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_means(X, idx, K)
        if (centroids == previous_centroids).all():
            return centroids
        else:
            previous_centroids = centroids
    return centroids, idx
def compute_mse(X, X_reconstructed):
    return np.sum((X-X_reconstructed)**2) / len(X) 
try:
    image_path = sys.argv[1]
    assert os.path.isfile(image_path)
except (IndexError, AssertionError):
    print('Please specify an image')

st.header('COMPRESS IMAGE USING K_MEAN ALGORITHM')
upload_file = None
upload_file = st.file_uploader('Choose image')


if upload_file is not None:
    image_byte = upload_file.read()
    image = Image.open(io.BytesIO(image_byte))
    #st.image(image)
    image = np.asarray(image)/255
    w, h, d = image.shape

    st.write('Compressing...')
    #print('Image found with width: {}, height: {}, deptSh: {}'.format(w, h, d))

    tic = time.time()

    X = image.reshape((w*h, d))
    K = 10

    colors, _ = find_k_means(X, K, max_iters=20)

    idx = find_closest_centroids(X, colors)

    idx = np.array(idx, dtype=np.uint8)
    X_reconstructed = np.array(colors[idx, :]*255, dtype=np.uint8).reshape((w, h, d))
    compressed_image = Image.fromarray(X_reconstructed)
    
    toc = time.time()
    st.write('Done! Time taken: ', toc-tic)
    st.image([image,compressed_image])

    st.write('Compute MSE...')
    mse = compute_mse(X, X_reconstructed.reshape((w*h,d)))
    st.write('MSE=', mse)
    #st.image(compressed_image)
    compressed_image.save('out.png')
