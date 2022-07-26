import numpy as np
from time import time
import argparse
import os
import argparse
from embedding import Embedding


parser = argparse.ArgumentParser(description='ArcFace feature extraction')
parser.add_argument('--batch_size', default=90, type=int, help='Batch size')
parser.add_argument('--dataset', required=True, type=str, help='Dataset name')
parser.add_argument('--landmarks', required=True, type=str, help='Path to the file with paths and landmarks')
parser.add_argument('--bb_scale', default=0.5, type=float, help='Bounding box scale')
parser.add_argument('--cuda', default=-1, type=int, help='Cuda device')
parser.add_argument('--model', default='MobileFaceNet', choices=['MS1MV2-ResNet100-Arcface', 'MobileFaceNet'], help='path to load model.')

args = parser.parse_args()

dataset_name = args.dataset
bb_scale = args.bb_scale

os.makedirs(f'results/{dataset_name}_{bb_scale}/', exist_ok=True)


def save_results(features, norms):
    np.save(f"results/{dataset_name}_{bb_scale}/features.npy", features)
    np.save(f"results/{dataset_name}_{bb_scale}/norms.npy", norms)


def batches(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def feature_extraction(model: Embedding, landmarks=None):
    
    n_imgs = len(landmarks)

    n_features = 128 if 'MobileFaceNet' in model.name else 512
    
    features = np.empty((n_imgs, n_features))
    norms = np.empty((n_imgs))
    start, finished = 0, 0
    
    # faces = np.concatenate((faces, landmarks[:, 1:]), axis=1)
    
    save_each = 100

    for batch_iter, batch in enumerate(batches(landmarks, args.batch_size)):
        run = time()
        size = batch.shape[0]
        
        # preprocess whole batch
        images_batch = np.empty((size, 3, 112, 112))
        for i, image in enumerate(batch):
            landmarks = image[1:].astype(int)
            path = f"images/{dataset_name}/{image[0]}"
            images_batch[i] = model.preprocess(path, landmarks=landmarks)
        
        # extract features
        f = model.extract(images_batch)
        
        start = finished
        finished += batch.shape[0]

        # normalize features to unit length
        norm = np.linalg.norm(f, axis=1, keepdims=True)
        f = f / norm
        
        features[start:finished] = f
        norms[start:finished] = norm[:, 0]

        if batch_iter % save_each == 0:
            save_results(features, norms)
        print(f"-> processed {finished}/{n_imgs} images in {time() - run:.4f} seconds")

    save_results(features, norms)

    return features, norms

    
if __name__ == '__main__':
    landmarks = np.genfromtxt(args.landmarks, dtype=str, delimiter=',')
    model = Embedding(args.model, args.cuda, args.batch_size)
    feature_extraction(model, landmarks)