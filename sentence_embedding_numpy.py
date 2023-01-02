from os.path import exists
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


def embedding():
    model_path = "./all-mpnet-base-v2"

    if not exists(model_path):
        print('downloading model')
        # Load model
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
        # Save model
        model.save(model_path)
        
    # Load backbone
    model = SentenceTransformer(model_path, device='cuda')
    # data = pd.read_csv('train.csv')
    data = open('sentences.csv', 'r', encoding='UTF-8')
    print(data.readline())
    sentences = []
    for line in data.readlines():
        sentences.append(line[1:-3])
    print(len(sentences))
    embeddings = model.encode(sentences, device='cuda', show_progress_bar=True)

    np.save('sentence_embeddings.npy', embeddings)



if __name__ == '__main__':
    embedding()
