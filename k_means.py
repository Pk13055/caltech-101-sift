#!/usr/bin/env python3

from sklearn.cluster import k_means
import numpy as np
import pickle
import h5py

def main():
    dataset = pickle.load(open('complete_sifts.pkl', 'rb'))
    km = k_means(X=dataset, n_jobs=-1, n_clusters=243,
                 precompute_distances=True,
                 verbose=True)
    pickle.dump(km, open('km_data.pkl', 'wb'))


if __name__ == "__main__":
    main()
