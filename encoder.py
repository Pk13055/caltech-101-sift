#!/usr/bin/env python3
"""
    encode the current dataset into the bag of words
    feature vector
"""
import pickle
from multiprocessing import Pool, Process
from sys import argv as rd
import os

import cv2
import numpy as np

centroids = pickle.load(open('dataset/descriptors.pkl', 'rb'))
print(centroids.shape)


def get_inpv(descriptor):
    '''
        @description helper function to perfom async tasks
        calculates the actual nx1 vector for each image
        @param descriptor -> list: list of sift features for a given image
            |descriptor| = a * 128, variable a
        @return inp_vec -> np.array: bag of words repr for the given image
            |inp_vec| = (1, n), n => number of centroids
    '''
    counts = {}
    bag_size = centroids.shape[0]
    distances = [np.argmax(np.sqrt(np.sum((_ - centroids ) ** 2 , axis=1))) for _ in descriptor]
    [counts.update({_ : distances.count(_)}) for _ in range(bag_size)]
    inp_vec = np.zeros(bag_size)
    for _ in counts: inp_vec[_] = counts[_]
    return inp_vec


def normalizeInput(descriptors):
    '''
        @param descriptors -> list: each images descriptors.
        @param centroids -> np.array: n x 128
            Finally dimensionality of the input will be n x 1
        @return input_array -> np.array: n x 1
    '''
    pool = Pool(4)
    bag_size = centroids.shape[0]
    input_array = pool.map_async(get_inpv, descriptors).get()
    return np.array(input_array) / bag_size

def getCategory(category):
    '''
        @description process based category BOW representation
        @param category -> str: list containing the descriptors
        per training example
    '''
    getfull = lambda x: os.path.join(os.getcwd(), 'dataset', 'sifts', x)
    data = pickle.load(open(getfull('%s_sifts.pkl' % category), 'rb'))
    pool = Pool(4)
    data_inps = normalizeInput(data)
    print("Final dataset : ", data_inps.shape)
    pickle.dump(data_inps, open('BOW_repr_%s.pkl' % category, 'wb'))


def main():
    filenames = rd[1:]
    processess = [Process(target=getCategory, args=(_,), name=_) for _ in filenames]
    [_.start() for _ in processess]
    [_.join() for _ in processess]

if __name__ == '__main__':
    main()
