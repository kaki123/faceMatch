"""
Author      : Yi-Chieh Wu, Ka Ki Fung, Jasmine Seo, Anya Wallace
Class       : HMC CS 121
Date        : 2018 Sep 13
Description : Eigenfaces
"""

import numpy as np
import pandas as pd
import cv2 as cv

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

import util

######################################################################
# globals
######################################################################

IMAGE_SHAPE = (150,150)

######################################################################
# functions
######################################################################

def plot_gallery(images, title=None, subtitles=[], n_row=3, n_col=4):
    """
    Plot array of images.

    Adapted from
    http://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html

    Parameters
    --------------------
        images       -- list of images
        title        -- title, title for entire plot
        subtitles    -- list of strings or empty list, subtitles for subimages
        n_row, n_col -- ints, number of rows and columns for plot
    """
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    if title:
        plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(comp.reshape(IMAGE_SHAPE), cmap=plt.cm.gray,
                   interpolation='nearest')
        if subtitles:
            plt.title(subtitles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.02)


######################################################################
# main
######################################################################

def main():

    # faces data
    # this resizes the img. 
    # TO-DO : Fix file path. 
    util.make_csv("data/faces", "data/faces_150x150.csv", size=(150, 150))
    X = pd.read_csv("data/faces_150x150.csv", header = None).values
    
    """
    # run PCA
    pca = PCA(n_components = 12) # eigenfaces
    pca.fit(X)
    plot_gallery(pca.components_)
    plt.savefig('results/eigenfaces.png')
    plt.show()
    

    # apply model to Jasmine's face
    galleryList = []
    jasmine = X[9,:]
    # plot every fourth k from 1 to 44 inclusive
    for k in range(0,12):
        pca = PCA(n_components = 4*k+1) 
        pca.fit(X)
        jasmine_pca = pca.transform(jasmine)
        jasmine_reconstructed = pca.inverse_transform(jasmine_pca)
        galleryList.append(jasmine_reconstructed)
    plot_gallery(galleryList)
    plt.savefig('results/jasmineseo_gallery.png')
    plt.show()    
    """
if __name__ == "__main__":
    main()