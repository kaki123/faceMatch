"""
Author      : Yi-Chieh Wu, Ka Ki Fung, Jasmine Seo, Anya Wallace
Class       : HMC CS 121
Date        : 2018 Sep 13
Description : Utility functions
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import cv2 as cv
from sklearn import metrics
import matplotlib.pyplot as plt

######################################################################
# functions
######################################################################

def get_files(path):
    """Get full pathname of all (non-hidden) files in path directory.

    Parameters
    --------------------
        path   -- directory path, str

    Return
    --------------------
        fns    -- list of filenames, each of type str
    """

    fns = []
    for fn in os.listdir(path):
        full_fn = os.path.join(path, fn)
        if os.path.isdir(full_fn):
            continue
        if full_fn.startswith('.'):
            continue
        fns.append(full_fn)
    return fns


def read_csv(fn, label_col=-1, header=None):
    """Load csv file.

    Parameters
    --------------------
        fn          -- csv filename, str
        label_col   -- label column, int
        header      -- header row, int

    Return
    --------------------
        X           -- features, numpy array of shape (n,d)
        y           -- labels, numpy array of shape (n,)
    """

    # use pandas to read csv file
    df = pd.read_csv(fn, header=header)

    # get label column
    if label_col < 0:
        label_col = len(df.columns) + label_col

    # get features
    # iloc : integer-location based indexing of a dataframe
    # [:,cols] gets all rows (:) but only the selected columns (cols)
    # .values converts the panda dataframe into a numpy array
    cols = [col for col in range(len(df.columns)) if col != label_col]
    X = df.iloc[:,cols].values

    # get label
    y = df.iloc[:,label_col].values

    print("Total dataset size: {} \n".format(len(X)))
    return X, y


def print_summary(X, y_true, y_pred):
    """Print summary of clustering.

    Parameters
    --------------------
        X       -- features, numpy array of shape (n,d)
        y_true  -- true labels, numpy array of shape (n,)
        y_pred  -- predicted labels, numpy array of shape (n,)

    Return
    --------------------
        mapping -- mapping of predicted label to true label, dict
    """

    # count true for each predicted
    set_predicted = set(y_pred)
    counter = {}
    for p in set_predicted:
        counter[p] = Counter()
    for t,p in zip(y_true, y_pred):
        counter[p][t] += 1

    # compute mode, mode freq, and points for each predicted
    modes = []
    for p in set_predicted:
        t,ct = counter[p].most_common(1)[0]
        tot = sum(counter[p].values())
        modes.append((p, t, ct, tot))

    # print information
    print("Cluster Summary")
    print("-"*20)
    for mode in modes:
        print("\tcluster {} -- mode: {}, count: {}, total: {}".format(*mode))

    # print scores
    h, c, v = metrics.homogeneity_completeness_v_measure(y_true, y_pred)
    s = metrics.silhouette_score(X, y_pred)
    print("homogeneity score: {}".format(h))
    print("completeness score: {}".format(c))
    print("v-measure score: {}".format(v))
    print("silhouette score: {}".format(s))
    print("\n")

    # get mapping
    mapping = {}
    for p, t, ct, tot in modes:
        mapping[p] = t
    return mapping


def plot(X, y_true, y_pred):
    """Plot data points (first two features) with true and predicted labels.

    Parameters
    --------------------
        X -- data, numpy array of shape (n,d)
        y_true -- true labels, numpy array of shape (n,)
        y_pred -- predicted labels, numpy array of shape (n,)
    """
    f, (ax1, ax2) = plt.subplots(2,1)

    # plot points using true labels
    # first remap labels to integers to pass to scatter as colors
    d = dict([(y,i) for i,y in enumerate(sorted(set(y_true)))])
    colors = [d[y] for y in y_true]
    ax1.scatter(X[:,0], X[:,1], c=colors)
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.set_title("True Labels")

    # plot points using predicted labels
    d = dict([(y,i) for i,y in enumerate(sorted(set(y_pred)))])
    colors = [d[y] for y in y_pred]
    ax2.scatter(X[:,0], X[:,1], c=colors)
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.set_title("Predicted Labels")

    # adjust layout
    plt.tight_layout()

    plt.show()


def make_csv(path, csv, size=None, label=None):
    """Make csv file from images in directory path.
    each row represents an image's
    1) width
    2) height
    3) luminosity values in row-major order
    4) label

    Parameters
    --------------------
        path        -- directory path, str
        csv         -- csv filename, str
        size        -- dimensions to resize, tuple of ints
        label       -- function to obtain label from filename, lambda function
    """

    # get sorted files
    fns = get_files(path)
    fns.sort()
    
    save = []

    # add image data into each row of csv file
    for fn in fns:
        img = cv.imread(fn)     # current image
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # resize image
        if size is not None:
            newImg = cv.resize(img,size)
            # cv.imwrite(os.path.join(path, fn), newImg)
        else: 
            newImg = img

        # add image luminosity values
        arrayImg = np.asarray(newImg)
        flatImg = arrayImg.flatten().tolist()

        # add image label if it exists
        if label is not None:
            flatImg.append(label(fn))

        save.append(flatImg)

    # output file
    save = np.asarray(save)
    np.savetxt(csv, save, fmt='%i', delimiter=',')
            

