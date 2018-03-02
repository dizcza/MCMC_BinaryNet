#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2016 Paul Brodersen <paulbrodersen+entropy_estimators@gmail.com>

# Author: Paul Brodersen <paulbrodersen+entropy_estimators@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


def get_h(x, k=1):
    """
    Estimates the entropy H of a random variable x (in bits) based on
    the kth-nearest neighbour distances between point samples.

    @reference:
    Kozachenko, L., & Leonenko, N. (1987). Sample estimate of the entropy of a random vector.
    Problemy Peredachi Informatsii, 23(2), 9–16.

    Arguments:
    ----------
    x: (n, d) ndarray
        n samples from a d-dimensional multivariate distribution

    k: int (default 1)
        kth nearest neighbour to use in density estimate;
        imposes smoothness on the underlying probability distribution

    Returns:
    --------
    h: float
        entropy H(X)
    """

    n, d = x.shape
    kdtree = cKDTree(x)

    # query all points -- k+1 as query point also in initial set
    # distances, idx = kdtree.query(x, k + 1, eps=0, p=norm)
    distances, idx = kdtree.query(x, k + 1, eps=0, p=np.inf)
    distances = distances[:, -1]

    # enforce non-zero distances
    distances = distances[distances > 0]
    # np.clip(distances, a_min=0, a_max=None, out=distances)

    sum_log_dist = np.sum(np.log2(2 * distances))  # where did the 2 come from? radius -> diameter
    h = -digamma(k) + digamma(n) + (d / float(n)) * sum_log_dist

    return h


def get_mi(x, y, k=1, estimator='ksg'):
    """
    Estimates the mutual information (in bits) between two point clouds, x and y,
    in a D-dimensional space.

    I(X,Y) = H(X) + H(Y) - H(X,Y)

    @reference:
    Kraskov, Stoegbauer & Grassberger (2004). Estimating mutual information. PHYSICAL REVIEW E 69, 066138

    Arguments:
    ----------
    x, y: (n, d) ndarray
        n samples from d-dimensional multivariate distributions

    k: int (default 1)
        kth nearest neighbour to use in density estimate;
        imposes smoothness on the underlying probability distribution

    estimator: 'ksg' or 'naive' (default 'ksg')
        'ksg'  : see Kraskov, Stoegbauer & Grassberger (2004) Estimating mutual information, eq(8).
        'naive': entropies are calculated individually using the Kozachenko-Leonenko estimator implemented in get_h()

    Returns:
    --------
    mi: float
        mutual information I(X,Y)

    """
    # construct state array for the joint process:
    xy = np.c_[x, y]

    if estimator == 'naive':
        # compute individual entropies
        hx = get_h(x, k=k)
        hy = get_h(y, k=k)
        hxy = get_h(xy, k=k)

        # compute mi
        mi = hx + hy - hxy

    elif estimator == 'ksg':

        # store data pts in kd-trees for efficient nearest neighbour computations
        # TODO: choose a better leaf size
        x_tree = cKDTree(x)
        y_tree = cKDTree(y)
        xy_tree = cKDTree(xy)

        # kth nearest neighbour distances for every state
        # query with k=k+1 to return the nearest neighbour, not counting the data point itself
        # dist, idx = xy_tree.query(xy, k=k+1, p=norm)
        dist, idx = xy_tree.query(xy, k=k + 1, p=np.inf)
        epsilon = dist[:, -1]

        # for each point, count the number of neighbours
        # whose distance in the x-subspace is strictly < epsilon
        # repeat for the y subspace
        n = len(x)
        nx = np.empty(n, dtype=np.int)
        ny = np.empty(n, dtype=np.int)
        for ii in range(n):
            nx[ii] = len(x_tree.query_ball_point(x_tree.data[ii], r=epsilon[ii], p=np.inf)) - 1
            ny[ii] = len(y_tree.query_ball_point(y_tree.data[ii], r=epsilon[ii], p=np.inf)) - 1

        mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)  # version (1)
        # mi = digamma(k) -1./k -np.mean(digamma(nx) + digamma(ny)) + digamma(n) # version (2)

    else:
        raise NotImplementedError("Estimator is one of 'naive', 'ksg'; currently: {}".format(estimator))

    return mi
