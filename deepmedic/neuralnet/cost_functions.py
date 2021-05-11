# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import tensorflow as tf


def x_entr( p_y_given_x_train, y_gt, weightPerClass, eps=1e-6 ):
    # p_y_given_x_train : tensor5 [batchSize, classes, r, c, z]
    # y: T.itensor4('y'). Dimensions [batchSize, r, c, z]
    # weightPerClass is a vector with 1 element per class.
    
    #Weighting the cost of the different classes in the cost-function, in order to counter class imbalance.
    log_p_y_given_x_train = tf.math.log( p_y_given_x_train + eps)
    
    weightPerClass5D = tf.reshape(weightPerClass, shape=[1, tf.shape(p_y_given_x_train)[1], 1, 1, 1])
    weighted_log_p_y_given_x_train = log_p_y_given_x_train * weightPerClass5D
    
    y_one_hot = tf.one_hot( indices=y_gt, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32" )
    
    num_samples = tf.cast( tf.reduce_prod( tf.shape(y_gt) ), "float32")
    
    return - (1./ num_samples) * tf.reduce_sum( weighted_log_p_y_given_x_train * y_one_hot )

def x_entr_mean_teacher( p_y_given_x_train, y_gt, weightPerClass, y_data, eps=1e-6 ):
    # p_y_given_x_train : tensor5 [batchSize, classes, r, c, z]
    # y: T.itensor4('y'). Dimensions [batchSize, r, c, z]
    # weightPerClass is a vector with 1 element per class.
    # y_data: ... [batchsize, classes]
    
    #Weighting the cost of the different classes in the cost-function, in order to counter class imbalance.
    log_p_y_given_x_train = tf.math.log( p_y_given_x_train + eps) # [batchSize, classes, r, c, z]
    
    weightPerClass5D = tf.reshape(weightPerClass, shape=[1, tf.shape(p_y_given_x_train)[1], 1, 1, 1]) # [1, classes, 1,1,1]
    weighted_log_p_y_given_x_train = log_p_y_given_x_train * weightPerClass5D # [batchSize, classes, r, c, z]
    
    y_one_hot = tf.one_hot( indices=y_gt, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32" ) # [batchSize, classes, r, c, z]
    
    num_samples = tf.cast( tf.reduce_prod( tf.shape(y_gt) ), "float32")

    batches_to_include = tf.reshape( tf.cast( tf.math.logical_not( tf.math.reduce_any(y_data, axis=1) ), "float32" ), shape=[-1,1,1,1,1] ) 
    
    return - (1./ num_samples) * tf.reduce_sum( weighted_log_p_y_given_x_train * y_one_hot * batches_to_include )


def iou(p_y_given_x_train, y_gt, eps=1e-5):
    # Intersection-Over-Union / Jaccard: https://en.wikipedia.org/wiki/Jaccard_index
    # Analysed in: Nowozin S, Optimal Decisions from Probabilistic Models: the Intersection-over-Union Case, CVPR 2014
    # First computes IOU per class. Finally averages over the class-ious.
    # p_y_given_x_train : tensor5 [batchSize, classes, r, c, z]
    # y: T.itensor4('y'). Dimensions [batchSize, r, c, z]
    y_one_hot = tf.one_hot( indices=y_gt, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32" )
    ones_at_real_negs = tf.cast( tf.less(y_one_hot, 0.0001), dtype="float32") # tf.equal(y_one_hot,0), but less may be more stable with floats.
    numer = tf.reduce_sum(p_y_given_x_train * y_one_hot, axis=(0,2,3,4)) # 2 * TP
    denom = tf.reduce_sum(p_y_given_x_train * ones_at_real_negs, axis=(0,2,3,4)) + tf.reduce_sum(y_one_hot, axis=(0,2,3,4)) # Pred + RP
    iou = (numer + eps) / (denom + eps) # eps in both num/den => dsc=1 when class missing.
    av_class_iou = tf.reduce_mean(iou) # Along the class-axis. Mean DSC of classes. 
    cost = 1. - av_class_iou
    return cost


def dsc(p_y_given_x_train, y_gt, eps=1e-5):
    # Similar to Intersection-Over-Union / Jaccard above.
    # Dice coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    y_one_hot = tf.one_hot( indices=y_gt, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32" )
    numer = 2. * tf.reduce_sum(p_y_given_x_train * y_one_hot, axis=(0,2,3,4)) # 2 * TP
    denom = tf.reduce_sum(p_y_given_x_train, axis=(0,2,3,4)) + tf.reduce_sum(y_one_hot, axis=(0,2,3,4)) # Pred + RP
    dsc = (numer + eps) / (denom + eps) # eps in both num/den => dsc=1 when class missing.
    av_class_dsc = tf.reduce_mean(dsc) # Along the class-axis. Mean DSC of classes. 
    cost = 1. - av_class_dsc
    return Cost

def ace(p_y_given_x_train, y_gt, y_data=None, eps=1e-5, weightPerClass=None):
    # p_y_given_x_train : tensor5 [batchSize, classes, r, c, z]
    # y_gt: T.itensor4('y'). Dimensions [batchSize, r, c, z]
    # y_data: ... [batchsize, classes] 
    # Adaptive corss entropy:

    # Get one hot encoding for ground truth. 
    n_classes = tf.shape(p_y_given_x_train)[1]
    y_one_hot = tf.one_hot( indices=y_gt, depth=n_classes, axis=1, dtype="float32" )

    # Get sum of propabilities for all overlapping classes excluding the background in each background location
    if y_data is None:
        sum_p_non_overlapping_in_loc_of_backgr = 0.0
    else:
        pixel_is_backgr = tf.equal(y_gt, 0) # [batchSize, r, c, z]
        p_non_overlapping_else_0 = p_y_given_x_train * tf.reshape(tf.cast(y_data, "float32"), shape=[-1, n_classes, 1, 1, 1]) # [batchSize, classes, r, c, z]
        p_non_overlapping_in_location_of_background_pixel_else_0 = p_non_overlapping_else_0 * tf.expand_dims(tf.cast(pixel_is_backgr, "float32"), axis=1) # [batchSize, classes, r, c, z]
        sum_p_non_overlapping_in_loc_of_backgr = tf.reduce_sum(p_non_overlapping_in_location_of_background_pixel_else_0, axis=1) # [batchSize, r, c, z]
    
    # Probability value for each pixel according to adaptive cross entropy
    p_of_correct_class_train_pixel = p_y_given_x_train * y_one_hot
    p_correct_class = tf.reduce_sum(p_of_correct_class_train_pixel, axis=1)  # [batchsize, r, c, z], float32 pred prob of correct class
    # For each pixel => p_correct_class + p_non_overlapping_in_location_of_background_pixel.
    p_correct_class_with_non_overlap_added_to_backgr = tf.math.add(p_correct_class, sum_p_non_overlapping_in_loc_of_backgr)  # [batchsize, r, c, z] + [batchsize, t, c, z] 

    # Negative log of probabilites
    log_p_correct_class = tf.math.log(p_correct_class_with_non_overlap_added_to_backgr + eps)
    neg_log_p = tf.math.negative(log_p_correct_class)

    # Mean loss values for all voxels
    mean_loss_over_pixels_per_sample = tf.reduce_mean(neg_log_p, axis=[1,2,3])
    
    # Mean loss values across batch
    mean_loss_over_batch = tf.reduce_mean(mean_loss_over_pixels_per_sample)

    # batch_mean = is_database_1 * standard_cross_entropy + is_database_2 * (ce_for_overlapping + y_gt[0] log(p_non_overlapping_else_0))

    return mean_loss_over_batch

def consistency_reg(p_y_given_x_train, p_y_given_x_ma_train):
    # TODO[gf4417] Explore KL divergence etc.
    # calculate the MSE
    mse = tf.reduce_mean((p_y_given_x_train - p_y_given_x_ma_train) ** 2, -1)
    return mse

def cost_L1(prms):
    # prms: list of tensors
    cost = 0
    for prm in prms:
        cost += tf.reduce_sum(tf.abs(prm))
    return cost

def cost_L2(prms) : #Called for L2 weigths regularisation
    # prms: list of tensors
    cost = 0
    for prm in prms:
        cost += tf.reduce_sum(prm ** 2)
    return cost
    

