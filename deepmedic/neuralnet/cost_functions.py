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
    #print(p_y_given_x_train)

    #Get sum of propabilities for all overlapping classes excluding the background
    if y_data is None:
        p_y_given_x_train_overlapping = tf.transpose(tf.transpose(p_y_given_x_train, perm=[2,3,4,0,1]) * [0.0], perm=[3,4,0,1,2])
    else:
        pixel_is_backgr = tf.equal(y_gt, 0)
        p_non_overlapping_else_0 = p_y_given_x_train * tf.reshape(tf.cast(y_data, "float32"), shape=[-1, n_classes, 1, 1, 1])
        p_non_overlapping_in_location_of_background_pixel_else_0 = p_non_overlapping_else_0 * tf.expand_dims(tf.cast(pixel_is_backgr, "float32"), axis=1)

        # p_y_given_x_train_overlapping = tf.transpose(tf.transpose(p_y_given_x_train, perm=[2,3,4,0,1]) * tf.cast(y_data, "float32"), perm=[3,4,0,1,2])
    #sum_p_y_given_x_train_overlapping = tf.reduce_sum(p_y_given_x_train_overlapping, 1) # [batchSize, r, c, z]. Sum over all non-overlaping classes (except background)
    sum_p_non_overlapping_in_loc_of_backgr = tf.reduce_sum(p_non_overlapping_in_location_of_background_pixel_else_0, axis=1)
    #print("Sum out", sum_out)
    
    # Cross Entropy Value for each pixel
    p_of_correct_class_train_pixel = p_y_given_x_train * y_one_hot
    #print("p_y_given_x_train_ace", p_y_given_x_train_ace)
    p_correct_class = tf.reduce_sum(p_of_correct_class_train_pixel, axis=1)  # [batchsize, r, c, z], float32 pred prob of correct class
    #p_correct_class = tf.gather(p_y_given_x_train, indices=y_gt, axis=1)  # Equivalent to p_y_given_x_train[:, y_gt, :, :, :] in numpy. Shape [batchsize, r, c, z]
    #print("pixel_sum", pixel_sum)
    p_correct_class_with_non_overlap_added_to_backgr = p_correct_class + sum_p_non_overlapping_in_loc_of_backgr  # [batchsize, r, c, z] + [batchsize, t, c, z] => for each pixel => p_correct_class + p_non_overlapping_in_location_of_background_pixel.
    #print(addition)

    # Negative log, for entropy
    log_p_correct_class = tf.math.log(p_correct_class_with_non_overlap_added_to_backgr + eps)
    neg_log_p = tf.math.negative(log_p_correct_class)

    # Average values for all voxels
    mean_loss_over_pixels_per_sample = tf.reduce_mean(neg_log_p, axis=[1,2,3])
    #print("Voxel mean:", voxel_mean)
    
    # Average values across batch
    mean_loss_over_batch = tf.reduce_mean(mean_loss_over_pixels_per_sample)
    #print("batch_mean:",batch_mean)

    # batch_mean = is_database_1 * standard_cross_entropy + is_database_2 * (ce_for_overlapping + y_gt[0] log(p_non_overlapping_else_0))

    return mean_loss_over_batch


def ace_old(p_y_given_x_train, y_gt, eps=1e-5, weightPerClass=None):
    # p_y_given_x_train : tensor5 [batchSize, classes, r, c, z]
    # y_gt: T.itensor4('y'). Dimensions [batchSize, r, c, z]
    # Adaptive corss entropy:
    y_one_hot = tf.one_hot( indices=y_gt, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32" )
    
    # Get all classes in the gt
    num_classes = len(tf.unstack(p_y_given_x_train, axis=1))
    unique_classes, _ = tf.unique(tf.reshape(y_gt, [-1]))
    
    # For each class not in gt set value to 1 in one hot encoding in all places where == 1 for class 0 (background)
    transpose_y_one_hot = tf.transpose(y_one_hot, perm=[1,0,2,3,4])
    unstacked_transpose_y_one_hot = [transpose_y_one_hot[0]]
    #print("Unstacked:", unstacked_transpose_y_one_hot)
    for i in range(num_classes - 1):
        f1 = lambda: tf.identity(transpose_y_one_hot[i+1])
        f2 = lambda: tf.identity(tf.where(tf.not_equal(transpose_y_one_hot[0], 1), x=transpose_y_one_hot[i+1] , y=[1]))
        condition = tf.reduce_any(tf.math.equal(unique_classes, tf.constant(i+1)))
        unstacked_transpose_y_one_hot.append(tf.cond(condition, f1, f2))
    stacked_transpose_y_one_hot = tf.stack(unstacked_transpose_y_one_hot)
    #print(stacked_transpose_y_one_hot)
    updated_y_one_hot = tf.transpose(stacked_transpose_y_one_hot, perm=[1,0,2,3,4])

    # Apply one hot encoding mask to genetated probabilities
    p_y_given_x_train_ace = p_y_given_x_train * updated_y_one_hot
    
    # Calculate the Adaptive Cross Entropy
    # TODO: Add weighthing
    if weightPerClass is not None:
        weightPerClass5D = tf.reshape(weightPerClass, shape=[1, tf.shape(p_y_given_x_train)[1], 1, 1, 1])
        weighted_log_p_y_given_x_train = p_y_given_x_train_ace * weightPerClass5D

    # for each voxel sum along the classes
    pixel_sum = tf.reduce_sum(tf.transpose(weighted_log_p_y_given_x_train, perm=[0,2,3,4,1]), 4)
    #print("Pixel_sum:", pixel_sum)

    # Negative log, for entropy
    log_p_y_given_x_train_ace = tf.math.log(pixel_sum)
    neg_log_p_y_given_x_train_ace = tf.math.negative(log_p_y_given_x_train_ace)

    # Average values for all voxels
    voxel_mean = tf.reduce_mean(neg_log_p_y_given_x_train_ace, 1)
    #print("Voxel mean:", voxel_mean)
    
    # Average values across batch
    batch_mean = tf.reduce_mean(voxel_mean)
    #print("batch_mean:",batch_mean)
    return batch_mean

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
    

