Information
===========

This is an implementation of the cascade algorithm described in [1]. This algorithm
is used to accelerate detection for Deformable Part Models [2] with a high
number of components (~10x the usual number of components). To train and test
our DPMs we use the [AirplanOID dataset](http://www.robots.ox.ac.uk/~vgg/data/oid/),
which contains 7413 aeroplane images with annotations for five object part types 
(nose, verticalStabilizer, wheel, wing, aeroplane).

For questions concerning the code or bug reports, please send an email to: 
stavros DOT tsogkas AT ecp DOT fr



References
==========

[1] **Understanding Objects in Detail with Fine-grained Attributes**, A. Vedaldi,
	S. Mahendran, S. Tsogkas, S. Maji, B. Girshick, J. Kannala, E. Rahtu, I. Kokkinos, 
	M. B. Blaschko, D. Weiss, B. Taskar, K. Simonyan, N. Saphra, and S. Mohamed,
	in Proceedings of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2014

[2] **Object detection with discriminatively trained part based models**,
	P. F. Felzenszwalb, R. B. Girshick, D. McAllester, and D. Ramanan. PAMI 2010.
	
How to cite
===========
@inproceedings{vedaldi14understanding,
    Author    = {A. Vedaldi and 
                 S. Mahendran and   
                 S. Tsogkas and   
                 S. Maji and  
                 B. Girshick and 
                 J. Kannala and  
                 E. Rahtu and  
                 I. Kokkinos and  
                 M. B. Blaschko and 
                 D. Weiss and  
                 B. Taskar and  
                 K. Simonyan and  
                 N. Saphra and   
                 S. Mohamed},
    Title     = {Understanding Objects in Detail with Fine-grained Attributes},  
    Booktitle = {Proceedings of the {IEEE} Conf. on Computer Vision and Pattern Recognition ({CVPR})},  
    Year      = {2014}}

	
Folder structure
================
In the root directory you will find the following folders:

###data/
This is where all your data go. Dataset images, saved models, cascade .mat files 
and results go into separate corresponding folders.

###jar-tools/
Functions for collecting data, loading the AirplanOID annotations with additional
fields, evaluating a trained model etc.

###voc-release5/
The code for training and testing a typical DPM, slightly modified/extended to support our 
cascade. You can find the original version [here](http://www.cs.berkeley.edu/~rbg/latent/index.html).

###cpp/
c++ source code and mex-files.


Getting started
===============

To get started  just do the following:

1. Download the [AirplanOID dataset](http://www.robots.ox.ac.uk/~vgg/data/oid/) 
and place the images under oid_1.0/data/images/aeroplane/.

2. Download and install [VLFeat](http://www.vlfeat.org/install-matlab.html).
If you don't want to train/test a model on PASCAL and you just want to try out 
the cascade, you can skip this step.

3. Open a Matlab session, cd into the root directory oid_1.0/ 
and run startupOID.m. This will add all subfolders in your working path and
compile the necessary mex files placing them into the cpp/ directory.

4. Run demo.m to visualize and compare the cascade. If you want more information
about different configurations, run: help oidConfig.m.

> *Hint*: If you want to automatically add the paths and compile the mex files 
each time you open matlab while in the oid_1.0/ folder, just change the name of 
the startup script from startupOID.m to startup.m


Building your own hierarchical filter cascade
=============================================

The above steps use the files and pre-trained models already included in this distribution.
If you want to train a new model and build an hierarchical filter tree on top of that, perform the following steps: 

1. Download the 2007 [trainval data](http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and [VOCdevkit](http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCdevkit_08-Jun-2007.tar) and place the merged 2007 folder in oid_1.0/data/.

2. Change training configuration settings (if you do not want to use the defaults) in voc_config.m

3. Run model = jar_train(part, nComp, 1, 1), where *part* is the name of the aeroplane part
(e.g. 'nose') and *nComp* the number of components for your model. Please before training a new model read the section 'Model alignment' below.

4. Use the *testCascade* function to build the hierarchical filter tree and evaluate the performance of your model using precision-recall curves. 


Model alignement
=================

There are two ways to train a DPM model on the OID dataset using left/right 
facing clusters during the first round of training (see jar_train.m).

* The first one is to use the default function *lrsplit* that comes with voc-release5
(call jar_train with usevp = 0). This function clusters the positive examples in
left- and right- facing but arranges left- and right-facing filters in a random 
way inside the model structure. 

* The second way is to use *getOrientationStrings* and *splitLeftRight*. These functions
take advantage of the viewpoint annotations that come with the OID and arrange filters
so that left-facing are assigned odd indices and right-facing are assigned even indices.
While the filter order in the model structure does not affect the final result (since all
filters are considered at test time), the code that constructs the filter tree hierarchy
assumes that odd indices correspond to left-facing filters and even indices to right-facing
filters. For this reason we suggest that you use the second way (call jar_train.m with 
usevp = 1) if you want to train a new model and use our cascade.


