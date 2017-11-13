import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

#######from cifar10 import img_size, num_channels, num_classes
img_size_cropped = 24


# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt
import cifar10


################################################################################### my dataset
from PIL import Image as PImage
import numpy as np 
from os import listdir
import re
#import tflearn
import tensorflow as tf
import scipy.misc as smp
import cv2

#το αρχικό resize (see size tupple) 
img_size = 28

num_channels = 3
num_classes = 2 #Normal this is 5 (now we are on PP4) 

test_len = 500 #δεν αλλάζει μόνο με αυτό 


def loadImages(path):
    # return array of images
    loadedPixels = []
    #natural sort
    imagesList = sorted(listdir(path), key=lambda x: (int(re.sub('\D','',x)),x))
    loadedImages = []
    for image in imagesList:
        img = PImage.open(path + image)
        loadedImages.append(img)
        loadedPixels.append(np.asarray(img))
        img.close()

    return loadedImages, loadedPixels


path = "all_data/"
imgs, pixels = loadImages(path)

#lengh of dataset
lenOfDS = len(pixels)

#showImage(pixels[0])

#load labels
with open("labels.txt") as f:
    labels = f.readlines()
labels = [x.strip() for x in labels] 

#convert the line to attributes 
for i in range(lenOfDS):
    labels[i]=labels[i].split()

#convert -1 to 0
for i in range(lenOfDS):
    for j in range(5):
        if (labels[i][j] == '-1'):
            labels[i][j] = 0
        else:
            labels[i][j] = 1


Y=np.array(labels)

#print(pixels.shape)

#convert all images to 50x50 using opencv
X = []
size = (28,28)


#add all small images to X matrix
for i in range(lenOfDS):
    resized_image = cv2.resize(pixels[i], size) 
    X.append(resized_image)    

#shuffle
import random
c = list(zip(X, Y))
random.shuffle(c)
X, Y = zip(*c)

X = np.array(X)
Y = np.array(Y)

class_names = ['desert', 'mountains', 'sea', 'sunset', 'trees']



def PP4Trainer(do_optimize, label_to_play, no_of_iterations=1000):


    #PT2 transformation keep only the one hot features
    def PT2_transformation (X,Y):

        retX = []
        retY = []

        for i in range(len(Y)):
            positive_counter = 0
            for j in range(5):
                if Y[i][j] == 1:
                    positive_counter = positive_counter + 1
            #keep only the one hot
            if positive_counter == 1: 
                retX.append(X[i])
                retY.append(Y[i])

        return np.array(retX), np.asarray(retY)

    #XP2, YP2 = PT2_transformation(X,Y)

    #δημιουργεί 5 διαφορετικά label lists ένα για κάθε feature
    #δυαδικά -> [ανήκει στο feature, δεν ανήκει στο feature]

    #give the label you want to receive staff for
    def PP4_transformation(label):
        # [1,0] ανήκει στη κλάση 
        # [0,1] δεν ανήκει στη κλάση 
        desertLabels = []
        mountainsLabels = []
        seaLabels = []
        sunsetLabels = []
        treeLabels = []

        cls_desert = []
        cls_mountains = []
        cls_sea = []
        cls_sunset = []
        cls_tree = []

        #desert Labels
        for i in range(len(Y)):
            if np.argmax(Y[i]) == 0:
                desertLabels.append([1,0])
                cls_desert.append(0)
            else:
                desertLabels.append([0,1])
                cls_desert.append(1)


        desertLabels = np.array(desertLabels)
        cls_desert = np.array(cls_desert)

        #mountain Labels
        for i in range(len(Y)):
            if np.argmax(Y[i]) == 1:
                mountainsLabels.append([1,0])
                cls_mountains.append(1)
            else:
                mountainsLabels.append([0,1])
                cls_mountains.append(1)


        mountainsLabels = np.array(mountainsLabels)
        cls_mountains = np.array(cls_mountains)

        #sea Labels
        for i in range(len(Y)):
            if np.argmax(Y[i]) == 2:
                seaLabels.append([1,0])
                cls_sea.append(0)
            else:
                seaLabels.append([0,1])
                cls_sea.append(1)


        seaLabels = np.array(seaLabels)
        cls_sea = np.array(cls_sea)

        #sunset Labels
        for i in range(len(Y)):
            if np.argmax(Y[i]) == 3:
                sunsetLabels.append([1,0])
                cls_sunset.append(0)
            else:
                sunsetLabels.append([0,1])
                cls_sunset.append(1)


        sunsetLabels = np.array(sunsetLabels)
        cls_sunset = np.array(cls_sunset)


        #trees labels
        for i in range(len(Y)):
            if np.argmax(Y[i]) == 4:
                treeLabels.append([1,0])
                cls_tree.append(0)
            else:
                treeLabels.append([0,1])
                cls_tree.append(1)


        treeLabels = np.array(treeLabels)
        cls_tree = np.array(cls_tree)


        if label == "desert":
            return desertLabels, cls_desert
        elif label == "mountains":
            return mountainsLabels, cls_mountains
        elif label == "sea":
            return seaLabels, cls_sea
        elif label == "sunset":
            return sunsetLabels, cls_sunset
        elif label == "trees":
            return treeLabels, cls_tree



    images_train = X[:1500]
    images_test = X[1500:]

    all_labels, all_cls = PP4_transformation(label_to_play)

    labels_train = all_labels[:1500]
    cls_train = all_cls[:1500]

    labels_test = all_labels[1500:]
    cls_test = all_cls[1500:]

    print(cls_test)

    #print(labels_test)


    #####this example run --> 2 classes 

     




    def plot_images(images, cls_true, cls_pred=None, smooth=True):

        assert len(images) == len(cls_true) == 9

        # Create figure with sub-plots.
        fig, axes = plt.subplots(3, 3)

        # Adjust vertical spacing if we need to print ensemble and best-net.
        if cls_pred is None:
            hspace = 0.3
        else:
            hspace = 0.6
        fig.subplots_adjust(hspace=hspace, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            # Interpolation type.
            if smooth:
                interpolation = 'spline16'
            else:
                interpolation = 'nearest'

            # Plot image.
            ax.imshow(images[i, :, :, :],
                      interpolation=interpolation)
                
            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
            
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()




    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)


    def pre_process_image(image, training):
        # This function takes a single image as input,
        # and a boolean whether to build the training or testing graph.
        
        if training:
            # For training, add the following to the TensorFlow graph.

            # Randomly crop the input image.
            image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)
            
            # Randomly adjust hue, contrast and saturation.
            image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

            # Some of these functions may overflow and result in pixel
            # values beyond the [0, 1] range. It is unclear from the
            # documentation of TensorFlow 0.10.0rc0 whether this is
            # intended. A simple solution is to limit the range.

            # Limit the image pixels between [0, 1] in case of overflow.
            image = tf.minimum(image, 1.0)
            image = tf.maximum(image, 0.0)
        else:
            # For training, add the following to the TensorFlow graph.

            # Crop the input image around the centre so it is the same
            # size as images that are randomly cropped during training.
            image = tf.image.resize_image_with_crop_or_pad(image,
                                                           target_height=img_size_cropped,
                                                           target_width=img_size_cropped)

        return image


    def pre_process(images, training):
        # Use TensorFlow to loop over all the input images and call
        # the function above which takes a single image as input.
        images = tf.map_fn(lambda image: pre_process_image(image, training), images)

        return images


    distorted_images = pre_process(images=x, training=True)


    def main_network(images, training):
        # Wrap the input images as a Pretty Tensor object.
        x_pretty = pt.wrap(images)

        # Pretty Tensor uses special numbers to distinguish between
        # the training and testing phases.
        if training:
            phase = pt.Phase.train
        else:
            phase = pt.Phase.infer

        # Create the convolutional neural network using Pretty Tensor.
        # It is very similar to the previous tutorials, except
        # the use of so-called batch-normalization in the first layer.
        with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
            y_pred, loss = x_pretty.\
                conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
                max_pool(kernel=2, stride=2).\
                conv2d(kernel=5, depth=64, name='layer_conv2').\
                max_pool(kernel=2, stride=2).\
                flatten().\
                fully_connected(size=256, name='layer_fc1').\
                fully_connected(size=128, name='layer_fc2').\
                softmax_classifier(num_classes=num_classes, labels=y_true)

        return y_pred, loss

    def create_network(training):
        # Wrap the neural network in the scope named 'network'.
        # Create new variables during training, and re-use during testing.
        with tf.variable_scope('network', reuse=not training):
            # Just rename the input placeholder variable for convenience.
            images = x

            # Create TensorFlow graph for pre-processing.
     #       images = pre_process(images=images, training=training)

            # Create TensorFlow graph for the main processing.
            y_pred, loss = main_network(images=images, training=training)

        return y_pred, loss


    #use cpu if set config in Session
    config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
    #sess = tf.Session(config=config)

    #trainable = false tf will not try to optimize that
    #number of optimization iterations so far
    global_step = tf.Variable(initial_value=0,
                              name='global_step', trainable=False)

    #training network
    _, loss = create_network(training=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

    #for the test phase
    y_pred, _ = create_network(training=False)
    #το y_pred είναι το one-hot array
    #το y_pled_cls είναι σαν το σωστό label σαν αριθμός (argmaxing)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    #calculate accuracy
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()


    def get_weights_variable(layer_name):
        # Retrieve an existing variable named 'weights' in the scope
        # with the given layer_name.
        # This is awkward because the TensorFlow function was
        # really intended for another purpose.

        with tf.variable_scope("network/" + layer_name, reuse=True):
            variable = tf.get_variable('weights')

        return variable


    weights_conv1 = get_weights_variable(layer_name='layer_conv1')
    weights_conv2 = get_weights_variable(layer_name='layer_conv2')


    def get_layer_output(layer_name):
        # The name of the last operation of the convolutional layer.
        # This assumes you are using Relu as the activation-function.
        tensor_name = "network/" + layer_name + "/Relu:0"

        # Get the tensor with this name.
        tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

        return tensor



    output_conv1 = get_layer_output(layer_name='layer_conv1')
    output_conv2 = get_layer_output(layer_name='layer_conv2')

    #CPU
    #session = tf.Session(config=config)

    #GPU
    session = tf.Session()


    save_dir = 'checkpoints_multi/' #dynamic for every net
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, label_to_play + 'multi_cnn') 

    #restore weights
    try:
        print("Trying to restore last checkpoint ...")

        # Use TensorFlow to find the latest checkpoint - if any.
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

        # Try and load the data in the checkpoint.
        saver.restore(session, save_path=last_chk_path)

        # If we get to this point, the checkpoint was successfully loaded.
        print("Restored checkpoint from:", last_chk_path)
    except:
        # If the above failed for some reason, simply
        # initialize all the variables for the TensorFlow graph.
        print("Failed to restore checkpoint. Initializing variables instead.")
        session.run(tf.global_variables_initializer())

    train_batch_size = 64

    def random_batch():
        # Number of images in the training-set.
        num_images = len(images_train)

        # Create a random index.
        idx = np.random.choice(num_images,
                               size=train_batch_size,
                               replace=False)

        # Use the random index to select random images and labels.
        x_batch = images_train[idx, :, :, :]
        y_batch = labels_train[idx, :]

        return x_batch, y_batch

    def optimize(num_iterations):
        # Start-time used for printing time-usage below.
        start_time = time.time()

        for i in range(num_iterations):
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = random_batch()

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            # We also want to retrieve the global_step counter.
            i_global, _ = session.run([global_step, optimizer],
                                      feed_dict=feed_dict_train)

            # Print status to screen every 100 iterations (and last).
            if (i_global % 100 == 0) or (i == num_iterations - 1):
                # Calculate the accuracy on the training-batch.
                batch_acc = session.run(accuracy,
                                        feed_dict=feed_dict_train)

                # Print status.
                msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
                print(msg.format(i_global, batch_acc))

            # Save a checkpoint to disk every 1000 iterations (and last).
            if (i_global % 1000 == 0) or (i == num_iterations - 1):
                # Save all variables of the TensorFlow graph to a
                # checkpoint. Append the global_step counter
                # to the filename so we save the last several checkpoints.
                saver.save(session,
                           save_path=save_path,
                           global_step=global_step)

                print("Saved checkpoint.")

        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


    def plot_example_errors(cls_pred, correct):
        # This function is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # correct is a boolean array whether the predicted class
        # is equal to the true class for each image in the test-set.

        # Negate the boolean array.
        incorrect = (correct == False)
        
        # Get the images from the test-set that have been
        # incorrectly classified.
        images = images_test[incorrect]
        
        # Get the predicted classes for those images.
        cls_pred = cls_pred[incorrect]

        # Get the true classes for those images.
        cls_true = cls_test[incorrect]
        
        # Plot the first 9 images.
        plot_images(images=images[0:9],
                    cls_true=cls_true[0:9],
                    cls_pred=cls_pred[0:9])

    def plot_confusion_matrix(cls_pred):
        # This is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # Get the confusion matrix using sklearn.
        cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                              y_pred=cls_pred)  # Predicted class.

        # Print the confusion matrix as text.
        for i in range(num_classes):
            # Append the class-name to each line.
            class_name = "({}) {}".format(i, class_names[i])
            print(cm[i, :], class_name)

        # Print the class-numbers for easy reference.
        class_numbers = [" ({0})".format(i) for i in range(num_classes)]
        print("".join(class_numbers))


    # Split the data-set in batches of this size to limit RAM usage.
    batch_size = 256

    def predict_cls(images, labels, cls_true):
        # Number of images.
        num_images = len(images)

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_images, dtype=np.int)

        # Now calculate the predicted classes for the batches.
        # We will just iterate through all the batches.
        # There might be a more clever and Pythonic way of doing this.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_images:
            # The ending index for the next batch is denoted j.
            j = min(i + batch_size, num_images)

            # Create a feed-dict with the images and labels
            # between index i and j.
            feed_dict = {x: images[i:j, :],
                         y_true: labels[i:j, :]}

            # Calculate the predicted class using TensorFlow.
            cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j

        # Create a boolean array whether each image is correctly classified.
        #correct = (cls_true == cls_pred)
        correct = []

        for i in range(len(cls_true)):
            if cls_true[i] == cls_pred[i]:
                correct.append(True)
            else:
                correct.append(False)

        correct = np.array(correct)

        return correct, cls_pred



    def predict_cls_test():
        return predict_cls(images = images_test,
                           labels = labels_test,
                           cls_true = cls_test)


    def classification_accuracy(correct):
        # When averaging a boolean array, False means 0 and True means 1.
        # So we are calculating: number of True / len(correct) which is
        # the same as the classification accuracy.
        
        # Return the classification accuracy
        # and the number of correct classifications.
        return correct.mean(), correct.sum()



    def print_test_accuracy(show_example_errors=False,show_confusion_matrix=False):

        # For all the images in the test-set,
        # calculate the predicted classes and whether they are correct.
        correct, cls_pred = predict_cls_test()
        
        # Classification accuracy and the number of correct classifications.
        acc, num_correct = classification_accuracy(correct)
        
        # Number of images being classified.
        num_images = len(correct)

        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, num_correct, num_images))

        # Plot some examples of mis-classifications, if desired.
        if show_example_errors:
            print("Example errors:")
            plot_example_errors(cls_pred=cls_pred, correct=correct)

        # Plot the confusion matrix, if desired.
        if show_confusion_matrix:
            print("Confusion Matrix:")
            plot_confusion_matrix(cls_pred=cls_pred)


    def plot_conv_weights(weights, input_channel=0):
        # Assume weights are TensorFlow ops for 4-dim variables
        # e.g. weights_conv1 or weights_conv2.

        # Retrieve the values of the weight-variables from TensorFlow.
        # A feed-dict is not necessary because nothing is calculated.
        w = session.run(weights)

        # Print statistics for the weights.
        print("Min:  {0:.5f}, Max:   {1:.5f}".format(w.min(), w.max()))
        print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))
        
        # Get the lowest and highest values for the weights.
        # This is used to correct the colour intensity across
        # the images so they can be compared with each other.
        w_min = np.min(w)
        w_max = np.max(w)
        abs_max = max(abs(w_min), abs(w_max))

        # Number of filters used in the conv. layer.
        num_filters = w.shape[3]

        # Number of grids to plot.
        # Rounded-up, square-root of the number of filters.
        num_grids = math.ceil(math.sqrt(num_filters))
        
        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(num_grids, num_grids)

        # Plot all the filter-weights.
        for i, ax in enumerate(axes.flat):
            # Only plot the valid filter-weights.
            if i<num_filters:
                # Get the weights for the i'th filter of the input channel.
                # The format of this 4-dim tensor is determined by the
                # TensorFlow API. See Tutorial #02 for more details.
                img = w[:, :, input_channel, i]

                # Plot image.
                ax.imshow(img, vmin=-abs_max, vmax=abs_max,
                          interpolation='nearest', cmap='seismic')
            
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()


    def plot_layer_output(layer_output, image):
        # Assume layer_output is a 4-dim tensor
        # e.g. output_conv1 or output_conv2.

        # Create a feed-dict which holds the single input image.
        # Note that TensorFlow needs a list of images,
        # so we just create a list with this one image.
        feed_dict = {x: [image]}
        
        # Retrieve the output of the layer after inputting this image.
        values = session.run(layer_output, feed_dict=feed_dict)

        # Get the lowest and highest values.
        # This is used to correct the colour intensity across
        # the images so they can be compared with each other.
        values_min = np.min(values)
        values_max = np.max(values)

        # Number of image channels output by the conv. layer.
        num_images = values.shape[3]

        # Number of grid-cells to plot.
        # Rounded-up, square-root of the number of filters.
        num_grids = math.ceil(math.sqrt(num_images))
        
        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(num_grids, num_grids)

        # Plot all the filter-weights.
        for i, ax in enumerate(axes.flat):
            # Only plot the valid image-channels.
            if i<num_images:
                # Get the images for the i'th output channel.
                img = values[0, :, :, i]

                # Plot image.
                ax.imshow(img, vmin=values_min, vmax=values_max,
                          interpolation='nearest', cmap='binary')
            
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()



    def plot_distorted_image(image, cls_true):
        # Repeat the input image 9 times.
        image_duplicates = np.repeat(image[np.newaxis, :, :, :], 9, axis=0)

        # Create a feed-dict for TensorFlow.
        feed_dict = {x: image_duplicates}

        # Calculate only the pre-processing of the TensorFlow graph
        # which distorts the images in the feed-dict.
        result = session.run(distorted_images, feed_dict=feed_dict)

        # Plot the images.
        plot_images(images=result, cls_true=np.repeat(cls_true, 9))


    def get_test_image(i):
        return images_test[i, :, :, :], cls_test[i]


    #img, cls = get_test_image(16)
    #plot_distorted_image(img, cls)


    if do_optimize:
        optimize(num_iterations=no_of_iterations)
    

    print_test_accuracy()

    return predict_cls_test()



#create the ensample (αρχικοποποίηση)
ensPrediction = []
for i in range(test_len):
    ensPrediction.append([0,0,0,0,0])



for i in class_names:


    #i="sea"

    print(i)


    #run all the staff for desert
    print("Run a Network to identify the " + i)

    #train the network
    _ , cls_pred = PP4Trainer(True, i, no_of_iterations=500)
    for j in range(test_len):
        ensPrediction[j][class_names.index(i)] = cls_pred[j]
    #get the predictions from test set (1 -> is desert 0 -> not desert )


    break

#print(ensPrediction)
