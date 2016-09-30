from __future__ import print_function
from pyspark import  SparkContext,SparkConf
import tensorflow as tf
import tensorflow as ops


import numpy as np


sc = SparkContext(conf=SparkConf().setAppName("MalwareClassifier"))

def joinString(inputString):
    temp=inputString[0]+" "+inputString[1]
    return temp
def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    print ("label",label)
    return example, label
def main():
    ***REMOVED***
    ***REMOVED***
    imageFileName = sc.textFile("s3n://eds-uga-csci8360/data/project3/metadata/X_small_train.txt").map(
        lambda doc: doc.encode("utf-8").strip())
    imageFileNameWithZip = imageFileName.zipWithIndex().map(lambda doc: (doc[1], "s3n://eds-uga-csci8360/data/project3/images/"+doc[0]+".png"))
    #byteFile = imageFileName.map(lambda doc: ("s3n://eds-uga-csci8360/data/project3/images/" + doc + ".png"))
    imageLabel = sc.textFile("s3n://eds-uga-csci8360/data/project3/metadata/y_small_train.txt").map(
        lambda doc: doc.encode("utf-8").strip()).cache()
    imageLabelwithZip = imageLabel.zipWithIndex().map(lambda doc: (doc[1], doc[0]))
    imageFileLabelPair=imageFileNameWithZip.join(imageLabelwithZip,numPartitions=2)
    imageFileLabelValue=imageFileLabelPair.values()
    imagePath=imageFileLabelValue.map(lambda line:line[0])
    imagesPath_List=imagePath.collect()
    imagePathBroacast=sc.broadcast(imagesPath_List)

    #print ("imagesPath_List",len(imagePathBroacast.value))
    labelPath=imageFileLabelValue.map(lambda line:[1])
    labelPath_List=labelPath.collect()
    labelPathBroadCast=sc.broadcast(labelPath_List)
    with tf.Session() as sess:
        images = ops.convert_to_tensor(imagePathBroacast.value, dtype=tf.string)
        labels=ops.convert_to_tensor(labelPathBroadCast.value,dtype=tf.int32)
        input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=5,
                                                shuffle=True)
        image, label = read_images_from_disk(input_queue)
        tf.initialize_all_variables().run()

    #print ("labelPath_List",len(labelPathBroadCast.value))


if __name__ == "__main__":
    main()
