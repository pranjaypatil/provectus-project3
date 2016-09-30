from __future__ import print_function
from pyspark import  SparkContext,SparkConf
import tensorflow as tf
import tensorflow as ops
from StringIO import StringIO
from PIL import Image
import numpy as np
import math
import os


sc = SparkContext(conf=SparkConf().setAppName("MalwareClassifier"))

def joinString(inputString):
    temp=inputString[0]+" "+inputString[1]
    return temp

def main():
    ***REMOVED***
    ***REMOVED***
    imageFileName = sc.textFile("s3n://eds-uga-csci8360/data/project3/metadata/X_small_train.txt").map(
        lambda doc: doc.encode("utf-8").strip())
    imageFileNameWithZip = imageFileName.zipWithIndex().map(lambda doc: (doc[1], doc[0]))
    path = imageFileName.map(lambda doc: ("s3n://eds-uga-csci8360/data/project3/images/" + doc + ".png"))
    filePath = path.reduce(lambda str1, str2: str1 + "," + str2)
    imageFileCollect = sc.binaryFiles(filePath, 36)
    imageContent = imageFileCollect.map(lambda (x,y): (os.path.splitext(os.path.basename(x))[0].encode('utf-8'),(np.asarray(Image.open(StringIO(y))))))
    #imageContent=imageFileCollect.map(lambda (x, y): (os.path.splitext(os.path.basename(x))[0].encode('utf-8'), y))


    #byteFile = imageFileName.map(lambda doc: ("s3n://eds-uga-csci8360/data/project3/images/" + doc + ".png"))
    imageLabel = sc.textFile("s3n://eds-uga-csci8360/data/project3/metadata/y_small_train.txt").map(
        lambda doc: doc.encode("utf-8").strip()).cache()
    imageLabelwithZip = imageLabel.zipWithIndex().map(lambda doc: (doc[1], doc[0]))
    imageFileLabelPair=imageFileNameWithZip.join(imageLabelwithZip,numPartitions=2)
    imageFileLabelValue=imageFileLabelPair.values()
    imageFileLabel = imageFileLabelValue.keyBy(lambda line: line[0]).mapValues(lambda line: line[1])
    imageLabel=imageContent.join(imageFileLabel,numPartitions=2)

    print (imageLabel.take(1))




    #print ("labelPath_List",len(labelPathBroadCast.value))


if __name__ == "__main__":
    main()
