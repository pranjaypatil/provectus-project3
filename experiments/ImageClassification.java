package org.edu.uga.imageclassification;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.AbstractJavaRDDLike;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.ComposableRecordReader;
import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.edu.uga.imageclassification.model.MultiLayerModel;
import org.edu.uga.imageclassification.utility.ImageUtility;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import scala.Tuple2;

public class ImageClassification {

	protected static final int height = 32;
	protected static final int width = 32;
	protected static final int channels = 3;
	protected static final int numLabels = 10;
	protected static int batchSize = 32;
	protected static int iterations = 1;
	protected static int seed = 42;
	protected static Random rng = new Random(seed);
	protected static String[] label = new String[] { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" };
	protected static List<String> labels = Arrays.asList(label);

	public static void main(String[] args) throws Exception {

        // Setup SparkContext
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[6]");
        sparkConf.setAppName("ImageClassifcation");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        
        
        
        //Create JavaRDD for training and testing data
        ImageUtility imgUtil=new ImageUtility();
        Map<String, Integer> imgFileLabel=imgUtil.mapImageLabel("/home/shubhi/X_small_train.txt", "/home/shubhi/y_small_train.txt");
        Map<String,Integer> imgTestFileLabel=imgUtil.mapImageLabel("/home/shubhi/X_small_test.txt", "/home/shubhi/y_small_test.txt");
        List<DataSet> trainDataset=imgUtil.converImagetoMatrix(imgFileLabel);
        List<DataSet> testDataset=imgUtil.converImagetoMatrix(imgTestFileLabel);
        JavaRDD<DataSet> training=sc.parallelize(trainDataset);
        training.cache();
        JavaRDD<DataSet> testing=sc.parallelize(testDataset);
   
      
        //Creating batch size Dataset for Training data
        JavaRDD<DataSet> trainFinal=training.mapPartitions(new FlatMapFunction<Iterator<DataSet>, DataSet>(){
        	public Iterable<DataSet> call(Iterator<DataSet> dataSetIterator) throws Exception {
        		List<DataSet> temp = new ArrayList<>();
        		List<DataSet> dataBatch = new ArrayList<>();
        		while (dataSetIterator.hasNext()) {
        			temp.add(dataSetIterator.next());
        			if (temp.size() == batchSize) {
        				dataBatch.add(DataSet.merge(temp));
        				 temp.clear();
        			}
        			 if(!temp.isEmpty()){
        				 dataBatch.add(DataSet.merge(temp));}
	        		
		        	
        		}
        		return dataBatch;
        	
        	}
        });
 
        
      //Creating batch size Dataset for Testing data
        JavaRDD<DataSet> testFinal=testing.mapPartitions(new FlatMapFunction<Iterator<DataSet>, DataSet>(){
        	public Iterable<DataSet> call(Iterator<DataSet> dataSetTestIterator) throws Exception {
        		List<DataSet> temp = new ArrayList<>();
        		List<DataSet> dataBatch = new ArrayList<>();
        		while (dataSetTestIterator.hasNext()) {
        			//DataSet temp =dataSetIterator.next();
        			temp.add(dataSetTestIterator.next());
        			if (temp.size() == batchSize) {
        				dataBatch.add(DataSet.merge(temp));
        				 temp.clear();
        			}
        			 if(!temp.isEmpty()){
        				 dataBatch.add(DataSet.merge(temp));}
	        		
		        	
        		}
        		return dataBatch;
        	
        	}
        });

    		 MultiLayerNetwork network = new MultiLayerModel(
    	             height,
    	             width,
    	             channels,
    	             numLabels,
    	             seed,
    	             iterations,
    	             null,
    	             new int[]{32, 32, 64},
    	             "relu",
    	             WeightInit.XAVIER,
    	             OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT,
    	             Updater.ADAM,
    	             LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD,
    	             1e-3,
    	             2e-3,
    	             true,
    	             4e-3,
    	             0.9).buildCNNetwork();
		   		
     
    
    //Parameter averaging for building Training Master
     ParameterAveragingTrainingMaster trainMaster = new ParameterAveragingTrainingMaster.Builder(1)
             .workerPrefetchNumBatches(0)
             .saveUpdater(true)
             .averagingFrequency(5)
             .batchSizePerWorker(batchSize)
             .build();

     //Create Spark multi layer network from configuration
     SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, network, trainMaster);

     
     // Data fitted to the training model
     sparkNetwork.fit(trainFinal);
    
     trainFinal.unpersist();

     
     // Making prediction using the trained model 
     Evaluation predictedData =sparkNetwork.evaluate(testFinal);
     System.out.println(predictedData.stats());
     
     
     testFinal.unpersist();

	
	
	}

}
