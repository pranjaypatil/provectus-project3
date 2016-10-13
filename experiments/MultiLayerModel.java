package org.edu.uga.imageclassification.model;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class MultiLayerModel {


		    private int height;
		    private int width;
		    private int channels;
		    private int numLabels;
		    private long seed;
		    private int iterations;
		    protected int[] nIn;
		    protected int[] nOut;
		    protected String activation;
		    protected WeightInit weightInit;
		    protected OptimizationAlgorithm optimizationAlgorithm;
		    protected Updater updater;
		    protected LossFunctions.LossFunction lossFunctions;
		    protected double learningRate;
		    protected double biasLearningRate;
		    protected boolean regularization;
		    protected double l2;
		    protected double momentum;

		    MultiLayerConfiguration conf;


		    public MultiLayerModel(int height, int width, int channels, int numLabels, long seed,
		                            int iterations, int[] nIn, int[] nOut, String activation,
		                            WeightInit weightInit, OptimizationAlgorithm optimizationAlgorithm,
		                            Updater updater, LossFunctions.LossFunction lossFunctions,
		                            double learningRate, double biasLearningRate,
		                            boolean regularization, double l2, double momentum) {

		        this.height = height;
		        this.width = width;
		        this.channels = channels;
		        this.numLabels = numLabels;
		        this.seed = seed;
		        this.iterations = iterations;
		        this.nIn = nIn;
		        this.nOut = nOut;
		        this.activation = activation;
		        this.weightInit = weightInit;
		        this.optimizationAlgorithm = optimizationAlgorithm;
		        this.updater = updater;
		        this.lossFunctions = lossFunctions;
		        this.learningRate = learningRate;
		        this.biasLearningRate = (biasLearningRate == Double.NaN)? learningRate: biasLearningRate;
		        this.regularization = regularization;
		        this.l2 = l2;
		        this.momentum = momentum;
		    }
		    



		public MultiLayerNetwork buildCNNetwork() {
			MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
		            .seed(seed)
		            .iterations(iterations)
		            .activation(activation)
		            .weightInit(weightInit).dist(new GaussianDistribution(0, 1e-4))
		            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
		            .learningRateDecayPolicy(LearningRatePolicy.Step)
		            .lrPolicyDecayRate(1)
		            .lrPolicySteps(5000)
		            .optimizationAlgo(optimizationAlgorithm)
		            .learningRate(learningRate).biasLearningRate(biasLearningRate)
		            .updater(updater).momentum(momentum)
		            .regularization(regularization).l2(l2)
		            .list()
		            .layer(0, new ConvolutionLayer.Builder(7, 7)
		                    .name("cnn1")
		                    .nOut(50)
		                    .activation("relu")
		                    .nIn(channels)
		                    .stride(1, 1)
		                    .padding(2,2)
		                    .build())
		            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
		                    .name("pool1")
		                    .build())
		            .layer(2, new BatchNormalization.Builder().build())
		            .layer(3, new ActivationLayer.Builder().build())
		            .layer(4, new ConvolutionLayer.Builder(5, 5)
		                    .name("cnn2")
		                    .nOut(nOut[1])
		                    .activation("identity")
		                    .stride(1, 1)
		                    .padding(2, 2)
		                    .biasInit(0)
		                    .dist(new GaussianDistribution(0, 1e-2))
		                    .build())
		            .layer(5, new ConvolutionLayer.Builder(5, 5)
		                    .name("cnn2")
		                    .nOut(nOut[1])
		                    .activation("identity")
		                    .stride(1, 1)
		                    .padding(2, 2)
		                    .biasInit(0)
		                    .dist(new GaussianDistribution(0, 1e-2))
		                    .build())
		            .layer(6, new ConvolutionLayer.Builder(5, 5)
		                    .name("cnn2")
		                    .nOut(nOut[1])
		                    .activation("identity")
		                    .stride(1, 1)
		                    .padding(2, 2)
		                    .biasInit(0)
		                    .dist(new GaussianDistribution(0, 1e-2))
		                    .build())
		            .layer(7, new ConvolutionLayer.Builder(5, 5)
		                    .name("cnn2")
		                    .nOut(nOut[1])
		                    .activation("identity")
		                    .stride(1, 1)
		                    .padding(2, 2)
		                    .biasInit(0)
		                    .dist(new GaussianDistribution(0, 1e-2))
		                    .build())
		            .layer(8, new BatchNormalization.Builder().build())
		            .layer(9, new ActivationLayer.Builder().build())
		            .layer(10, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
		                    .name("pool2")
		                    .build())
		            .layer(11, new ConvolutionLayer.Builder(5, 5)
		                    .name("cnn3")
		                    .nOut(nOut[2])
		                    .activation("identity")
		                    .stride(1, 1)
		                    .padding(2, 2)
		                    .biasInit(0)
		                    .dist(new GaussianDistribution(0, 1e-2))
		                    .build())
		            .layer(12, new ConvolutionLayer.Builder(5, 5)
		                    .name("cnn3")
		                    .nOut(nOut[2])
		                    .activation("identity")
		                    .stride(1, 1)
		                    .padding(2, 2)
		                    .biasInit(0)
		                    .dist(new GaussianDistribution(0, 1e-2))
		                    .build())
		            .layer(13, new ConvolutionLayer.Builder(5, 5)
		                    .name("cnn3")
		                    .nOut(nOut[2])
		                    .activation("identity")
		                    .stride(1, 1)
		                    .padding(2, 2)
		                    .biasInit(0)
		                    .dist(new GaussianDistribution(0, 1e-2))
		                    .build())
		            .layer(14, new ConvolutionLayer.Builder(5, 5)
		                    .name("cnn3")
		                    .nOut(nOut[2])
		                    .activation("identity")
		                    .stride(1, 1)
		                    .padding(2, 2)
		                    .biasInit(0)
		                    .dist(new GaussianDistribution(0, 1e-2))
		                    .build())
		            .layer(15, new ConvolutionLayer.Builder(5, 5)
		                    .name("cnn3")
		                    .nOut(nOut[2])
		                    .activation("identity")
		                    .stride(1, 1)
		                    .padding(2, 2)
		                    .biasInit(0)
		                    .dist(new GaussianDistribution(0, 1e-2))
		                    .build())
		            .layer(16, new BatchNormalization.Builder().build())
		            .layer(17, new ActivationLayer.Builder().build())
		            .layer(18, new OutputLayer.Builder(lossFunctions)
		                    .nOut(numLabels)
		                    .activation("softmax")
		                    .dist(new GaussianDistribution(0, 1e-2))
		                    .build())
		            .backprop(true).pretrain(false)
		            .setInputType(InputType.convolutional(height, width, channels));
		            
		    conf = builder.build();
		    
		    MultiLayerNetwork network = new MultiLayerNetwork(conf);
	        network.init();
	        return network;
		}
	
}
