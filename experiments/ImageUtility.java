package org.edu.uga.imageclassification.utility;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import org.datavec.image.loader.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class ImageUtility {


	
	public List<DataSet>  converImagetoMatrix(Map<String,Integer> path) throws FileNotFoundException, IOException{
		List<DataSet> listDataSet=new ArrayList<DataSet>();
		INDArray imgArray=null;
		int i=1;
		for(Map.Entry<String,Integer> entry:path.entrySet()){
			ImageLoader image=new ImageLoader(32,32,3);
			 imgArray=image.asMatrix(new FileInputStream(new File("/home/shubhi/Downloads/image_data/"+entry.getKey()+".png")));		
			INDArray label=Nd4j.create(new double[]{entry.getValue()},new int[]{0});
			DataSet dataSet=new DataSet(imgArray, label);
			listDataSet.add(dataSet);
			
		}
		return listDataSet;
		
	}
	/*public static void main(String[] args) throws FileNotFoundException, IOException{
		Map<String, Integer> imgFileLabel=mapImageLabel("/home/dharamendra/X_small_train.txt", "/home/dharamendra/y_small_train.txt");
		
		//converImagetoMatrix(imgFileLabel);
		
	}*/
	
	public  Map<String,Integer> mapImageLabel(String imageFile,String labelPath) throws IOException{
		Map<String,Integer> imgFileLabel=new LinkedHashMap<String,Integer>();
		BufferedReader brImg=null;
		Scanner brLbl=null;
		String imgName=null;
		brImg=new BufferedReader(new FileReader(new File(imageFile)));
		brLbl=new Scanner(new FileReader(new File(labelPath)));
		
		while((imgName=brImg.readLine())!=null&&brLbl.hasNext()){
			int i=brLbl.nextInt();
			imgFileLabel.put(imgName, i);
			
		}
		brImg.close();
		brLbl.close();
		return imgFileLabel;
		
	}
}
