package com.spark.ml.unsupervised;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Kmeans {

	public static void main(String[] args) {
		//System.setProperty("hadoop.home.dir", "c:/hadoop");
		Logger.getLogger("org.apache").setLevel(Level.WARN);

		SparkSession spark = SparkSession.builder()
				.appName("Kmeans clustering Gym Data")
				//.config("spark.sql.warehouse.dir","file:///c:/tmp/")
				.master("local[*]").getOrCreate();
		
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("/root/data/spark_ml_data/GymCompetition.csv");
		csvData.show();
		
		StringIndexer conditionIndexer = new StringIndexer();
		conditionIndexer.setInputCol("Gender");
		conditionIndexer.setOutputCol("GenderIndex");
		csvData = conditionIndexer.fit(csvData).transform(csvData);
		
		OneHotEncoderEstimator encoder = new OneHotEncoderEstimator();
		encoder.setInputCols(new String[] {"GenderIndex"});
		encoder.setOutputCols(new String[] {"GenderVector"});
		csvData = encoder.fit(csvData).transform(csvData);
//		csvData.printSchema();
//		csvData.show();
		
		VectorAssembler vectorAssembler = new VectorAssembler();
		Dataset<Row> inputData = vectorAssembler
				.setInputCols(new String[] {"GenderVector","Age","Height","Weight","NoOfReps"})
				.setOutputCol("features")
				.transform(csvData).select("features");
		inputData.show();
		
		KMeans kmeans = new KMeans();
		
		for (int numberOfClusters=2; numberOfClusters<8; numberOfClusters++) {
			System.out.println("Number of clusters = " + numberOfClusters);
			kmeans.setK(numberOfClusters);
			KMeansModel model = kmeans.fit(inputData);
			Dataset<Row> predictions = model.transform(inputData);
			predictions.show();
	
	//		Vector[] clusterCentres = model.clusterCenters();
	//		for(Vector vector : clusterCentres) {System.out.println(vector);};
			
			predictions.groupBy("prediction").count().show();
			
			// Lets dive in to prection accuracy SSE & 
			//Sum of squared error SSE should be minimum
			System.out.println("Sum of squared error SSE = "+ model.computeCost(inputData));
			//Slihouette with squared Euclidian distance should be close to 1
			ClusteringEvaluator clusteringEvaluator = new ClusteringEvaluator();
			System.out.println("Slihouette with squared Euclidian distance = " + clusteringEvaluator.evaluate(predictions));
	
			// TASK - optimum number of clusters
			//Plot seperate graphs for SSE and Slihouette with respect to NoOfClusters at x-axis
			
		}
	}

}
