package com.spark.ml.unsupervised;

import static org.apache.spark.sql.functions.col;

import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class CollaborativeFiltering {

	public static void main(String[] args) {
		//System.setProperty("hadoop.home.dir", "c:/hadoop");
		Logger.getLogger("org.apache").setLevel(Level.WARN);

		SparkSession spark = SparkSession.builder()
				.appName("Vpp Course Views.")
//				.config("spark.sql.warehouse.dir","file:///c:/tmp/")
				.master("local[*]").getOrCreate();
		
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("/root/data/spark_ml_data/VPPcourseViews.csv");
		
		csvData = csvData.withColumn("proportionWatched", col("proportionWatched").multiply(100));
//		csvData.show();
		
//		csvData = csvData.groupBy("userId").pivot("courseId").sum("proportionWatched");
//		csvData.show();
		
//		Dataset<Row>[] trainingAndHoldoutData = csvData.randomSplit(new double[] {0.9,0.1});
//		Dataset<Row> trainingData = trainingAndHoldoutData[0];
//		Dataset<Row> holdoutata = trainingAndHoldoutData[1];
		
		ALS als = new ALS()
				.setMaxIter(10)
				.setRegParam(0.1)
				.setUserCol("userId")
				.setItemCol("courseId")
				.setRatingCol("proportionWatched");		
		ALSModel alsModel = als.fit(csvData);
		//alsModel.setColdStartStrategy("nan");
		alsModel.setColdStartStrategy("drop");
		
//		Dataset<Row> predictions = alsModel.transform(holdoutata);
		
//		Dataset<Row> userRecce = alsModel.recommendForAllUsers(5);
//		List<Row> userRecceList = userRecce.takeAsList(5);
//		
//		for (Row r : userRecceList) {
//			int userId = r.getAs(0);
//			String recce = r.getAs(1).toString();
//			System.out.println("user " + userId + ", We might want to recommend " + recce);
//			System.out.println("This user has already watched.");
//			csvData.filter("userId = " + userId).show();
//		}
		
		Dataset<Row> testData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("/root/data/spark_ml_data/VPPcourseViewsTest.csv");
		testData.show();
		alsModel.transform(testData).show();
		alsModel.recommendForUserSubset(testData, 5).show();
	}
}