package com.spark.ml.supervised;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class HousePriceAnalysis {

	public static void main(String[] args) {
		
		//System.setProperty("hadoop.home.dir", "c:/hadoop");
		Logger.getLogger("org.apache").setLevel(Level.WARN);

		SparkSession spark = SparkSession.builder()
				.appName("Linear Regression Model Fitting ")
				//.config("spark.sql.warehouse.dir","file:///c:/tmp/")
				.master("local[*]").getOrCreate();
		
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("/root/data/spark_ml_data/kc_house_data.csv");
		
//		csvData.printSchema();
//		csvData.show();
		
		VectorAssembler vectorAssembler = new VectorAssembler()
				.setInputCols(new String[] {"bedrooms","bathrooms","sqft_living"})
				.setOutputCol("features");
		Dataset<Row> inputModelData = vectorAssembler.transform(csvData)
				.select("price","features")
				.withColumnRenamed("price", "label");
//		inputModelData.show();
		
		Dataset<Row>[] completeModelData = inputModelData.randomSplit(new double[] {0.8,0.2});
		Dataset<Row> trainingData = completeModelData[0];
		Dataset<Row> testData = completeModelData[1];
		
//		LinearRegressionModel model = new LinearRegression().fit(trainingData);
		LinearRegression linearRegression = new LinearRegression();
		ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
		ParamMap[] paramMap = paramGridBuilder
				.addGrid(linearRegression.regParam(), new double[] {0.01,0.5,0.1})
				.addGrid(linearRegression.elasticNetParam(), new double[] {0,0.5,1})
				.build();
		
		System.out.println("training data =>  r2: " + model.summary().r2() + "      And the RMSE: "+ model.summary().rootMeanSquaredError());
//		model.transform(testData).show();
		System.out.println("test data =>  r2: " + model.evaluate(testData).r2() + "     And the RMSE: "+ model.evaluate(testData).rootMeanSquaredError());
	}

}
