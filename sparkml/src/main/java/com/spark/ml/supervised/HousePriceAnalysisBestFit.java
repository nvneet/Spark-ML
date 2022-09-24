package com.spark.ml.supervised;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.col;

public class HousePriceAnalysisBestFit {

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
		
//		StringIndexer waterfrontIndexer = new StringIndexer();
//		waterfrontIndexer.setInputCol("waterfront");
//		waterfrontIndexer.setOutputCol("waterfrontIndex");
//		csvData = waterfrontIndexer.fit(csvData).transform(csvData);
		
		StringIndexer conditionIndexer = new StringIndexer();
		conditionIndexer.setInputCol("condition");
		conditionIndexer.setOutputCol("conditionIndex");
		//csvData = conditionIndexer.fit(csvData).transform(csvData);
		
		StringIndexer gradeIndexer = new StringIndexer();
		gradeIndexer.setInputCol("grade");
		gradeIndexer.setOutputCol("gradeIndex");
		//csvData = gradeIndexer.fit(csvData).transform(csvData);
		
		StringIndexer zipcodeIndexer = new StringIndexer();
		zipcodeIndexer.setInputCol("zipcode");
		zipcodeIndexer.setOutputCol("zipcodeIndex");
		//csvData = zipcodeIndexer.fit(csvData).transform(csvData);
		
		OneHotEncoderEstimator encoder = new OneHotEncoderEstimator();
		encoder.setInputCols(new String[] {"conditionIndex","gradeIndex","zipcodeIndex"});
		encoder.setOutputCols(new String[] {"conditionVector","gradeVector","zipcodeVector"});
		//csvData = encoder.fit(csvData).transform(csvData);
//		csvData.show();
		
		//csvData.printSchema();
		csvData = csvData.withColumn("sqft_above_percentage", col("sqft_above").divide(col("sqft_living")))
				.withColumnRenamed("price","label");
		
		
		VectorAssembler vectorAssembler = new VectorAssembler()
				.setInputCols(new String[] {"bedrooms","bathrooms","sqft_living","sqft_above_percentage","floors","conditionVector","gradeVector","zipcodeVector","waterfront"})
				.setOutputCol("features");
		Dataset<Row> inputModelData = vectorAssembler.transform(csvData)
				.select("price","features")
				.withColumnRenamed("price", "label");
//		inputModelData.show();
		
		Dataset<Row>[] completeModelData = inputModelData.randomSplit(new double[] {0.8,0.2});
		Dataset<Row> trainingAndTestData = completeModelData[0];
		Dataset<Row> holdoutData = completeModelData[1];
		
//		LinearRegressionModel model = new LinearRegression().fit(trainingData);
		LinearRegression linearRegression = new LinearRegression();
		ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
		ParamMap[] paramMap = paramGridBuilder
				.addGrid(linearRegression.regParam(), new double[] {0.01,0.5,0.1})
				.addGrid(linearRegression.elasticNetParam(), new double[] {0,0.5,1})
				.build();
		
		 TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
					.setEstimator(linearRegression)
					.setEvaluator(new RegressionEvaluator().setMetricName("r2"))
					.setEstimatorParamMaps(paramMap)
					.setTrainRatio(0.8);

//		TrainValidationSplitModel model = trainValidationSplit.fit(trainingAndTestData);
//		LinearRegressionModel lrModel = (LinearRegressionModel) model.bestModel();
		Pipeline pipeline = new Pipeline();
		pipeline.setStages(new PipelineStage[] {conditionIndexer,gradeIndexer,zipcodeIndexer, encoder, vectorAssembler, trainValidationSplit});
		PipelineModel pipelineModel = pipeline.fit(trainingAndTestData);
		//retrieve model from pipeline object
		TrainValidationSplitModel model = (TrainValidationSplitModel) pipelineModel.stages()[5];
		LinearRegressionModel lrModel = (LinearRegressionModel) model.bestModel();
		// handle feature error
		Dataset<Row> holdoutResults = pipelineModel.transform(holdoutData);
		holdoutResults.show();
		//
		holdoutResults = holdoutResults.drop("prediction");
		
		System.out.println("training data =>  r2: " + lrModel.summary().r2() + "      And the RMSE: "+ lrModel.summary().rootMeanSquaredError());
//		model.transform(testData).show();
		System.out.println("holdoutData data =>  r2: " + lrModel.evaluate(holdoutData).r2() + "     And the RMSE: "+ lrModel.evaluate(holdoutData).rootMeanSquaredError());
	}

}
