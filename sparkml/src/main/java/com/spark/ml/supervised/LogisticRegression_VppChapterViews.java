package com.spark.ml.supervised;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.when;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionSummary;
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

public class LogisticRegression_VppChapterViews {

	public static void main(String[] args) {
		//System.setProperty("hadoop.home.dir", "c:/hadoop");
		Logger.getLogger("org.apache").setLevel(Level.WARN);

		SparkSession spark = SparkSession.builder()
				.appName("Task VPP Chapter Views")
				//.config("spark.sql.warehouse.dir","file:///c:/tmp/")
				.master("local[*]").getOrCreate();
		
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("/root/data/spark_ml_data/vppChapterViews/*.csv");
		
		/*
		 * Customer watched no videos is 0; customer watched some videos 1
		 */
		csvData = csvData.filter("is_cancelled = false").drop("observation_date","is_cancelled");
		csvData = csvData.withColumn("firstSub", when(col("firstSub").isNull(), 0).otherwise(col("firstSub")))
					.withColumn("all_time_views", when(col("all_time_views").isNull(), 0).otherwise(col("all_time_views")))
					.withColumn("last_month_views", when(col("last_month_views").isNull(), 0).otherwise(col("last_month_views")))
					.withColumn("next_month_views", when(col("next_month_views").$greater(0),0).otherwise(1));
		
		csvData = csvData.withColumnRenamed("next_month_views", "label");
		
		StringIndexer payMethodIndexer = new StringIndexer();
		csvData = payMethodIndexer.setInputCol("payment_method_type")
				.setOutputCol("payIndex")
				.fit(csvData)
				.transform(csvData);
		StringIndexer countryIndexer = new StringIndexer();
		csvData = countryIndexer.setInputCol("country")
				.setOutputCol("countryIndex")
				.fit(csvData)
				.transform(csvData);
		StringIndexer rebillPeriodIndexer = new StringIndexer();
		csvData = rebillPeriodIndexer.setInputCol("rebill_period_in_months")
				.setOutputCol("rebillPeriodIndex")
				.fit(csvData)
				.transform(csvData);
		OneHotEncoderEstimator encoder = new OneHotEncoderEstimator();
		csvData = encoder.setInputCols(new String[] {"payIndex","countryIndex","rebillPeriodIndex"})
						 .setOutputCols(new String[] {"payVector","countryVector","rebillPeriodVector"})
						 .fit(csvData)
						 .transform(csvData);
		VectorAssembler vectorAssembler = new VectorAssembler();
		Dataset<Row> inputModelData = vectorAssembler.setInputCols(new String[] {"firstSub", "age", "all_time_views", "last_month_views", "payVector","countryVector","rebillPeriodVector"})
				.setOutputCol("features")
				.transform(csvData).select("label","features");
		Dataset<Row>[] trainAndHoldoutData = inputModelData.randomSplit(new double[] {0.9,0.1});
		Dataset<Row> trainAndTestData = trainAndHoldoutData[0];
		Dataset<Row> holdoutData = trainAndHoldoutData[1];
		
		LogisticRegression logisticRegression = new LogisticRegression();
		
		ParamGridBuilder pgb = new ParamGridBuilder();
		ParamMap[] paramMap = pgb.addGrid(logisticRegression.regParam(), new double[] {0.01,0.1,0.3,0.5,0.7,1})
						         .addGrid(logisticRegression.elasticNetParam(), new double[] {0,0.5,1})
								 .build();
		TrainValidationSplit tvs = new TrainValidationSplit();
		tvs.setEstimator(logisticRegression)
		   .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
		   .setEstimatorParamMaps(paramMap)
		   .setTrainRatio(0.9);
		
		TrainValidationSplitModel model = tvs.fit(trainAndTestData);
		
		/*****/
		// To test how model worked
		LogisticRegressionModel logisticRegressionModel = (LogisticRegressionModel) model.bestModel();
		System.out.println("Model R2 value is = "+ logisticRegressionModel.summary().accuracy());
		System.out.println("Model intercept value = "+logisticRegressionModel.intercept()+ "  &  Model coeficient value = "+logisticRegressionModel.coefficients());
		System.out.println("Model regParam = "+logisticRegressionModel.getRegParam()+ "  &  Model elasticNetParam = "+logisticRegressionModel.getElasticNetParam());
		/*****/
		
		LogisticRegressionSummary summary = logisticRegressionModel.evaluate(holdoutData);
		System.out.println("summary: "+summary);
		/**
				TruePositive => truePositiveRateByLabel()[1]
				TrueNegative => truePositiveRateByLabel()[0]
				FalsePositive => falsePositiveRateByLabel()[0]
				FalseNegative => falsePositiveRateByLabel()[1]
		**/
//		double truePositive = logisticRegressionModel.evaluate(holdoutData).truePositiveRateByLabel()[1];
//		double trueNegative = logisticRegressionModel.evaluate(holdoutData).truePositiveRateByLabel()[0];
//		double falsePositive = logisticRegressionModel.evaluate(holdoutData).falsePositiveRateByLabel()[0];
//		double falseNegative = logisticRegressionModel.evaluate(holdoutData).falsePositiveRateByLabel()[1];
		
		double truePositive = summary.truePositiveRateByLabel()[1];
		double trueNegative = summary.truePositiveRateByLabel()[0];
		double falsePositive = summary.falsePositiveRateByLabel()[0];
		double falseNegative = summary.falsePositiveRateByLabel()[1];
		
		System.out.println("truePositive: "+truePositive);
		System.out.println("trueNegative: "+trueNegative);
		System.out.println("falsePositive: "+falsePositive);
		System.out.println("falseNegative: "+falseNegative);
		

		System.out.println("Probability of Positive to be correct from holdout data is: "+truePositive/(truePositive + falsePositive));
		System.out.println("holdout data accuracy is: "+summary.accuracy());
		logisticRegressionModel.transform(holdoutData).groupBy("label","prediction").count().show();
		
		
		
		
		
	}
}
