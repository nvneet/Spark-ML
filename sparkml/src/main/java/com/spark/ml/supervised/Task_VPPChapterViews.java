package com.spark.ml.supervised;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
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
import static org.apache.spark.sql.functions.*;

public class Task_VPPChapterViews {

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
		
		csvData = csvData.filter("is_cancelled = false").drop("observation_date","is_cancelled");
		csvData = csvData.withColumn("firstSub", when(col("firstSub").isNull(), 0).otherwise(col("firstSub")))
					.withColumn("all_time_views", when(col("all_time_views").isNull(), 0).otherwise(col("all_time_views")))
					.withColumn("last_month_views", when(col("last_month_views").isNull(), 0).otherwise(col("last_month_views")))
					.withColumn("next_month_views", when(col("next_month_views").isNull(), 0).otherwise(col("next_month_views")));
		csvData = csvData.withColumnRenamed("next_month_views", "label");
		
		StringIndexer payMethodIndexer = new StringIndexer();
		csvData = payMethodIndexer.setInputCol("payment_method_type")
				.setOutputCol("paymentIndex")
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
		csvData = encoder.setInputCols(new String[] {"paymentIndex","countryIndex","rebillPeriodIndex"})
						 .setOutputCols(new String[] {"payVector","countryVector","rebillPeriodVector"})
						 .fit(csvData)
						 .transform(csvData);
		
		VectorAssembler vectorAssembler = new VectorAssembler();
		Dataset<Row> inputModelData = vectorAssembler.setInputCols(new String[] {"firstSub", "all_time_views", "last_month_views", "next_month_views", "payVector","countryVector","rebillPeriodVector"})
				.setOutputCol("features")
				.transform(csvData).select("label","features");
		Dataset<Row>[] trainAndHoldoutData = inputModelData.randomSplit(new double[] {0.9,0.1});
		Dataset<Row> trainAndTestData = trainAndHoldoutData[0];
		Dataset<Row> holdoutData = trainAndHoldoutData[1];
		
		LinearRegression lr = new LinearRegression();
		
		ParamGridBuilder pgb = new ParamGridBuilder();
		ParamMap[] paramMap = pgb.addGrid(lr.regParam(), new double[] {0.01,0.1,0.3,0.5,0.7,1})
								.addGrid(lr.elasticNetParam(), new double[] {0,0.5,1})
								.build();
		TrainValidationSplit tvs = new TrainValidationSplit();
		tvs.setEstimator(lr)
		.setEvaluator(new RegressionEvaluator().setMetricName("r2"))
		.setEstimatorParamMaps(paramMap)
		.setTrainRatio(0.9);
		
		TrainValidationSplitModel model = tvs.fit(trainAndTestData);
		
		/*****/
		// To test how model worked
		LinearRegressionModel lrModel = (LinearRegressionModel) model.bestModel();
		System.out.println("Model R2 value is = "+ lrModel.summary().r2());
		System.out.println("Model intercept value = "+lrModel.intercept()+ "  &  Model coeficient value = "+lrModel.coefficients());
		System.out.println("Model regParam = "+lrModel.getRegParam()+ "  &  Model elasticNetParam = "+lrModel.getElasticNetParam());
		/*****/
	}
}
