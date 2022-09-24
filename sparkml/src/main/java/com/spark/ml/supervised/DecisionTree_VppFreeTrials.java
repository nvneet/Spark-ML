package com.spark.ml.supervised;

import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.regression.*;

import static org.apache.spark.sql.functions.*;

public class DecisionTree_VppFreeTrials {

	private static final String DecisionTreeRegressor = null;

	public static void main(String[] args) {
		
		//System.setProperty("hadoop.home.dir", "c:/hadoop");
		Logger.getLogger("org.apache").setLevel(Level.WARN);

		SparkSession spark = SparkSession.builder()
				.appName("Decision Tree vppFreeTrials")
				//.config("spark.sql.warehouse.dir","file:///c:/tmp/")
				.master("local[*]").getOrCreate();
		spark.udf().register("countryGrouping", countryGrouping, DataTypes.StringType);
		
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("/root/data/spark_ml_data/DecisionTree/vppFreeTrials.csv");
		
		
		csvData = csvData.withColumn("country", callUDF("countryGrouping", col("country")))
						 .withColumn("label", when(col("payments_made").geq(1),lit(1)).otherwise(lit(0)));
		
		StringIndexer countryIndexer = new StringIndexer();
		csvData = countryIndexer.setInputCol("country")
							    .setOutputCol("countryIndex")
							    .fit(csvData)
							    .transform(csvData);
//		Dataset<Row> countryIndexes = csvData.select("countryIndex").distinct();
//		IndexToString indexToString = new IndexToString();
//		indexToString.setInputCol("countryIndex")
//					 .setOutputCol("value").transform(countryIndexes).show();
		
//		new IndexToString().setInputCol("countryIndex").setOutputCol("value").transform(csvData.select("countryIndex").distinct()).show();
		new IndexToString()
				.setInputCol("countryIndex")
				.setOutputCol("value")
				.transform(csvData.select("countryIndex").distinct())
				.show();
		VectorAssembler vectorAssembler = new VectorAssembler();
		vectorAssembler.setInputCols(new String[] {"countryIndex","rebill_period","chapter_access_count","seconds_watched"})
					   .setOutputCol("features");
		Dataset<Row> inputData = vectorAssembler.transform(csvData).select("label","features");
		//inputData.show();
		
		// Now Start building the Model using Decision Tree Regressor
//		Dataset<Row>[] trainingAndHoldoutData = inputData.randomSplit(new double[] {0.8,0.2});
//		Dataset<Row> trainingData = trainingAndHoldoutData[0];
//		Dataset<Row> holdoutData = trainingAndHoldoutData[1];
//		DecisionTreeRegressor dtRegressor = new DecisionTreeRegressor();
//		dtRegressor.setMaxDepth(3);
//		DecisionTreeRegressionModel model = dtRegressor.fit(trainingData);
//		model.transform(holdoutData).show();
//		System.out.println(model.toDebugString());
		
		// Now Start building the Model using Decision Tree Classifier
		Dataset<Row>[] trainingAndHoldoutData = inputData.randomSplit(new double[] {0.8,0.2});
		Dataset<Row> trainingData = trainingAndHoldoutData[0];
		Dataset<Row> holdoutData = trainingAndHoldoutData[1];
		DecisionTreeClassifier dtClassifier = new DecisionTreeClassifier();
		dtClassifier.setMaxDepth(3);
		DecisionTreeClassificationModel model = dtClassifier.fit(trainingData);
		Dataset<Row> prediction = model.transform(holdoutData);
		System.out.println(model.toDebugString());
		
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
		evaluator.setMetricName("accuracy");
		System.out.println("The accuracy of model is = " + evaluator.evaluate(prediction));
		
	}
	
	public static UDF1<String,String> countryGrouping = new UDF1<String,String>() {

		private static final long serialVersionUID = 3755260946466573083L;

		public String call(String country) throws Exception {
			List<String> topCountries =  Arrays.asList(new String[] {"GB","US","IN","UNKNOWN"});
			List<String> europeanCountries =  Arrays.asList(new String[] {"BE","BG","CZ","DK","DE","EE","IE","EL","ES","FR","HR","IT","CY","LV","LT","LU","HU","MT","NL","AT","PL","PT","RO","SI","SK","FI","SE","CH","IS","NO","LI","EU"});
			
			if (topCountries.contains(country)) return country; 
			if (europeanCountries .contains(country)) return "EUROPE";
			else return "OTHER";
		}
	};
}
