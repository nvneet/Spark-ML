package com.spark.ml.supervised;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class HousePriceFeatureSelection {

	public static void main(String[] args) {
		Logger.getLogger("org.apache").setLevel(Level.WARN);

		SparkSession spark = SparkSession.builder()
				.appName("Linear Regression Model Fitting ")
				//.config("spark.sql.warehouse.dir","file:///c:/tmp/")
				.master("local[*]").getOrCreate();
		
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("/root/data/spark_ml_data/kc_house_data.csv");
		
//		csvData.describe().show();
		csvData = csvData.drop("id","date","waterfront","view","condition","grade","yr_renovated","zipcode","lat","long");
		
		for (String col : csvData.columns())
			System.out.println("The correlation between price and "+col+" = " +  csvData.stat().corr("price", col));
		
		csvData = csvData.drop("sqft_lot","yr_built","sqft_lot15","sqft_living15");
		for (String col1 : csvData.columns()) {
			for (String col2 : csvData.columns()) {
				System.out.println("The correlation between "+col1+" and "+col2+" = " +  csvData.stat().corr(col1, col2));
			}
		}
			

	}

}
