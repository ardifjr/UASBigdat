from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when
from pyspark.sql.types import StructType, StringType, IntegerType

spark = SparkSession.builder \
    .appName("RealTimeVitalSignsStreaming") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Schema data vital (UPDATE)
schema = StructType() \
    .add("patient_id", StringType()) \
    .add("patient_name", StringType()) \
    .add("room", StringType()) \
    .add("heart_rate", IntegerType()) \
    .add("systolic", IntegerType()) \
    .add("diastolic", IntegerType()) \
    .add("timestamp", StringType())

# Baca data dari socket
raw_stream = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Parse JSON
vitals_df = raw_stream.select(
    from_json(col("value"), schema).alias("data")
).select("data.*")

# Tambah status kesehatan
vitals_df = vitals_df.withColumn(
    "status",
    when(col("heart_rate") >= 130, "Critical")
    .when(col("heart_rate") >= 110, "Warning")
    .otherwise("Normal")
)

# Output ke console
query = vitals_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()
