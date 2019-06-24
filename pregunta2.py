from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorIndexer, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Cargar el CSV
def cargar_data():
	conf = SparkConf().setAppName("Pregunta2").setMaster("local")
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)
	return sqlContext.read.csv("data_tarea3.csv", header=True).rdd

# Columnas del CSV a utilizar
def preprocesar_data(rdd):
	rdd = rdd.map(lambda x: ( x[21] ,int(x[54]), int(x[55]), int(x[56]), int(x[57]), int(x[58]) , int(x[59]),
		                  int(x[60]), int(x[61]), int(x[62]),int(x[63]), int(x[64]), int(x[65]) , int(x[66]),
		                  int(x[67]), int(x[68]), int(x[69]),int(x[70]), int(x[71]), int(x[72]) , int(x[73]),
		                  int(x[74]), int(x[75]), int(x[76]),int(x[77]), int(x[78]), int(x[79]) , int(x[80]),
		                  int(x[81]), int(x[82]), int(x[83]),int(x[84]), int(x[85]), int(x[86]) , int(x[87])))
	df = rdd.toDF(["Position","Crossing","Finishing","HeadingAccuracy","ShortPassing","Volleys","Dribbling","Curve",
                   "FKAccuracy", "LongPassing","BallControl","Acceleration","SprintSpeed","Agility","Reactions",
                   "Balance","ShotPower","Jumping","Stamina","Strength","LongShots","Aggression",
                   "Interceptions","Positioning","Vision","Penalties","Composure","Marking","StandingTackle",
                   "SlidingTackle","GKDiving","GKHandling","GKKicking","GKPositioning","GKReflexes"])
	return df

# Entrenamiento
def entrenar(df):
	vectorAssembler = VectorAssembler(
		inputCols=["Position","Crossing","Finishing","HeadingAccuracy","ShortPassing","Volleys","Dribbling","Curve",
                   "FKAccuracy", "LongPassing","BallControl","Acceleration","SprintSpeed","Agility","Reactions",
                   "Balance","ShotPower","Jumping","Stamina","Strength","LongShots","Aggression",
                   "Interceptions","Positioning","Vision","Penalties","Composure","Marking","StandingTackle",
                   "SlidingTackle","GKDiving","GKHandling","GKKicking","GKPositioning","GKReflexes"],
        outputCol="features")

	stringIndexer = StringIndexer(inputCol="target", 
		outputCol="indexedLabel")
	vectorIndexer = VectorIndexer(inputCol="features", 
		outputCol="indexedFeatures")

	# Division en data de entrenamiento y data de test
	(training_df, test_df) = df.randomSplit([0.7, 0.3])

	# Configurar Red Neuronal
	capas = [13, 13, 13, 2]
	entrenador = MultilayerPerceptronClassifier(
		layers=capas, 
		featuresCol="indexedFeatures",
		labelCol="indexedLabel",
		maxIter=10000
	)

	# Crear pipeline
	pipeline = Pipeline(
		stages=[vectorAssembler,
				stringIndexer, 
				vectorIndexer, 
				entrenador]
	)
	return pipeline.fit(training_df), test_df

# Validacion
def validar(modelo, test_df):
	predictions_df = modelo.transform(test_df)
	predictions_df.select("indexedLabel", 
		"probability", "prediction").show()
	evaluador = MulticlassClassificationEvaluator(
		labelCol="indexedLabel", predictionCol="prediction",
		metricName="accuracy"
	)

	# Exactitud
	exactitud = evaluador.evaluate(predictions_df)
	print("Exactitud: {}".format(exactitud))
	print("PARAMS:{}".format(modelo.explainParams()))


if __name__ == "__main__":
	rdd = cargar_data()
	df = preprocesar_data(rdd)
	modelo, test_df = entrenar(df)
	validar(modelo, test_df)







	



