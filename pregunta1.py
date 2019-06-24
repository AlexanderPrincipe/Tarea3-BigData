from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


conf = SparkConf().setAppName("Tarea3").setMaster("local")
sc = SparkContext(conf=conf)

sqlContext = SQLContext(sc)

# Leemos el CSV
rdd = sqlContext.read.csv("data_final.csv", header=True).rdd

# rdd
rdd = rdd.map(
	lambda x: ( int(x[21]) ,int(x[54]), int(x[55]), int(x[56]), int(x[57]), int(x[58]) , int(x[59]),
		                  int(x[60]), int(x[61]), int(x[62]),int(x[63]), int(x[64]), int(x[65]) , int(x[66]),
		                  int(x[67]), int(x[68]), int(x[69]),int(x[70]), int(x[71]), int(x[72]) , int(x[73]),
		                  int(x[74]), int(x[75]), int(x[76]),int(x[77]), int(x[78]), int(x[79]) , int(x[80]),
		                  int(x[81]), int(x[82]), int(x[83]),int(x[84]), int(x[85]), int(x[86]) , int(x[87]) ))
# Convertimos el rdd a rf                          
df = rdd.toDF(["Position","Crossing","Finishing","HeadingAccuracy","ShortPassing","Volleys","Dribbling","Curve",
                   "FKAccuracy", "LongPassing","BallControl","Acceleration","SprintSpeed","Agility","Reactions",
                   "Balance","ShotPower","Jumping","Stamina","Strength","LongShots","Aggression",
                   "Interceptions","Positioning","Vision","Penalties","Composure","Marking","StandingTackle",
                   "SlidingTackle","GKDiving","GKHandling","GKKicking","GKPositioning","GKReflexes"])
# Creamos vectorassembler
assembler = VectorAssembler(inputCols=["Position","Crossing","Finishing","HeadingAccuracy","ShortPassing","Volleys","Dribbling","Curve",
                   "FKAccuracy", "LongPassing","BallControl","Acceleration","SprintSpeed","Agility","Reactions",
                   "Balance","ShotPower","Jumping","Stamina","Strength","LongShots","Aggression",
                   "Interceptions","Positioning","Vision","Penalties","Composure","Marking","StandingTackle",
                   "SlidingTackle","GKDiving","GKHandling","GKKicking","GKPositioning","GKReflexes"], outputCol = "features")
df = assembler.transform(df)


labelIndexer = StringIndexer(
	inputCol="Position", outputCol="indexedTarget")

# Vectorindexer   
featureIndexer = VectorIndexer(
	inputCol="features",
	outputCol="indexedFeatures" ,
	maxCategories=4)

# Dividimos el dataset
(training_df, test_df) = df.randomSplit([0.7, 0.3])

# Entrenamiento
entrenador = DecisionTreeClassifier(
	labelCol="indexedTarget", 
	featuresCol="indexedFeatures")

# Creacion de pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, entrenador])
# Se entrena el modelo
model = pipeline.fit(training_df) 

# Validacion
predictions_df = model.transform(test_df)
predictions_df.select(
	"indexedFeatures", "indexedTarget", 
	"prediction", "rawPrediction").show()

# Evaluador --> Accuracy
evaluator = MulticlassClassificationEvaluator(
	labelCol="indexedTarget", predictionCol="prediction",
	metricName="accuracy")

# Accuracy
exactitud = evaluator.evaluate(predictions_df)
print("Exactitud: {}".format(exactitud))












