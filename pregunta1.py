from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.feature import VectorIndexer, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import MultilayerPerceptronClassifier

def leer_df(rdd):
    conf = SparkConf().setAppName("Pregunta1").setMaster("local")
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)

	# Leemos el CSV
	rdd = sqlContext.read.csv("data_tarea3.csv", header=True).rdd

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

def feature_selection():
    assembler = VectorAssembler(
        inputCols=["Position","Crossing","Finishing","HeadingAccuracy","ShortPassing","Volleys","Dribbling","Curve",
                   "FKAccuracy", "LongPassing","BallControl","Acceleration","SprintSpeed","Agility","Reactions",
                   "Balance","ShotPower","Jumping","Stamina","Strength","LongShots","Aggression",
                   "Interceptions","Positioning","Vision","Penalties","Composure","Marking","StandingTackle",
                   "SlidingTackle","GKDiving","GKHandling","GKKicking","GKPositioning","GKReflexes"],
        outputCol="features")

    df = assembler.transform(df)

    indexer = VectorIndexer(
        inputCol = "features",
        outputCol = "indexedFeatures",
        maxCategorires = 20)

    df = indexer.fit(df).transform(df)

    # feature selector
    selector = ChiSqSelector(
        numTopFeatures=20,
        featuresCol="indexedFeatures",
        labelCol="Position",
        outputCol="selectedFeatures")
    resultado = selector.fit(df).transform(df)
    resultado.select("features", "selectedFeatures").show()

    # Dividir nuestro dataset
    (training_df, test_df) = df.randomSplit([0.7, 0.3])

    ## ENTRENAMIENTO
    entrenador = DecisionTreeClassifier(
	labelCol="indexedTarget", 
	featuresCol="indexedFeatures")

    # create and return pipeline
    pipeline = Pipeline(stages=[assembler,indexer,entrenador])
    model = pipeline.fit(training_df)

    ## VALIDACION
    predictions_df = model.transform(test_df)
    predictions_df.select(
        "indexedFeatures", "indexedTarget", 
	    "prediction", "rawPrediction").show()

    evaluator = MulticlassClassificationEvaluator(
	    labelCol="indexedTarget", predictionCol="prediction",
	    metricName="accuracy")

    exactitud = evaluator.evaluate(predictions_df)
    print("Exactitud: {}".format(exactitud))

