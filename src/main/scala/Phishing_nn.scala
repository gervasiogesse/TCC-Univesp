import com.intel.analytics.bigdl.BIGDL_VERSION
import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.nn.{ClassNLLCriterion, Linear, LogSoftMax, Sequential}
import com.intel.analytics.bigdl.dllib.nnframes.NNClassifier
import com.intel.analytics.bigdl.dllib.optim.Adam
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.{col, expr, from_json, lit, schema_of_json, when}

//Spark
import org.apache.spark.sql
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Column
// Import dos módulos VectorAssembler e Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StandardScaler

import breeze.numerics.sqrt

//Import para o logger
import org.apache.log4j.Logger
import org.apache.log4j.Level

object Phishing_nn extends App {
  println("Inicio")
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("com").setLevel(Level.ERROR)

  def carga_csv(path: String): sql.DataFrame = {
    spark.read.option("header", "true")
      .option("inferSchema", "true")
      .option("mode", "DROPMALFORMED")
      .format("csv")
      .load(path)
  }

  val conf = Engine.createSparkConf()
    .setAppName("Train nn Phish metadata")
    .setMaster("local[*]")
    .set("spark.task.maxFailures", "1")
  val spark = SparkSession.builder().config(conf = conf).getOrCreate()
  val sc = NNContext.initNNContext("Run Example")

  import spark.implicits._
  println("BigDL" + BIGDL_VERSION)

  //cargas df
  val total = spark.read
    .option("inferSchema" , "true")
    .option("header", "true")
    .csv("amostra.csv")
  total.show(false)
  total.groupBy("isPhish").count().show()
  // Cria um novo objeto VectorAssembler chamado assembler as features
  // Defina a coluna de saída
  val assembler = new VectorAssembler()
    .setInputCols(Array(
      "detectedSizeBytes","duration","file_size","durationSecs","rscore","adult","bulk","impostor",
      "lowpriority","malware","mlx","mlxlog","sn","spam","suspect","unknownsender","engine",
      "overall","msgSizeBytes","rcpts","sender","filter_score","linecount","sizeBytes","size",
      "timeendpos","timestartpos","sizeDecodedBytes","structureId"
    )).setOutputCol("features1")
  //  Use 'StandardScaler' to scale the features
  val fv1 = assembler.transform(total).select($"features1", $"isPhish")
  println("fv1 qtd = " + fv1.count().toInt)

  val scaler = new StandardScaler().setInputCol("features1").setOutputCol("features")
  val fv = scaler.fit(fv1).transform(fv1)
  fv.show(2, false)

  // Use randomSplit para criar uma divisão em treino e teste em 70/30
  val Array(training, test) = fv.randomSplit(Array(0.7, 0.3), seed = 12345)
  val qtd = training.count().toInt - (training.count() % 4).toInt
  println("qtd = " + qtd)

  // Definindo o numero de camadas ocultas
  val n_input = 29
  val n_classes = 2

  // Função para determinar a quantdade de neuronios da primeira camada para iniciar os testes
  val n_hidden_1 = sqrt(sqrt((n_classes + 2) * n_input) + 2 * sqrt(n_input / (n_classes + 2))).toInt + 1
  println("n camadas 1 = " + n_hidden_1)
  // Segunda camada
  val n_hidden_2 = n_classes * sqrt(n_input / (n_classes + 2)).toInt
  println("n camadas 2 = " + n_hidden_2)

  // Incializa um container sequencial
  val nn = Sequential
    .apply()
    .add(Linear.apply(n_input, 20))
    .add(Linear.apply(20, n_hidden_1))
    .add(Linear.apply(n_hidden_1, n_hidden_2))
    .add(Linear.apply(n_hidden_2, n_classes))
    .add(LogSoftMax.apply())
  val criterion = ClassNLLCriterion()
  val estimator = NNClassifier(model = nn, criterion = criterion, Array(n_input))
    .setMaxEpoch(100)
    .setBatchSize(qtd)
    .setLearningRate(0.1)
    .setLabelCol("isPhish").setFeaturesCol("features1")
    .setOptimMethod(new Adam[Float](learningRate = 0.1))
  println("Inicio do treino")
  training.printSchema()
  val model = estimator.fit(fv)
  println("Fim do treino")
  val predictions = model.transform(fv)
  predictions.groupBy("prediction").count().show()
  test.groupBy("isPhish").count().show()
  predictions.show()
  // Para métricas e avaliação, importe MulticlassMetrics

  import org.apache.spark.mllib.evaluation.MulticlassMetrics

  // Converta os resultados do teste em um RDD usando .as e .rdd
  val predictionAndLabels = predictions.select($"prediction", $"isPhish").as[(Double, Double)].rdd

  // Instanciar um novo objeto MulticlassMetrics
  val metrics = new MulticlassMetrics(predictionAndLabels)

  // Confusion matrix
  println("Confusion matrix:")
  println(metrics.confusionMatrix)
  println("Precisão:")
  println(metrics.accuracy)

}
