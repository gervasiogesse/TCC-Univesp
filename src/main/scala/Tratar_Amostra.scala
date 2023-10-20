//Spark
import org.apache.spark.sql
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, when}

//Import para o logger
import org.apache.log4j.Logger
import org.apache.log4j.Level
object Tratar_Amostra extends App{

  println("Inicio")
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("com").setLevel(Level.ERROR)

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Train nn Phish metadata")
    .getOrCreate()


  import spark.implicits._
  //cargas df
  //  val amostra_phish = carga_csv("file:///mnt/c/Users/gogj/My Private Documents/Amostras/ProofPoint/proof_phish_gt0.csv")
  val amostra1_phish = spark.read.json("file:///mnt/c/Users/gogj/My Private Documents/Amostras/ProofPoint/json/gt75.json")
  val amostra2_phish = spark.read.json("file:///mnt/c/Users/gogj/My Private Documents/Amostras/ProofPoint/json/lt75.json")
  val amostra3_phish = spark.read.json("file:///mnt/c/Users/gogj/My Private Documents/Amostras/ProofPoint/json/gl.json")
  amostra2_phish.printSchema()
  //  amostra_phish.select($"`result.msg.lang`").show()
  val gt75 = amostra1_phish.select($"result.*").select(
    $"`duration`".cast("double"),
    $"`file_size`".cast("double"),
    $"`filter.durationSecs`".cast("double").as("durationSecs"),
    $"`filter.modules.pdr.v2.rscore`".cast("double").as("rscore"),
    $"`filter.modules.spam.scores.classifiers.adult`".cast("double").as("adult"),
    $"`filter.modules.spam.scores.classifiers.bulk`".cast("double").as("bulk"),
    $"`filter.modules.spam.scores.classifiers.impostor`".cast("double").as("impostor"),
    $"`filter.modules.spam.scores.classifiers.lowpriority`".cast("double").as("lowpriority"),
    $"`filter.modules.spam.scores.classifiers.malware`".cast("double").as("malware"),
    $"`filter.modules.spam.scores.classifiers.mlx`".cast("double").as("mlx"),
    $"`filter.modules.spam.scores.classifiers.mlxlog`".cast("double").as("mlxlog"),
    $"`filter.modules.spam.scores.classifiers.phish`".cast("double").as("phish"),
    $"`filter.modules.spam.scores.classifiers.sn`".cast("double").as("sn"),
    $"`filter.modules.spam.scores.classifiers.spam`".cast("double").as("spam"),
    $"`filter.modules.spam.scores.classifiers.suspect`".cast("double").as("suspect"),
    $"`filter.modules.spam.scores.classifiers.unknownsender`".cast("double").as("unknownsender"),
    //    $"`filter.modules.spam.scores.classifiers.unsafe`".cast("double"),
    $"`filter.modules.spam.scores.engine`".cast("double").as("engine"),
    $"`filter.modules.spam.scores.overall`".cast("double").as("overall"),
    $"`filter.msgSizeBytes`".cast("double").as("msgSizeBytes"),
    $"`msgParts{}.detectedSizeBytes`".cast("double").as("detectedSizeBytes"),
    $"`msgParts{}.sizeDecodedBytes`".cast("double").as("sizeDecodedBytes"),
    $"`msgParts{}.structureId`".cast("double").as("structureId"),
    $"`filter.suborgs.rcpts{}`".cast("double").as("rcpts"),
    $"`filter.suborgs.sender`".cast("double").as("sender"),
    $"`filter_score`".cast("double"),
    $"`linecount`".cast("double"),
    $"`msg.sizeBytes`".cast("double").as("sizeBytes"),
    $"`size`".cast("double"),
    $"`timeendpos`".cast("double"),
    $"`timestartpos`".cast("double")
  )

  val glt75 = amostra2_phish.select($"result.*").select(
    $"`duration`".cast("double"),
    $"`file_size`".cast("double"),
    $"`filter.durationSecs`".cast("double").as("durationSecs"),
    $"`filter.modules.pdr.v2.rscore`".cast("double").as("rscore"),
    $"`filter.modules.spam.scores.classifiers.adult`".cast("double").as("adult"),
    $"`filter.modules.spam.scores.classifiers.bulk`".cast("double").as("bulk"),
    $"`filter.modules.spam.scores.classifiers.impostor`".cast("double").as("impostor"),
    $"`filter.modules.spam.scores.classifiers.lowpriority`".cast("double").as("lowpriority"),
    $"`filter.modules.spam.scores.classifiers.malware`".cast("double").as("malware"),
    $"`filter.modules.spam.scores.classifiers.mlx`".cast("double").as("mlx"),
    $"`filter.modules.spam.scores.classifiers.mlxlog`".cast("double").as("mlxlog"),
    $"`filter.modules.spam.scores.classifiers.phish`".cast("double").as("phish"),
    $"`filter.modules.spam.scores.classifiers.sn`".cast("double").as("sn"),
    $"`filter.modules.spam.scores.classifiers.spam`".cast("double").as("spam"),
    $"`filter.modules.spam.scores.classifiers.suspect`".cast("double").as("suspect"),
    $"`filter.modules.spam.scores.classifiers.unknownsender`".cast("double").as("unknownsender"),
    //    $"`filter.modules.spam.scores.classifiers.unsafe`".cast("double"),
    $"`filter.modules.spam.scores.engine`".cast("double").as("engine"),
    $"`filter.modules.spam.scores.overall`".cast("double").as("overall"),
    $"`filter.msgSizeBytes`".cast("double").as("msgSizeBytes"),
    $"`msgParts{}.detectedSizeBytes`".cast("double").as("detectedSizeBytes"),
    $"`msgParts{}.sizeDecodedBytes`".cast("double").as("sizeDecodedBytes"),
    $"`msgParts{}.structureId`".cast("double").as("structureId"),
    $"`filter.suborgs.rcpts{}`".cast("double").as("rcpts"),
    $"`filter.suborgs.sender`".cast("double").as("sender"),
    $"`filter_score`".cast("double"),
    $"`linecount`".cast("double"),
    $"`msg.sizeBytes`".cast("double").as("sizeBytes"),
    $"`size`".cast("double"),
    $"`timeendpos`".cast("double"),
    $"`timestartpos`".cast("double")
  )

  val gl = amostra3_phish.select($"result.*").select(
    $"`duration`".cast("double"),
    $"`file_size`".cast("double"),
    $"`filter.durationSecs`".cast("double").as("durationSecs"),
    $"`filter.modules.pdr.v2.rscore`".cast("double").as("rscore"),
    $"`filter.modules.spam.scores.classifiers.adult`".cast("double").as("adult"),
    $"`filter.modules.spam.scores.classifiers.bulk`".cast("double").as("bulk"),
    $"`filter.modules.spam.scores.classifiers.impostor`".cast("double").as("impostor"),
    $"`filter.modules.spam.scores.classifiers.lowpriority`".cast("double").as("lowpriority"),
    $"`filter.modules.spam.scores.classifiers.malware`".cast("double").as("malware"),
    $"`filter.modules.spam.scores.classifiers.mlx`".cast("double").as("mlx"),
    $"`filter.modules.spam.scores.classifiers.mlxlog`".cast("double").as("mlxlog"),
    $"`filter.modules.spam.scores.classifiers.phish`".cast("double").as("phish"),
    $"`filter.modules.spam.scores.classifiers.sn`".cast("double").as("sn"),
    $"`filter.modules.spam.scores.classifiers.spam`".cast("double").as("spam"),
    $"`filter.modules.spam.scores.classifiers.suspect`".cast("double").as("suspect"),
    $"`filter.modules.spam.scores.classifiers.unknownsender`".cast("double").as("unknownsender"),
    //    $"`filter.modules.spam.scores.classifiers.unsafe`".cast("double"),
    $"`filter.modules.spam.scores.engine`".cast("double").as("engine"),
    $"`filter.modules.spam.scores.overall`".cast("double").as("overall"),
    $"`filter.msgSizeBytes`".cast("double").as("msgSizeBytes"),
    $"`msgParts{}.detectedSizeBytes`".cast("double").as("detectedSizeBytes"),
    $"`msgParts{}.sizeDecodedBytes`".cast("double").as("sizeDecodedBytes"),
    $"`msgParts{}.structureId`".cast("double").as("structureId"),
    $"`filter.suborgs.rcpts{}`".cast("double").as("rcpts"),
    $"`filter.suborgs.sender`".cast("double").as("sender"),
    $"`filter_score`".cast("double"),
    $"`linecount`".cast("double"),
    $"`msg.sizeBytes`".cast("double").as("sizeBytes"),
    $"`size`".cast("double"),
    $"`timeendpos`".cast("double"),
    $"`timestartpos`".cast("double")
  )

  println(gt75.count())
  println(glt75.count())
  println(gl.count())

  val total = gt75.union(glt75).union(gl).
    select(col("*"),
      when(col("phish") > 75, 2.0.toDouble).when(col("phish") <= 75, 1.0.toDouble).alias("isPhish"),
      //      expr("CASE WHEN filter.modules.spam.scores.classifiers.phish > 75 then 1.0 when filter.modules.spam.scores.classifiers.phish <=75 then 0.0").alias("isPhish")
    ).na.fill(1.0)

  total.printSchema()
  println(total.count())
  total.show(false)
  total.coalesce(1)
    .write
    .option("header", "true")
    .mode("overwrite")
    .csv("amostra.csv")


}
