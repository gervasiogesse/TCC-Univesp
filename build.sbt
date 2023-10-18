ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "Phising_nn_dev"
  )
libraryDependencies ++= Seq(
//  "com.intel.analytics.bigdl" % "bigdl-assembly-spark_3.1.3" % "2.3.0",
  "org.apache.spark" %% "spark-core" % "3.1.3",
  "org.apache.spark" %% "spark-mllib" % "3.1.3",
  "org.apache.spark" %% "spark-sql" % "3.1.3"
)