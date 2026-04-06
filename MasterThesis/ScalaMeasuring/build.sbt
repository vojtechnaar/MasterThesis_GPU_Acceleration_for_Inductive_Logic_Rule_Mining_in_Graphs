name := "scala-measuring"

organization := "com.masterthesis"

version := "1.0.0"

scalaVersion := "2.13.8"

scalacOptions := Seq("-unchecked", "-deprecation", "-feature", "-encoding", "utf8")

// ── Resolve core from the local rdfrules checkout ────────────────────────
lazy val root = project
  .in(file("."))
  .dependsOn(rdfrules_core)

lazy val rdfrules_core = ProjectRef(file("../rdfrules"), "core")

libraryDependencies += "org.slf4j" % "slf4j-simple" % "1.7.36"

fork := true
javaOptions += "-Xmx10g"
