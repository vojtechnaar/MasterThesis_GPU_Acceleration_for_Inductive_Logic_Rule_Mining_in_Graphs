logLevel := Level.Warn

// Required because ProjectRef loads the rdfrules parent build
// which uses these plugins for other subprojects (gui, experiments)
addSbtPlugin("org.scala-js" % "sbt-scalajs" % "1.10.0")
addSbtPlugin("org.xerial.sbt" % "sbt-pack" % "0.14")
