package com.masterthesis

import com.github.propi.rdfrules.data.Graph
import com.github.propi.rdfrules.index.Index
import com.github.propi.rdfrules.rule.{Measure, ResolvedAtom, ResolvedRule}
import com.github.propi.rdfrules.ruleset.Ruleset
import com.github.propi.rdfrules.utils.{Debugger, ForEach}

import scala.io.Source

/**
  * Benchmark: measures RDFRules (Scala/JVM) support counting time
  * on the same data and rules used by the C++ implementation.
  *
  * Usage:  sbt "run <train.ttl> <rules.txt>"
  */
object SupportBenchmark {

  // ── Parse one atom string like "?a biolink:related_to ?b" ──────────────
  // The rules file uses two URI forms:
  //   "biolink:localName"                    → prefix shorthand
  //   "<http://www.w3.org/…/22-rdf-syntax-ns#type>"  → full URI
  // Variables are: ?a, ?b, ?c, …

  private var prefixMap: Map[String, String] = Map.empty

  private def parsePrefixesFromTtl(ttlPath: String): Map[String, String] = {
    val src = scala.io.Source.fromFile(ttlPath, "UTF-8")
    try {
      src.getLines()
        .map(_.trim)
        .filter(l => l.toLowerCase.startsWith("@prefix") || l.toLowerCase.startsWith("prefix"))
        .flatMap { line =>
          val colonIdx = line.indexOf(':')
          if (colonIdx < 0) None
          else {
            val nameStart = if (line.toLowerCase.startsWith("@prefix")) 7 else 6
            val name = line.substring(nameStart, colonIdx).trim
            val uriStart = line.indexOf('<', colonIdx)
            val uriEnd   = if (uriStart >= 0) line.indexOf('>', uriStart + 1) else -1
            if (uriStart < 0 || uriEnd < 0) None
            else Some(s"$name:" -> line.substring(uriStart + 1, uriEnd))
          }
        }
        .toMap
    } finally src.close()
  }

  private def expandUri(raw: String): String = {
    if (raw.startsWith("<") && raw.endsWith(">")) {
      raw
    } else {
      prefixMap.find { case (prefix, _) => raw.startsWith(prefix) } match {
        case Some((prefix, ns)) => s"<$ns${raw.substring(prefix.length)}>"
        case None => s"<$raw>"
      }
    }
  }

  /** Parse "( ?a biolink:pred ?b )" into a ResolvedAtom */
  private def parseAtom(s: String): ResolvedAtom = {
    // Strip outer parens and whitespace
    val inner = s.trim.stripPrefix("(").stripSuffix(")").trim
    val parts = inner.split("\\s+")
    if (parts.length != 3)
      throw new IllegalArgumentException(s"Cannot parse atom (expected 3 parts): '$s'")

    val subj = parts(0)
    val pred = parts(1)
    val obj  = parts(2)

    ResolvedAtom.parse(subj, expandUri(pred), obj)
  }

  /** Parse one line of the rules file into a ResolvedRule.
    * Format: ( body1 ) ^ ( body2 ) => ( head ) | Support: N, HeadCoverage: D, HeadSupport: N, HeadSize: N
    */
  private def parseRuleLine(line: String): ResolvedRule = {
    // Split at "|" to separate rule part from measures part
    val pipeIdx = line.lastIndexOf('|')
    val rulePart = if (pipeIdx >= 0) line.substring(0, pipeIdx).trim else line.trim
    val measuresPart = if (pipeIdx >= 0) line.substring(pipeIdx + 1).trim else ""

    // Split rulePart at "=>" into body and head
    val arrowIdx = rulePart.indexOf("=>")
    if (arrowIdx < 0)
      throw new IllegalArgumentException(s"No '=>' found in rule: '$line'")

    val bodyStr = rulePart.substring(0, arrowIdx).trim
    val headStr = rulePart.substring(arrowIdx + 2).trim

    val head = parseAtom(headStr)

    // Body atoms are separated by " ^ "
    val bodyAtoms: IndexedSeq[ResolvedAtom] = if (bodyStr.isEmpty) {
      IndexedSeq.empty
    } else {
      // Split on " ^ " but each atom is wrapped in ( … )
      // E.g. "( ?c biolink:foo ?b ) ^ ( ?a biolink:bar ?c )"
      val atomPattern = """\(([^)]+)\)""".r
      atomPattern.findAllMatchIn(bodyStr).map(m => parseAtom(m.group(0))).toIndexedSeq
    }

    // Parse measures
    val measures = scala.collection.mutable.ArrayBuffer.empty[Measure]
    if (measuresPart.nonEmpty) {
      // "Support: 1062, HeadCoverage: 1, HeadSupport: 1062, HeadSize: 1062"
      val parts = measuresPart.split(",").map(_.trim)
      for (p <- parts) {
        val kv = p.split(":\\s*", 2)
        if (kv.length == 2) {
          val key = kv(0).trim
          val value = kv(1).trim
          key match {
            case "Support"     => measures += Measure.Support(value.toInt)
            case "HeadCoverage"=> measures += Measure.HeadCoverage(value.toDouble)
            case "HeadSupport" => measures += Measure.HeadSupport(value.toInt)
            case "HeadSize"    => measures += Measure.HeadSize(value.toInt)
            case _             => // ignore unknown measures
          }
        }
      }
    }

    ResolvedRule(bodyAtoms, head, measures.toSeq: _*)
  }

  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println("Usage: SupportBenchmark <train.ttl> <rules.txt>")
      System.exit(1)
    }

    val ttlPath   = args(0)
    val rulesPath = args(1)

    prefixMap = parsePrefixesFromTtl(ttlPath)
    println(s"Loaded ${prefixMap.size} prefix(es) from $ttlPath")

    implicit val debugger: Debugger = Debugger.EmptyDebugger

    // ── Step 1: Load TTL and build index ─────────────────────────────────
    println(s"Loading TTL: $ttlPath")
    val t0 = System.nanoTime()
    val index: Index = Graph(ttlPath).toDataset.index
    val t1 = System.nanoTime()
    println(f"Index built in ${(t1 - t0) / 1e9}%.3f s")

    // ── Step 2: Parse rules from text file ───────────────────────────────
    println(s"Parsing rules: $rulesPath")
    val src = Source.fromFile(rulesPath, "UTF-8")
    val ruleLines = try {
      src.getLines().filter(_.trim.nonEmpty).toVector
    } finally {
      src.close()
    }
    val resolvedRules: IndexedSeq[ResolvedRule] = ruleLines.map(parseRuleLine)
    println(s"Parsed ${resolvedRules.size} rules")

    // ── Step 3: Warmup — force lazy index structures ────────────────────
    //  RDFRules builds internal sorted arrays / hash maps lazily on first
    //  query.  Running one throwaway rule forces that initialization so it
    //  does not pollute the per-rule timings.
    print("Warming up index (1 throwaway rule) ... ")
    val warmupT0 = System.nanoTime()
    val warmupRuleset = Ruleset(index, ForEach.from(resolvedRules.take(1))).setParallelism(1)
    warmupRuleset.computeSupport(minSupport = 1, injectiveMapping = true)
      .resolvedRules.foreach(_ => ())   // force evaluation
    val warmupT1 = System.nanoTime()
    println(f"done in ${(warmupT1 - warmupT0) / 1e9}%.3f s")

    // ── Step 4: Create Ruleset (maps URIs to index IDs) ──────────────────
    // setParallelism(1) = single-threaded: drastically reduces memory
    val ruleset: Ruleset = Ruleset(index, ForEach.from(resolvedRules)).setParallelism(1)

    // ── Step 5: Compute support (benchmark) ──────────────────────────────
    //  With setParallelism(1), computeSupport processes rules one at a time.
    //  The lazy chain means each rule is computed and emitted inside .foreach,
    //  so we can measure per-rule time by tracking nanoTime between callbacks.
    println("Computing support (single-threaded to save memory) ...")
    println("\n========== RESULTS (streaming) ==========")
    var count = 0
    val perRuleTimes = scala.collection.mutable.ArrayBuffer.empty[Double]
    val perRuleBodySizes = scala.collection.mutable.ArrayBuffer.empty[Int]
    val t2 = System.nanoTime()
    var lastRuleTime = t2
    val withSupport: Ruleset = ruleset.computeSupport(minSupport = 1, injectiveMapping = true)
    withSupport.resolvedRules.foreach { r =>
      val now = System.nanoTime()
      val ruleMs = (now - lastRuleTime) / 1e6
      lastRuleTime = now
      count += 1
      perRuleTimes += ruleMs
      perRuleBodySizes += r.body.size
      val supp = r.measures.get(Measure.Support).map(_.value).getOrElse(-1)
      val hc   = r.measures.get(Measure.HeadCoverage).map(_.value).getOrElse(0.0)
      val hs   = r.measures.get(Measure.HeadSupport).map(_.value).getOrElse(-1)
      val hsz  = r.measures.get(Measure.HeadSize).map(_.value).getOrElse(-1)
      println(f"[$count%4d] ${ruleMs}%10.1f ms | Support: $supp, HC: $hc, HS: $hs, HSz: $hsz | $r")
      System.out.flush()
    }
    val t3 = System.nanoTime()

    val supportTimeSec = (t3 - t2) / 1e9

    // ── Statistics ───────────────────────────────────────────────────────
    val sortedTimes = perRuleTimes.sorted
    val avgMs  = if (perRuleTimes.nonEmpty) perRuleTimes.sum / perRuleTimes.size else 0.0
    val medMs  = if (sortedTimes.nonEmpty) sortedTimes(sortedTimes.size / 2) else 0.0
    val minMs  = if (sortedTimes.nonEmpty) sortedTimes.head else 0.0
    val maxMs  = if (sortedTimes.nonEmpty) sortedTimes.last else 0.0
    val p95Ms  = if (sortedTimes.nonEmpty) sortedTimes((sortedTimes.size * 0.95).toInt.min(sortedTimes.size - 1)) else 0.0

    println("\n========== TIMING SUMMARY ==========")
    println(f"Index build time:        ${(t1 - t0) / 1e9}%.3f s")
    println(f"Warmup time:             ${(warmupT1 - warmupT0) / 1e9}%.3f s")
    println(f"Support counting time:   ${supportTimeSec}%.3f s")
    println(f"Total rules parsed:      ${resolvedRules.size}")
    println(f"Rules surviving (supp>=1): $count")
    println(f"Avg time per rule:       ${avgMs}%.1f ms")
    println(f"Median time per rule:    ${medMs}%.1f ms")
    println(f"Min time per rule:       ${minMs}%.1f ms")
    println(f"Max time per rule:       ${maxMs}%.1f ms")
    println(f"P95 time per rule:       ${p95Ms}%.1f ms")
    println(f"Total wall time:         ${(t3 - t0) / 1e9}%.3f s")

    // Per-body-size statistics
    println("\n========== STATISTICS BY BODY SIZE ==========")
    val grouped = perRuleTimes.zip(perRuleBodySizes).groupBy(_._2)
    for (bodySize <- grouped.keys.toSeq.sorted) {
      val times = grouped(bodySize).map(_._1).sorted
      val n = times.size
      val avg = times.sum / n
      val med = times(n / 2)
      val p95 = times(((n * 0.95).toInt).min(n - 1))
      println(f"Body size $bodySize ($n rules): avg=$avg%.1f ms, median=$med%.1f ms, min=${times.head}%.1f ms, max=${times.last}%.1f ms, P95=$p95%.1f ms")
    }
  }
}
