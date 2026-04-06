# GPU Acceleration for Inductive Logic Rule Mining in Graphs

Code and notes for a master's thesis on faster support counting for rule mining over knowledge graphs.

The repository contains:

- a Scala baseline based on RDFRules
- a direct C++ rewrite
- optimized CPU versions
- three CUDA experiments

## Layout

Most files are under `MasterThesis/`:

- `Rewritten_CPP/` - direct C++ rewrite
- `CPU_01_nonparallel/` - optimized single-threaded CPU version
- `CPU_01_parallel/` - OpenMP version
- `GPU_01/`, `GPU_02/`, `GPU_03/` - CUDA variants
- `ScalaMeasuring/` - Scala benchmark code
- `test_data/` - rule files and dataset placeholders

## Data

Large `.ttl` datasets are not kept in the repository because they are too large for GitHub.

Files such as `dbpedia.ttl`, `original_train.ttl`, and similar Turtle dumps should be downloaded separately and placed in `MasterThesis/test_data/`. These are standard/public datasets and can be found online from their original sources.

## Requirements

- C++17 compiler
- `raptor2`
- `pkg-config`
- OpenMP for `CPU_01_parallel`
- CUDA Toolkit for `GPU_01`, `GPU_02`, `GPU_03`
- Java and `sbt` for `ScalaMeasuring`

`ScalaMeasuring/` also expects a local `rdfrules` checkout next to it because `build.sbt` references `../rdfrules`.

## Running

Run from inside `MasterThesis/`, since the code uses paths like `test_data/original_train.ttl`.

```bash
cd MasterThesis
```

Optimized CPU version:

```bash
g++ -std=c++17 CPU_01_nonparallel/main.cpp CPU_01_nonparallel/RdfIndexes.cpp CPU_01_nonparallel/FinalRule.cpp CPU_01_nonparallel/SupportCounting.cpp CPU_01_nonparallel/RuleParser.cpp $(pkg-config --cflags --libs raptor2) -O2 -o CPU_01_nonparallel/rdf_rules_test
./CPU_01_nonparallel/rdf_rules_test
```

OpenMP version:

```bash
g++ -std=c++17 -fopenmp CPU_01_parallel/main.cpp CPU_01_parallel/RdfIndexes.cpp CPU_01_parallel/FinalRule.cpp CPU_01_parallel/SupportCounting.cpp CPU_01_parallel/RuleParser.cpp $(pkg-config --cflags --libs raptor2) -O2 -o CPU_01_parallel/rdf_rules_test
./CPU_01_parallel/rdf_rules_test
```

Example CUDA build:

```bash
nvcc -O3 -std=c++17 GPU_03/bench_gpu.cu GPU_03/FinalRule.cpp GPU_03/RdfIndexes.cpp GPU_03/RuleParser.cpp $(pkg-config --cflags --libs raptor2) -o GPU_03/bench_gpu
./GPU_03/bench_gpu
```

Scala benchmark:

```bash
cd ScalaMeasuring
sbt "run ../test_data/original_train.ttl ../test_data/rules_150minutes.txt"
```
