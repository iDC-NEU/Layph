# Layph: Making Change Propagation Constraint in Incremental Graph Processing by Layering Graph

 We implement Layph runtime engine based on Ingress[1] and Alibaba’s [libgrape-lite](https://github.com/alibaba/libgrape-lite)[2].

## Building **Layph**

### Dependencies
**Layph** is developed and tested on Ubuntu 9.4.0. It should also work on other unix-like distributions. Building Layph requires the following softwares installed as dependencies.

- [CMake](https://cmake.org/) (>=2.8)
- A modern C++ compiler compliant with C++-11 standard. (g++ >= 4.8.1)
- [MPICH](https://www.mpich.org/) (>= 3.0) or [OpenMPI](https://www.open-mpi.org/) (>= 3.0.0)
- [glog](https://github.com/google/glog) (>= 0.3.4)
- [gflags](https://github.com/gflags/gflags) (>= 2.2.0);

### Building Layph and examples

```bash
  mkdir build
  cd build
  cmake .. -DUSE_CILK=true
  make layph
  # run PageRank
  mpirun -n 1 ./layph -application pagerank -vfile test.v -efile test.base -efile_update test.update -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency 16 -compress=1 -portion=1 -build_index_concurrency=16
  # run single-source shortest path
  mpirun -n 1 ./layph -application pagerank -vfile test.v -efile test.base -efile_update test.update -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency 16 -compress=1 -portion=1 -sssp_source=0 -build_index_concurrency=16
```

## REFERENCES

 1. Gong S, Tian C, Yin Q, et al. Automating incremental graph processing with flexible memoization[J]. Proceedings of the VLDB Endowment, 2021, 14(9): 1613-1625.
 2. Fan W, Xu J, Wu Y, et al. GRAPE: Parallelizing sequential graph computations[J]. Proceedings of the VLDB Endowment, 2017, 10(12): 1889-1892.