/** Copyright 2020 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "flags.h"

#include <gflags/gflags.h>

/* flags related to the job. */
DEFINE_string(application, "", "application name");
DEFINE_string(efile, "", "edge file");
DEFINE_string(vfile, "", "vertex file");
DEFINE_string(efile_update, "", "edge file described how to edit edges");
DEFINE_string(out_prefix, "", "output directory of results");
DEFINE_string(jobid, "", "jobid, only used in LDBC graphanalytics.");
DEFINE_bool(directed, true, "input graph is directed or not.");
DEFINE_double(portion, 1, "priority.");
DEFINE_bool(cilk, false, "use cilk");
DEFINE_bool(verify, false, "verify correctness of result");

/* flags related to specific applications. */
DEFINE_int64(sssp_source, 0, "source vertex of sssp.");
DEFINE_int64(php_source, 0, "source vertex of sssp.");
DEFINE_double(php_tol, 0.001,
              "The probability diff of two continuous iterations");
DEFINE_double(php_d, 0.8, "damping factor of PHP");
DEFINE_int32(php_mr, 10, "max rounds of PHP");
DEFINE_double(pr_d, 0.85, "damping factor of pagerank");
DEFINE_int32(pr_mr, 100, "max rounds of pagerank");
DEFINE_double(pr_tol, 0.001, "pr tolerance");
DEFINE_double(pr_delta_sum, 0.0001, "delta sum of delta-based pagerank");
DEFINE_int32(gcn_mr, 3, "max rounds of GCN");

DEFINE_bool(segmented_partition, true,
            "whether to use segmented partitioning.");

DEFINE_string(serialization_prefix, "",
              "where to load/store the serialization files");

DEFINE_int32(app_concurrency, -1, "concurrency of application");

DEFINE_bool(debug, false, "");
DEFINE_double(termcheck_threshold, 1000000000, "");
DEFINE_bool(d2ud_weighted, false, "output weight");
DEFINE_bool(compress, false, "use compress");
DEFINE_bool(count_skeleton, false, "count skeleton");
DEFINE_int32(compress_concurrency, 1, "concurrency of compressor");
DEFINE_int32(build_index_concurrency, 1, "concurrency of build_index");
DEFINE_int32(max_node_num, 5000, "compress max_node_num");
DEFINE_int32(min_node_num, 2, "compress min_node_num");
DEFINE_int32(mirror_k, 2, "threshold of building a mirror");
DEFINE_string(serialization_cmp_prefix, "",
              "where to load/store the compress graph's supernode serialization files");
DEFINE_int32(compress_type, 2, "0:mode2, 1:use metis, 2:louvain++");
DEFINE_string(message_type, "push", "push, pull");
DEFINE_double(compress_threshold, 1, "threshold for compression");
