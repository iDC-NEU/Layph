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

#ifndef EXAMPLES_ANALYTICAL_APPS_FLAGS_H_
#define EXAMPLES_ANALYTICAL_APPS_FLAGS_H_

#include <gflags/gflags_declare.h>

DECLARE_bool(directed);
DECLARE_string(application);
DECLARE_string(efile);
DECLARE_string(vfile);
DECLARE_string(efile_update);
DECLARE_string(out_prefix);
DECLARE_string(jobid);
DECLARE_double(portion);

DECLARE_int64(sssp_source);
DECLARE_int64(php_source);
DECLARE_double(php_d);
DECLARE_double(php_tol);
DECLARE_int32(php_mr);
DECLARE_double(pr_d);
DECLARE_int32(pr_mr);
DECLARE_double(pr_tol);
DECLARE_double(pr_delta_sum);
DECLARE_int32(gcn_mr);
DECLARE_bool(cilk);
DECLARE_bool(verify);

DECLARE_bool(segmented_partition);

DECLARE_string(serialization_prefix);

DECLARE_int32(app_concurrency);

DECLARE_bool(debug);
DECLARE_double(termcheck_threshold);
DECLARE_bool(d2ud_weighted);
DECLARE_bool(compress);
DECLARE_bool(count_skeleton);
DECLARE_int32(compress_concurrency);
DECLARE_int32(build_index_concurrency);
DECLARE_int32(max_node_num);
DECLARE_int32(min_node_num);
DECLARE_int32(mirror_k);
DECLARE_string(serialization_cmp_prefix);
DECLARE_int32(compress_type);
DECLARE_string(message_type);
DECLARE_double(compress_threshold);

#endif  // EXAMPLES_ANALYTICAL_APPS_FLAGS_H_
