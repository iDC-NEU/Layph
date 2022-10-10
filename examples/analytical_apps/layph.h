
#ifndef EXAMPLES_ANALYTICAL_APPS_LAYPH_H_
#define EXAMPLES_ANALYTICAL_APPS_LAYPH_H_

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include <grape/fragment/immutable_edgecut_fragment.h>
#include <grape/fragment/loader.h>
#include <grape/grape.h>
#include <grape/util.h>
#include <grape/worker/sum_sync_iter_worker.h>
#include <grape/worker/sum_batch_worker.h>
#include <grape/worker/sum_sync_traversal_worker.h>
#include <sys/stat.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "flags.h"
#include "pagerank/pagerank_layph.h"
#include "php/php_layph.h"
#include "sssp/sssp_auto.h"
#include "sssp/sssp_layph.h"
#include "bfs/bfs_layph.h"
#include "timer.h"

namespace grape {

void Init() {
  if (FLAGS_vfile.empty() || FLAGS_efile.empty()) {
    LOG(FATAL) << "Please assign input vertex/edge files.";
  }

  if (access(FLAGS_out_prefix.c_str(), 0) != 0) {
    mkdir(FLAGS_out_prefix.c_str(), 0777);
  }

  InitMPIComm();
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);
  if (comm_spec.worker_id() == kCoordinatorRank) {
    VLOG(1) << "Workers of libgrape-lite initialized.";
  }
}

void Finalize() {
  FinalizeMPIComm();
  VLOG(1) << "Workers finalized.";
}

template <typename FRAG_T, typename APP_T>
void CreateAndQueryTypeOne(const CommSpec& comm_spec, const std::string efile,
                           const std::string& vfile,
                           const std::string& out_prefix,
                           const ParallelEngineSpec& spec) {
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();
  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(false, 0);
  SetSerialize(comm_spec, FLAGS_serialization_prefix, efile, vfile, graph_spec);

  auto fragment =
      LoadGraph<FRAG_T, SegmentedPartitioner<typename FRAG_T::oid_t>>(
          efile, vfile, comm_spec, graph_spec);
  auto app = std::make_shared<APP_T>();
  if (comm_spec.worker_id() == grape::kCoordinatorRank) {
    LOG(INFO) << "Iter worker";
  }

  if(FLAGS_message_type == "push"){
    SumSyncIterWorker<APP_T> worker(app, fragment);
    worker.Init(comm_spec, spec);
    worker.Query();
    if (!out_prefix.empty()) {
      std::ofstream ostream;
      std::string output_path =
          grape::GetResultFilename(out_prefix, fragment->fid());
      ostream.open(output_path);
      worker.Output(ostream);
      ostream.close();
    }
    worker.Finalize();
  } else{
    LOG(INFO) << "No this type: " << FLAGS_message_type;
  }
  fragment.reset();
}

template <typename FRAG_T, typename APP_T>
std::vector<typename APP_T::value_t> CreateAndQueryTypeTwo(
    const CommSpec& comm_spec, const std::string efile,
    const std::string& vfile, const std::string& out_prefix,
    const ParallelEngineSpec& spec) {
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();

  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(false, 0);
  SetSerialize(comm_spec, FLAGS_serialization_prefix, efile, vfile, graph_spec);

  auto fragment =
      LoadGraph<FRAG_T, SegmentedPartitioner<typename FRAG_T::oid_t>>(
          efile, vfile, comm_spec, graph_spec);
  auto app = std::make_shared<APP_T>();
  if (comm_spec.worker_id() == grape::kCoordinatorRank) {
    LOG(INFO) << "Trav worker";
  }
  SumSyncTraversalWorker<APP_T> worker(app, fragment);
  worker.Init(comm_spec, spec);
  worker.Query();

  if (!out_prefix.empty()) {
    std::ofstream ostream;
    std::string output_path =
        grape::GetResultFilename(out_prefix, fragment->fid());
    ostream.open(output_path);
    worker.Output(ostream);
    ostream.close();
    VLOG(1) << "Worker-" << comm_spec.worker_id()
            << " finished: " << output_path;
  }
  worker.Finalize();
  fragment.reset();
  return app->DumpResult();
}

template <typename FRAG_T, typename APP_T, typename... Args>
void CreateAndQuery(const CommSpec& comm_spec, const std::string& efile,
                    const std::string& efile_update, const std::string& vfile,
                    const std::string& out_prefix,
                    const ParallelEngineSpec& spec, Args... args) {
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();
  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(false, 0);
  SetSerialize(comm_spec, FLAGS_serialization_prefix, efile, vfile, graph_spec);
  using oid_t = typename FRAG_T::oid_t;

  auto fragment = LoadGraph<FRAG_T, SegmentedPartitioner<oid_t>>(
      efile, vfile, comm_spec, graph_spec);
  auto app = std::make_shared<APP_T>();
  auto worker = APP_T::CreateWorker(app);
  worker->Init(comm_spec, spec);
  worker->SetFragment(fragment);
  worker->Query(std::forward<Args>(args)...);

  if (!efile_update.empty()) {
    graph_spec = DefaultLoadGraphSpec();
    graph_spec.set_directed(FLAGS_directed);
    graph_spec.set_rebalance(false, 0);

    IncFragmentBuilder<FRAG_T> inc_fragment_builder(fragment);

    inc_fragment_builder.Init(efile_update);

    auto added_edges = inc_fragment_builder.GetAddedEdges();
    auto deleted_edges = inc_fragment_builder.GetDeletedEdges();

    fragment = inc_fragment_builder.Build();
    worker->SetFragment(fragment);
    worker->Inc(added_edges, deleted_edges);
  }

  if (!out_prefix.empty()) {
    std::ofstream ostream;
    std::string output_path =
        grape::GetResultFilename(out_prefix, fragment->fid());
    ostream.open(output_path);
    worker->Output(ostream);
    ostream.close();
    VLOG(1) << "Worker-" << comm_spec.worker_id()
            << " finished: " << output_path;
  }
  worker->Finalize();
  fragment.reset();
}

template <typename FRAG_T, typename APP_T, typename... Args>
std::vector<typename APP_T::context_t::data_t> CreateAndQueryBatch(
    const CommSpec& comm_spec, const std::string efile,
    const std::string& vfile, std::shared_ptr<FRAG_T>& fragment,
    const ParallelEngineSpec& spec, Args... args) {
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();
  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(false, 0);
  fragment = LoadGraph<FRAG_T, SegmentedPartitioner<typename FRAG_T::oid_t>>(
      efile, vfile, comm_spec, graph_spec);

  if (!FLAGS_efile_update.empty()) {
    IncFragmentBuilder<FRAG_T> fragment_builder(fragment, FLAGS_directed);

    fragment_builder.Init(FLAGS_efile_update);
    fragment = fragment_builder.Build();
  }

  auto app = std::make_shared<APP_T>();
  auto worker = APP_T::CreateWorker(app, fragment);
  worker->Init(comm_spec, spec);
  worker->Query(std::forward<Args>(args)...);

  std::vector<typename APP_T::context_t::data_t> result;
  auto& data = worker->GetContext()->data();

  for (const auto& v : fragment->InnerVertices()) {
    result.push_back(data[v]);
  }

  return result;
}

template <typename FRAG_T, typename APP_T>
void CreateAndQuerySumBatch(const CommSpec& comm_spec, const std::string efile,
                           const std::string& vfile,
                           const std::string& out_prefix,
                           const ParallelEngineSpec& spec) {
  timer_next("load graph");
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();
  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(false, 0);
  SetSerialize(comm_spec, FLAGS_serialization_prefix, efile, vfile, graph_spec);

  auto fragment =
      LoadGraph<FRAG_T, SegmentedPartitioner<typename FRAG_T::oid_t>>(
          efile, vfile, comm_spec, graph_spec);
  auto app = std::make_shared<APP_T>();
  timer_next("load application");
  if (comm_spec.worker_id() == grape::kCoordinatorRank) {
    LOG(INFO) << "CreateAndQuerySumBatch...";
  }

  LOG(INFO) << "#message_style: " << FLAGS_message_type;
  SumBatchWorker<APP_T> worker(app, fragment);
  worker.Init(comm_spec, spec);
  timer_next("run algorithm");
  worker.Query();
  timer_next("print output");

  if (!out_prefix.empty()) {
    std::ofstream ostream;
    std::string output_path =
        grape::GetResultFilename(out_prefix, fragment->fid());
    ostream.open(output_path);
    worker.Output(ostream);
    ostream.close();
    VLOG(1) << "Worker-" << comm_spec.worker_id()
            << " finished: " << output_path;
  }
  worker.Finalize();
  timer_end();
  fragment.reset();
}

void RunLayph() {
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  bool is_coordinator = comm_spec.worker_id() == kCoordinatorRank;
  timer_start(is_coordinator);

  std::string name = FLAGS_application;
  std::string efile = FLAGS_efile;
  std::string vfile = FLAGS_vfile;
  std::string out_prefix = FLAGS_out_prefix;
  auto spec = DefaultParallelEngineSpec();

  if (FLAGS_app_concurrency != -1) {
    spec.thread_num = FLAGS_app_concurrency;
    setWorkers(FLAGS_app_concurrency);
  }

  if (access(vfile.c_str(), 0) != 0) {
    LOG(ERROR) << "Can not access vfile, build oid set at runtime";
    vfile = "";
  }

  if (name == "pagerank") {
    using value_t = float; // float
    using GraphType =
        grape::ImmutableEdgecutFragment<int32_t, uint32_t, grape::EmptyType,
                                        grape::EmptyType, LoadStrategy::kBothOutIn>;
    using AppType = grape::PageRankLayph<GraphType, value_t>;
    CreateAndQueryTypeOne<GraphType, AppType>(comm_spec, efile, vfile,
                                              out_prefix, spec);
  } else if (name == "sssp") {
    using value_t = int32_t;
    using GraphType =
        grape::ImmutableEdgecutFragment<int32_t, uint32_t, grape::EmptyType,
                                        // float, LoadStrategy::kBothOutIn>; 
                                        uint16_t, LoadStrategy::kBothOutIn>;
    using AppType = grape::SSSPLayph<GraphType, value_t>;
    CreateAndQueryTypeTwo<GraphType, AppType>(
          comm_spec, efile, vfile, out_prefix, spec);

  } else if (name == "bfs") {
    using value_t = int32_t;
    using GraphType =
        grape::ImmutableEdgecutFragment<int32_t, uint32_t, grape::EmptyType,
                                        grape::EmptyType, LoadStrategy::kBothOutIn>;
    using AppType = grape::BFSLayph<GraphType, value_t>;
    CreateAndQueryTypeTwo<GraphType, AppType>(
          comm_spec, efile, vfile, out_prefix, spec);
  } else if (name == "php") {
    using value_t = float;
    using GraphType = grape::ImmutableEdgecutFragment<int32_t, uint32_t,
                                                      grape::EmptyType, uint16_t, LoadStrategy::kBothOutIn>;
    using AppType = grape::PHPLayph<GraphType, value_t>;
    CreateAndQueryTypeOne<GraphType, AppType>(comm_spec, efile, vfile,
                                              out_prefix, spec);
  } else {
    LOG(INFO) << "No this application: " << name;
  }
}
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_LAYPH_H_
