
#ifndef GRAPE_WORKER_SUM_SYNC_TRAVERSAL_WORKER_H_
#define GRAPE_WORKER_SUM_SYNC_TRAVERSAL_WORKER_H_

#include <grape/fragment/loader.h>

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>

#include "flags.h"
#include "grape/app/traversal_app_base.h"
#include "grape/communication/communicator.h"
#include "grape/communication/sync_comm.h"
#include "grape/graph/adj_list.h"
#include "grape/parallel/parallel_engine.h"
#include "grape/parallel/parallel_message_manager.h"
#include "timer.h"
#include "grape/fragment/trav_compressor.h"

namespace grape {

template <typename FRAG_T, typename VALUE_T>
class IterateKernel;

/**
 * @brief A Worker manages the computation cycle. DefaultWorker is a kind of
 * worker for apps derived from AppBase.
 *
 * @tparam APP_T
 */

template <typename APP_T>
class SumSyncTraversalWorker : public ParallelEngine {
  static_assert(std::is_base_of<TraversalAppBase<typename APP_T::fragment_t,
                                                 typename APP_T::value_t>,
                                APP_T>::value,
                "SumSyncTraversalWorker should work with App");

 public:
  using fragment_t = typename APP_T::fragment_t;
  using value_t = typename APP_T::value_t;
  using delta_t = typename APP_T::delta_t;
  using vertex_t = typename APP_T::vertex_t;
  using message_manager_t = ParallelMessageManager;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename APP_T::vid_t;
  using supernode_t = grape::SuperNodeForTrav<vertex_t, value_t, delta_t, vid_t>;
  using edata_t = typename fragment_t::edata_t;
  using adj_list_t = typename fragment_t::adj_list_t;
  using nbr_t = typename fragment_t::nbr_t;
  using nbr_index_t = Nbr<vid_t, delta_t>;
  using adj_list_index_t = AdjList<vid_t, delta_t>;

  SumSyncTraversalWorker(std::shared_ptr<APP_T> app,
                             std::shared_ptr<fragment_t>& graph)
      : app_(app), fragment_(graph) {}

  void Init(const CommSpec& comm_spec,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()) {
    fragment_->PrepareToRunApp(APP_T::message_strategy,
                               APP_T::need_split_edges);

    comm_spec_ = comm_spec;

    MPI_Barrier(comm_spec_.comm());
    messages_.Init(comm_spec_.comm());
    messages_.InitChannels(thread_num());
    communicator_.InitCommunicator(comm_spec.comm());

    InitParallelEngine(pe_spec);
    if (FLAGS_cilk) {
      LOG(INFO) << "cilk Thread num: " << getWorkers();
    }
    // allocate dependency arrays
    app_->Init(comm_spec_, fragment_);
    // init compressor
    if(FLAGS_compress){
      cpr_ = new TravCompressor<APP_T, supernode_t>(app_, fragment_);
      cpr_->init(comm_spec_, communicator_, pe_spec);
      cpr_->run();
      app_->reInit(cpr_->all_node_num); // for mirror node
      /* precompute supernode */
      cpr_->precompute_spnode(this->fragment_);
    }

  }

  void deltaCompute() {
    IncFragmentBuilder<fragment_t> inc_fragment_builder(fragment_,
                                                        FLAGS_directed);

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Parsing update file";
    }
    inc_fragment_builder.Init(FLAGS_efile_update);
    auto inner_vertices = fragment_->InnerVertices();
    auto outer_vertices = fragment_->OuterVertices();

    auto deleted_edges = inc_fragment_builder.GetDeletedEdgesGid();

    #if defined(DISTRIBUTED)
      LOG(INFO) << "Distributed vision...";
      std::unordered_set<vid_t> local_gid_set;
      for (auto v : fragment_->Vertices()) {
        local_gid_set.insert(fragment_->Vertex2Gid(v));
      }
    #else
      LOG(INFO) << "Single vision...";
    #endif

    auto vertices = fragment_->Vertices();
    DenseVertexSet<vid_t> curr_modified, next_modified, reset_vertices;

    curr_modified.Init(vertices);
    next_modified.Init(vertices);
    reset_vertices.Init(inner_vertices);  // Only used for counting purpose

    double reset_time = GetCurrentTime();

    size_t pair_num = deleted_edges.size();
    parallel_for(vid_t i = 0; i < pair_num; i++) {
      auto pair = deleted_edges[i];
      vid_t u_gid = pair.first, v_gid = pair.second;

      #if defined(DISTRIBUTED)
        if (local_gid_set.find(u_gid) != local_gid_set.end() &&
            fragment_->IsInnerGid(v_gid)) {
          vertex_t u, v;
          CHECK(fragment_->Gid2Vertex(u_gid, u));
          CHECK(fragment_->Gid2Vertex(v_gid, v));

          auto parent_gid = app_->DeltaParentGid(v);

          if (parent_gid == u_gid) {
            curr_modified.Insert(v);
          }
        }
      #else
        vertex_t u, v;
        CHECK(fragment_->Gid2Vertex(u_gid, u));
        CHECK(fragment_->Gid2Vertex(v_gid, v));

        auto parent_gid = app_->DeltaParentGid(v);
        if (parent_gid == u_gid) {
          curr_modified.Insert(v);
        }
      #endif
    }

    auto& channels = messages_.Channels();

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Resetting";
    }

    do {
      #if defined(DISTRIBUTED)
      messages_.StartARound();
      messages_.ParallelProcess<fragment_t, grape::EmptyType>(
          thread_num(), *fragment_,
          [&curr_modified](int tid, vertex_t v, const grape::EmptyType& msg) {
            curr_modified.Insert(v);
          });
      #endif

      // ForEachSimple(curr_modified, inner_vertices,
      ForEachCilkOfBitset(curr_modified, inner_vertices,
                    [this, &next_modified, &reset_vertices](int tid, vertex_t u) {
                      auto u_gid = fragment_->Vertex2Gid(u);
                      auto oes = fragment_->GetOutgoingAdjList(u);

                      auto out_degree = oes.Size();
                      auto it = oes.begin();
                      granular_for(j, 0, out_degree, (out_degree > 1024), {
                        auto& e = *(it + j);
                        auto v = e.neighbor;
                        if (app_->DeltaParentGid(v) == u_gid && u != v) { 
                          next_modified.Insert(v);
                        }
                      })

                      app_->values_[u] = app_->GetInitValue(u);
                      app_->deltas_[u] = app_->GetInitDelta(u); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
                      app_->CombineValueDelta(app_->values_[u], app_->deltas_[u]);
                      reset_vertices.Insert(u); // just count reset node!
                    });

      #if defined(DISTRIBUTED)
      ForEachCilkOfBitset(curr_modified, inner_vertices,
                    [this, &reset_vertices](int tid, vertex_t u) {
                      app_->values_[u] = app_->GetInitValue(u);
                      app_->deltas_[u] = app_->GetInitDelta(u); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
                      app_->CombineValueDelta(app_->values_[u], app_->deltas_[u]);

                      reset_vertices.Insert(u);
                    });
      #endif

      #if defined(DISTRIBUTED)
      ForEachCilkOfBitset(next_modified, outer_vertices,
              [&channels, this](int tid, vertex_t v) {
                grape::EmptyType dummy;
                channels[tid].SyncStateOnOuterVertex(*fragment_, v, dummy);
                app_->deltas_[v] = app_->GetInitDelta(v); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
              });
      messages_.FinishARound();
      if (next_modified.PartialEmpty(0, fragment_->GetVerticesNum()) {
        messages_.ForceContinue();
      }
      #endif

      curr_modified.Clear();
      curr_modified.Swap(next_modified);
    #if defined(DISTRIBUTED)
    } while (!messages_.ToTerminate());
    #else
    } while (curr_modified.ParallelCount(thread_num()) > 0);
    #endif

    LOG(INFO) << "#reset_time: " << (GetCurrentTime() - reset_time);
    print_active_edge("#reset");

    // #if defined(DISTRIBUTED)
    size_t n_reset = 0, local_n_reset = reset_vertices.Count();
    Communicator communicator;
    communicator.InitCommunicator(comm_spec_.comm());
    communicator.template Sum(local_n_reset, n_reset);
    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "# of reset vertices: " << n_reset << " reset percent: "
                << (float) n_reset / fragment_->GetTotalVerticesNum();
      LOG(INFO) << "Start a round from all vertices";
    }
    // #endif

    // We have to use hashmap to keep delta because the outer vertices may
    // change
    VertexArray<value_t, vid_t> values;
    VertexArray<delta_t, vid_t> deltas;

    values.Init(inner_vertices);
    deltas.Init(inner_vertices);

    for (auto v : inner_vertices) {
      values[v] = app_->values_[v];
      deltas[v] = app_->deltas_[v];
    }

    // fragment_ = inc_fragment_builder.Build();

    const std::shared_ptr<fragment_t>& new_graph = inc_fragment_builder.Build();
    if(FLAGS_compress){
      auto added_edges = inc_fragment_builder.GetAddedEdgesGid();
      cpr_->inc_run(deleted_edges, added_edges, new_graph);
      print_active_edge("#inc_run_cmpIndex");
    }
    fragment_ = new_graph;

    // Important!!! outer vertices may change, we should acquire it after new
    // graph is loaded
    outer_vertices = fragment_->OuterVertices();
    // Reset all states, active vertices will be marked in curr_modified_
    app_->Init(comm_spec_, fragment_);
    if (FLAGS_compress) {
      app_->reInit(cpr_->all_node_num); // for mirror node
    }

    // copy to new graph
    for (auto v : inner_vertices) {
      app_->values_[v] = values[v];
      app_->deltas_[v] = deltas[v];
    }

    // Start a round without any condition
    double resend_time = GetCurrentTime();
    vid_t inner_node_num = inner_vertices.end().GetValue() 
                           - inner_vertices.begin().GetValue();
    parallel_for(vid_t i = 0; i < inner_node_num; i++) {
      vertex_t u(i);
      auto& value = app_->values_[u];
      auto& delta = app_->deltas_[u];

      if (delta.value != app_->GetIdentityElement()) {
        app_->Compute(u, value, delta, next_modified);
      }
    }

    #if defined(DISTRIBUTED)
    messages_.StartARound();
    ForEach(
        next_modified, outer_vertices, [&channels, this](int tid, vertex_t v) {
          auto& delta_to_send = app_->deltas_[v];
          if (delta_to_send.value != app_->GetIdentityElement()) {
            channels[tid].SyncStateOnOuterVertex(*fragment_, v, delta_to_send);
          }
        });
    messages_.FinishARound();
    #endif
    app_->curr_modified_.Swap(next_modified);
    LOG(INFO) << "#resend_time: " << (GetCurrentTime() - resend_time);
    print_active_edge("#resend");

    LOG(INFO) << " app_->curr_modified_.size()=" << app_->curr_modified_.ParallelCount(thread_num());;
  }

    /**
   * Get the threshold by sampling
   * sample_size: Sample size
   * range: the range of the variable population
   * return threshold
   */
   value_t Scheduled(const vid_t sample_size, const VertexRange<vid_t>& range) {
     vid_t begin = range.begin().GetValue();
     vid_t end = range.end().GetValue();
     vid_t all_size = end - begin;
     if (all_size <= sample_size) {
       return app_->GetIdentityElement();
     } else {
       std::unordered_set<int> id_set;
       // random number generator
       std::mt19937 gen(time(0));
       std::uniform_int_distribution<> dis(0, all_size - 1); 
       // sample random pos, the sample reflect the whole data set more or less 
       std::vector<value_t> sample; 
       int i;
       for (i = 0; i < sample_size; i++) {
         int rand_pos = dis(gen);
         while (id_set.find(rand_pos) != id_set.end()) {
           rand_pos = dis(gen);
         }
         id_set.insert(rand_pos);
         vertex_t u(rand_pos);
         value_t pri = app_->GetPriority(u, app_->values_[u], app_->deltas_[u]);
         sample.emplace_back(fabs(pri));
       }
  
       sort(sample.begin(), sample.end());
       int cut_index = sample_size * (1 - FLAGS_portion);  // Select the threshold position 
       return sample[cut_index];
     }
   }

  void GetInnerIndex(){
    LOG(INFO) << "Build inner index's csr: source to inner_node";
    double start_time = GetCurrentTime();
    vid_t supernode_num = cpr_->supernodes_num;
    std::vector<size_t> degree(supernode_num+1, 0);
    source_nodes.resize(supernode_num);

    parallel_for (vid_t i = 0; i < supernode_num; i++) {
      supernode_t &spnode = cpr_->supernodes[i];
      source_nodes[i] = spnode.id;
      degree[i+1] = spnode.inner_delta.size();
    }

    for(vid_t i = 1; i <= supernode_num; i++) {
      degree[i] += degree[i-1];
    }
    size_t index_num = degree[supernode_num];
    is_iindex_offset_.resize(supernode_num+1);
    is_iindex_.resize(index_num);


    parallel_for(vid_t j = 0; j < supernode_num; j++){
      supernode_t &spnode = cpr_->supernodes[j];
      auto& oes = spnode.inner_delta;
      vid_t index = degree[j];
      is_iindex_offset_[j] = &is_iindex_[index];
      for (auto& oe : oes) {
        is_iindex_[index] = nbr_index_t(oe.first.GetValue(), oe.second);
        index++;
      }
    }
    is_iindex_offset_[supernode_num] = &is_iindex_[index_num];

    parallel_for (vid_t i = 0; i < supernode_num; ++i) {
      std::sort(is_iindex_offset_[i], is_iindex_offset_[i + 1],
              [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
              });
    }

    LOG(INFO) << "Build inner index's csr: cost_time=" << (GetCurrentTime() - start_time);
  }

  void first_step(VertexArray<value_t, vid_t>& values_temp,
                  VertexArray<delta_t, vid_t>& deltas_temp,
                  double& exec_time, bool is_inc = false) {
    double extra_all_time = GetCurrentTime();
    auto inner_vertices = fragment_->InnerVertices();
    vid_t inner_node_num = inner_vertices.end().GetValue() 
                            - inner_vertices.begin().GetValue();
    
    cpr_->get_nodetype(inner_node_num, node_type);

    if (is_inc == true) {
      double inc_pre_compute = GetCurrentTime();
      cpr_->inc_precompute_spnode_mirror(fragment_, node_type);
      inc_pre_compute = GetCurrentTime() - inc_pre_compute;
      LOG(INFO) << "#inc_pre_compute: " << inc_pre_compute;
    }

    cpr_->sketch2csr(inner_node_num, node_type, all_nodes, is_e_, is_e_offset_,
                              ib_e_, ib_e_offset_);

    {
      exec_time -= GetCurrentTime();
      // Update the source id to the new id
      vertex_t source;
      bool native_source = fragment_->GetInnerVertex(FLAGS_sssp_source, source);
      if (native_source) {
        app_->curr_modified_.Insert(source); // old source node
      }

      /* send one round */
      ForEachCilkOfBitset(
        app_->curr_modified_, fragment_->InnerVertices(), 
        [this](int tid, vertex_t u) {
          if (node_type[u.GetValue()] < 2) { // 0, 1
            auto& delta = app_->deltas_[u];
            if (delta.value != app_->GetIdentityElement()) {
              auto& value = app_->values_[u];
              app_->CombineValueDelta(value, delta);
              adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
              app_->Compute(u, value, delta, oes, app_->next_modified_);
            }
          } else if (node_type[u.GetValue()] < 3) { // 2
            auto& delta = app_->deltas_[u];
            if (delta.value != app_->GetIdentityElement()) {
              auto& value = app_->values_[u];
              app_->CombineValueDelta(value, delta);
              adj_list_index_t oes = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
              app_->ComputeByIndexDelta(u, value, delta, oes, app_->next_modified_);
            }
          } else if (node_type[u.GetValue()] < 4) { // 3
            auto& delta = app_->deltas_[u];
            if (delta.value != app_->GetIdentityElement()) {
              auto& value = app_->values_[u];
              app_->CombineValueDelta(value, delta);
              /* 1: bound node */
              adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
              app_->Compute(u, value, delta, oes_b, app_->next_modified_);
              /* 2: source node */
              adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
              app_->ComputeByIndexDelta(u, value, delta, oes_s, app_->next_modified_);
            }
          }
      });
      app_->next_modified_.Swap(app_->curr_modified_);
    }
    print_active_edge("#pre_exec");
  }

  void Query() {
    MPI_Barrier(comm_spec_.comm());

    int step = 0;
    bool batch_stage = true;
    double exec_time = 0;
    double corr_time = 0;
    bool compr_stage = FLAGS_compress; // true: supernode send
    VertexArray<value_t, vid_t> values_temp;
    VertexArray<delta_t, vid_t> deltas_temp;
    values_temp.Init(fragment_->InnerVertices());
    deltas_temp.Init(fragment_->InnerVertices());
    fid_t fid = fragment_->fid();
    auto vm_ptr = fragment_->vm_ptr();

    if (compr_stage == true) {
      first_step(values_temp, deltas_temp, exec_time, false);
    }

    if (compr_stage == false) {
      vertex_t source;
      bool native_source = fragment_->GetInnerVertex(FLAGS_sssp_source, source);
      if (native_source) {
        app_->curr_modified_.Insert(source);
      }
    }

    messages_.Start();

    // Run an empty round, otherwise ParallelProcess will stuck
    messages_.StartARound();
    messages_.InitChannels(thread_num());
    messages_.FinishARound();

    
    long long count = 0; // debug
    double clean_bitset_time = 0.d;
    value_t threshold;

    {
      app_->f_send_delta_num = 0;
      app_->node_update_num = 0;
      app_->touch_nodes.ParallelClear(8);
    }

    while (true) {
      exec_time -= GetCurrentTime();
      ++step;

      auto inner_vertices = fragment_->InnerVertices();
      auto outer_vertices = fragment_->OuterVertices();

      messages_.StartARound();
      app_->next_modified_.ParallelClear(thread_num()); 

      {
        messages_.ParallelProcess<fragment_t, DependencyData<vid_t, value_t>>(
            thread_num(), *fragment_,
            [this](int tid, vertex_t v,
                   const DependencyData<vid_t, value_t>& msg) {
              if (app_->AccumulateToAtomic(v, msg)) {
                app_->curr_modified_.Insert(v); 
              }
            });
      }

      if (FLAGS_cilk) {
        if(compr_stage == false){
          ForEachCilkOfBitset(
              app_->curr_modified_, inner_vertices, [this, &compr_stage, &count, &step](int tid, vertex_t u) {
                auto& value = app_->values_[u];
                auto last_value = value;
                // We don't cleanup delta with identity element, since we expect
                // the algorithm is monotonic
                auto& delta = app_->deltas_[u];

                if (app_->CombineValueDelta(value, delta)) {
                  app_->Compute(u, last_value, delta, app_->next_modified_);
                }
              });
        }
        if (compr_stage) {
          ForEachCilkOfBitset(
            app_->curr_modified_, fragment_->InnerVertices(), 
            [this](int tid, vertex_t u) {
              if (node_type[u.GetValue()] < 2) { // 0, 1
                auto& delta = app_->deltas_[u];
                auto& value = app_->values_[u];
                if (app_->CombineValueDelta(value, delta)) {
                  adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes, app_->next_modified_);
                }
              } else if (node_type[u.GetValue()] < 3) { // 2
                auto& delta = app_->deltas_[u];
                auto& value = app_->values_[u];
                if (app_->CombineValueDelta(value, delta)) {
                  adj_list_index_t oes = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                  app_->ComputeByIndexDelta(u, value, delta, oes, app_->next_modified_);
                }
              } else if (node_type[u.GetValue()] < 4) { // 3
                auto& delta = app_->deltas_[u];
                auto& value = app_->values_[u];
                if (app_->CombineValueDelta(value, delta)) {
                  /* 1: bound node */
                  adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes_b, app_->next_modified_);
                  /* 2: source node */
                  adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                  app_->ComputeByIndexDelta(u, value, delta, oes_s, app_->next_modified_);
                }
              }
          });
        }
      }
       
      auto& channels = messages_.Channels();

      // send local delta to remote
      ForEach(app_->next_modified_, outer_vertices,
              [&channels, vm_ptr, fid, this](int tid, vertex_t v) {
                auto& delta_to_send = app_->deltas_[v];

                if (delta_to_send.value != app_->GetIdentityElement()) {
                  vid_t& v_parent_gid = delta_to_send.parent_gid;
                  fid_t v_fid = vm_ptr->GetFidFromGid(v_parent_gid);
                  if (v_fid == fid) {
                    v_parent_gid = newGid2oldGid[v_parent_gid];
                  }
                  channels[tid].SyncStateOnOuterVertex(*fragment_, v,
                                                       delta_to_send);
                }
              });

      if (!app_->next_modified_.PartialEmpty(0, fragment_->GetInnerVerticesNum())) {
        messages_.ForceContinue();
      }

      VLOG(1) << "[Worker " << comm_spec_.worker_id()
              << "]: Finished IterateKernel - " << step;
      messages_.FinishARound();

      // app_->next_modified_.Swap(app_->curr_modified_);

      exec_time += GetCurrentTime();

      bool terminate = messages_.ToTerminate();

      if (terminate) {
        if(compr_stage){
          print_active_edge("#globalCompt");
          compr_stage = false;
          corr_time -= GetCurrentTime();

          // supernode send by inner_delta
          vertex_t source;
          bool native_source = fragment_->GetInnerVertex(FLAGS_sssp_source, source);
          // #pragma cilk grainsize = 16
          parallel_for(vid_t j = 0; j < cpr_->supernodes_num; j++){
            supernode_t &spnode = cpr_->supernodes[j];
            auto u = spnode.id;
            bool is_mirror = false;
            if (u.GetValue() >= cpr_->old_node_num) {
              is_mirror = true;
              u = cpr_->mirrorid2vid[u];
            }
            auto& value = app_->values_[u];
            if (value != app_->GetIdentityElement()) { 
              auto& delta = app_->deltas_[u];
              vid_t spid = cpr_->id2spids[u];
              vertex_t p;
              fragment_->Gid2Vertex(delta.parent_gid, p);
              if (is_mirror == true || spid != cpr_->id2spids[p] || (native_source && source == p)) { // Only nodes that depend on external nodes need to send
                auto& oes = spnode.inner_delta;
                app_->ComputeByIndexDelta(u, value, delta, oes, app_->next_modified_);  
              }
            }
          }
          corr_time += GetCurrentTime();
          LOG(INFO) << "#assignment_time: " << corr_time;
          print_active_edge("#localAss");
          app_->next_modified_.Swap(app_->curr_modified_); 
        }

        if (batch_stage) {
          batch_stage = false;

          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "#iter step: " << step;
            LOG(INFO) << "#Batch time: " << exec_time;
            LOG(INFO) << "#clean_bitset_time: " << clean_bitset_time;
            print_active_edge("#Batch");
          }
          exec_time = 0;
          corr_time = 0;
          step = 1;

          if (!FLAGS_efile_update.empty()) {
            LOG(INFO) << "-------------------------------------------------------------------";
            LOG(INFO) << "--------------------------INC COMPUTE------------------------------";
            LOG(INFO) << "-------------------------------------------------------------------";
            compr_stage = FLAGS_compress; // use supernode
            deltaCompute();  // reload graph
            if (compr_stage == true) {
              first_step(values_temp, deltas_temp, exec_time, true);
            }
            continue; 

          } else {
            LOG(ERROR) << "Missing efile_update or efile_updated";
            break;
          }
        } else {
          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "#Inc iter step: " << step;
            LOG(INFO) << "#Inc time: " << exec_time << " sec";
            print_active_edge("#curr");
            app_->f_send_delta_num = 0;
            app_->node_update_num = 0;
            app_->touch_nodes.ParallelClear(8);
          }
          break;
        }
      }

      app_->next_modified_.Swap(app_->curr_modified_);
    }

    // Analysis result
    double d_sum = 0;
    vertex_t source;
    bool native_source =
        fragment_->GetInnerVertex(FLAGS_sssp_source, source);
    size_t visited_num = 0;
    vid_t max_id = native_source ? source.GetValue() : 0;
    for (auto v : fragment_->InnerVertices()) {
      if (app_->values_[v] != app_->GetIdentityElement()) {
        d_sum += app_->values_[v];
        visited_num += 1;
        if (app_->values_[v] > app_->values_[vertex_t(max_id)]) {
          max_id = v.GetValue();
        }
      }
    }
    LOG(INFO) << "#visited_num: " << visited_num;
    LOG(INFO) << "#visited_rate: " << (visited_num * 1.0 / fragment_->GetVerticesNum() );

    MPI_Barrier(comm_spec_.comm());
  }

  void print_indegree_outdegree(std::vector<vertex_t> &node_set){
    LOG(INFO) << "--------------------------------------------------";
    LOG(INFO) << " node_set.size=" << node_set.size();
    {
      std::ofstream fout("./out_edge");
      for (auto u : node_set) {
        vid_t i = u.GetValue();
        std::unordered_map<vid_t, size_t> edge_count;
        vid_t spids = cpr_->id2spids[u];
        for(auto e : fragment_->GetOutgoingAdjList(u)) {
          vid_t to_ids = cpr_->id2spids[e.neighbor];
          if(to_ids != spids){
            edge_count[to_ids] += 1;
          }
        }
        fout << i << ":";
        for(const auto& pair : edge_count) {
          fout << " " << pair.second;
        }
        fout << "\n";
      }
      fout.close();
    }
    {
      std::ofstream fout("./in_edge");
      for (auto u : node_set) {
        vid_t i = u.GetValue();
        std::unordered_map<vid_t, size_t> edge_count;
        vid_t spids = cpr_->id2spids[u];
        for(auto e : fragment_->GetIncomingAdjList(u)) {
          vid_t to_ids = cpr_->id2spids[e.neighbor];
          if(to_ids != spids){
            edge_count[to_ids] += 1;
          }
        }
        fout << i << ":";
        for(const auto& pair : edge_count) {
          fout << " " << pair.second;
        }
        fout << "\n";
      }
      fout.close();
    }
    LOG(INFO) << "finish write.2.. ";
  }

  void print_result(std::string position = ""){
    auto inner_vertices = fragment_->InnerVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    LOG(INFO) << "-----------result---s------------in " << position;
    for (auto v : inner_vertices) {
      vertex_t p;
      LOG(INFO) << "oid=" << fragment_->GetId(v) << " id=" << v.GetValue() 
                << ": value=" << values[v] << " delta=" << deltas[v].value 
                << " oid=" << fragment_->GetId(p) << std::endl;
    }
    LOG(INFO) << "-----------result---e------------";
  }
  
  void check_result(std::string position = ""){
    auto inner_vertices = fragment_->InnerVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    double value_sum = 0;
    double delta_sum = 0;
    LOG(INFO) << "----------check_result in " << position;
    for (auto v : inner_vertices) {
      value_sum += values[v];
      delta_sum += deltas[v].value;
    }
    printf("---value_sum=%.10lf\n", value_sum);
    printf("---delta_sum=%.10lf\n", delta_sum);
  }

  void Output(std::ostream& os) {
    auto inner_vertices = fragment_->InnerVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;

    for (auto v : inner_vertices) {
      os << fragment_->GetId(v) << " " << values[v] << " " << deltas[v].parent_gid << std::endl;
    }
    return ;
    // Write hypergraph to file
    {
      if (FLAGS_compress == false) {
        return ;
      }
      LOG(INFO) << "write supergraph...";
      long long edge_num = 0;
      for (auto u : inner_vertices) {
        char type = node_type[u.GetValue()];
        if (type == 0 || type == 1) {
          adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
          edge_num += oes.Size();
          for (auto e : oes) {
            os << fragment_->GetId(u) << " "
               << fragment_->GetId(e.neighbor) 
               << " " << e.data << std::endl;
          }
        } else if (type == 2) {
          adj_list_index_t oes = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
          edge_num += oes.Size();
          for (auto e : oes) {
            os << fragment_->GetId(u) << " " 
               << fragment_->GetId(e.neighbor) 
               << " " << e.data.value << std::endl;
          }
        } else if (type == 3) {
          adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
          edge_num += oes_b.Size();
          for (auto e : oes_b) {
            os << fragment_->GetId(u) << " " 
               << fragment_->GetId(e.neighbor)
               << " " << e.data << std::endl;
          }
          // os << "----------\n";
          adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
          edge_num += oes_s.Size();
          for (auto e : oes_s) {
            os << fragment_->GetId(u) << " " 
               << fragment_->GetId(e.neighbor)
               << " " << e.data.value << std::endl;
          }
        }
      }
      LOG(INFO) << "edge_num=" << edge_num;
    }
  }

  void Finalize() { messages_.Finalize(); }

  void print_active_edge(std::string position = "") {
    #ifdef COUNT_ACTIVE_EDGE_NUM  
      LOG(INFO) << position << "_f_index_count_num: " << app_->f_index_count_num;
      LOG(INFO) << position << "_f_send_delta_num: " << app_->f_send_delta_num;
      app_->f_index_count_num = 0;
      app_->f_send_delta_num = 0;
    #endif
  }


 private:
  std::shared_ptr<APP_T> app_;
  std::shared_ptr<fragment_t> fragment_;
  message_manager_t messages_;
  Communicator communicator_;
  CommSpec comm_spec_;
  TravCompressor<APP_T, supernode_t>* cpr_;
  /* source to inner_node: index */
  std::vector<vertex_t> source_nodes; // source: type2 + type3
  Array<nbr_index_t, Allocator<nbr_index_t>> is_iindex_;
  Array<nbr_index_t*, Allocator<nbr_index_t*>> is_iindex_offset_;
  /* source to in_bound_node: index */
  Array<nbr_index_t, Allocator<nbr_index_t>> is_e_;
  Array<nbr_index_t*, Allocator<nbr_index_t*>> is_e_offset_;
  /* in_bound_node to out_bound_node: original edge */
  Array<nbr_t, Allocator<nbr_t>> ib_e_;
  Array<nbr_t*, Allocator<nbr_t*>> ib_e_offset_;
  /* each type of vertices */
  std::vector<std::vector<vertex_t>> all_nodes;
  std::vector<char> node_type; // all node's types, 0:out node, 1:bound node, 2:source node, 3:belong 1 and 2 at the same time, 4:inner node that needn't send message.
  Array<vid_t, Allocator<vid_t>> oldId2newId; // renumber all internal vertices
  Array<vid_t, Allocator<vid_t>> newId2oldId; // renumber all internal vertices
  Array<vid_t, Allocator<vid_t>> oldGid2newGid; // renumber all internal vertices
  Array<vid_t, Allocator<vid_t>> newGid2oldGid; // renumber all internal vertices
  std::vector<vid_t> node_range; // 0-1-2-3-4
  // DenseVertexSet<vid_t> curr_modified_new, next_modified_new;
};

}  // namespace grape

#endif