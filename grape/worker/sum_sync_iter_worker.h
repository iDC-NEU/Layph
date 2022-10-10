#ifndef GRAPE_WORKER_SUM_SYNC_ITER_WORKER_H_
#define GRAPE_WORKER_SUM_SYNC_ITER_WORKER_H_

#include <grape/fragment/loader.h>

#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "flags.h"
#include "grape/app/layph_app_base.h"
#include "grape/communication/communicator.h"
#include "grape/communication/sync_comm.h"
#include "grape/fragment/inc_fragment_builder.h"
#include "grape/graph/adj_list.h"
#include "grape/parallel/default_message_manager.h"
#include "grape/parallel/parallel.h"
#include "grape/parallel/parallel_engine.h"
#include "timer.h"
#include "grape/fragment/iter_compressor.h"

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
class SumSyncIterWorker : public ParallelEngine {
  static_assert(std::is_base_of<IterateKernel<typename APP_T::fragment_t,
                                              typename APP_T::value_t>,
                                APP_T>::value,
                "SumSyncIterWorker should work with App");

 public:
  using fragment_t = typename APP_T::fragment_t;
  using value_t = typename APP_T::value_t;
  using vertex_t = typename APP_T::vertex_t;
  using message_manager_t = ParallelMessageManager;
  using vid_t = typename APP_T::vid_t;
  using supernode_t = grape::SuperNodeForIter<vertex_t, value_t, vid_t>;
  using edata_t = typename fragment_t::edata_t;
  using adj_list_t = typename fragment_t::adj_list_t;
  using nbr_t = typename fragment_t::nbr_t;
  using nbr_index_t = Nbr<vid_t, value_t>;
  using adj_list_index_t = AdjList<vid_t, value_t>;

  SumSyncIterWorker(std::shared_ptr<APP_T> app,
                        std::shared_ptr<fragment_t>& graph)
      : app_(app), graph_(graph) {}

  void Init(const CommSpec& comm_spec,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()) {
    graph_->PrepareToRunApp(APP_T::message_strategy, APP_T::need_split_edges);

    comm_spec_ = comm_spec;

    MPI_Barrier(comm_spec_.comm());

    messages_.Init(comm_spec_.comm());
    messages_.InitChannels(thread_num());
    communicator_.InitCommunicator(comm_spec.comm());
    terminate_checking_time_ = 0;

    InitParallelEngine(pe_spec);
    LOG(INFO) << "Thread num: " << thread_num();

    app_->Init(comm_spec_, *graph_, false);
    app_->iterate_begin(*graph_);

    // init compressor
    if(FLAGS_compress){
      cpr_ = new IterCompressor<APP_T, supernode_t>(app_, graph_);
      cpr_->init(comm_spec_, communicator_, pe_spec);
      cpr_->run();
      app_->reInit(cpr_->all_node_num, *graph_); // for mirror node
      // cpr_->statistic();
    }
  }

   value_t Scheduled(int sample_size) {
     vid_t all_size = graph_->GetInnerVerticesNum();
     if (all_size <= sample_size) {
       return 0;
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
         value_t pri;
         app_->priority(pri, app_->values_[u], app_->deltas_[u]);
         sample.emplace_back(fabs(pri));
       }
  
       sort(sample.begin(), sample.end());
       int cut_index = sample_size * (1 - FLAGS_portion);
       return sample[cut_index];
     }
   }

  void AmendValue(int type) {
    MPI_Barrier(comm_spec_.comm());

    auto inner_vertices = graph_->InnerVertices();
    auto outer_vertices = graph_->OuterVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;

    {
      messages_.StartARound();
      ForEach(inner_vertices, [this, type, &values](int tid, vertex_t u) {
        auto& value = values[u];
        auto delta = type * value;
        auto oes = graph_->GetOutgoingAdjList(u);

        app_->g_function(*graph_, u, value, delta, oes);
      });

      auto& channels = messages_.Channels();

      ForEach(outer_vertices, [this, &deltas, &channels](int tid, vertex_t v) {
        auto& delta_to_send = deltas[v];

        if (delta_to_send != app_->default_v()) {
          channels[tid].template SyncStateOnOuterVertex<fragment_t, value_t>(
              *graph_, v, delta_to_send);
          delta_to_send = app_->default_v();
        }
      });
      messages_.FinishARound();

      messages_.StartARound();
      messages_.template ParallelProcess<fragment_t, value_t>(
          thread_num(), *graph_,
          [this](int tid, vertex_t v, value_t received_delta) {
            app_->accumulate_atomic(app_->deltas_[v], received_delta);
          });
      messages_.FinishARound();
    }
    MPI_Barrier(comm_spec_.comm());
  }

  void AmendValue_active(int type, std::unordered_set<vertex_t>& actives_nodes) {
    auto inner_vertices = graph_->InnerVertices();
    auto outer_vertices = graph_->OuterVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;

    {
      auto it = actives_nodes.begin();
      for (auto u : actives_nodes) {
        // auto& u = *(it + j);
        auto& value = values[u];
        auto delta = type * value;
        auto oes = graph_->GetOutgoingAdjList(u);

        app_->g_function(*graph_, u, value, delta, oes);
      }
    }
  }

  void reloadGraph() {
    IncFragmentBuilder<fragment_t> inc_fragment_builder(graph_);

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Parsing update file";
    }
    inc_fragment_builder.Init(FLAGS_efile_update);
    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Building new graph" << " old_graph_edgeNum=" << (graph_->GetEdgeNum()/2);
    }

    auto deleted_edges = inc_fragment_builder.GetDeletedEdgesGid();
    auto added_edges = inc_fragment_builder.GetAddedEdgesGid();
    LOG(INFO) << "deleted_edges_num=" << deleted_edges.size() << " added_edges_num=" << added_edges.size();

    //-----------------------------get active nodes-----------------------------
    double resivition_time_0 = GetCurrentTime();
    #if defined(DISTRIBUTED)
      LOG(INFO) << "Distributed vision...";
      std::unordered_set<vid_t> local_gid_set;
      for (auto v : fragment_->Vertices()) {
        local_gid_set.insert(fragment_->Vertex2Gid(v));
      }
    #else
      LOG(INFO) << "Single vision...";
    #endif
    size_t del_pair_num = deleted_edges.size();
    size_t add_pair_num = added_edges.size();
    std::unordered_set<vertex_t> actives_nodes; // vid set of origin node in edge
    VertexArray<bool, vid_t> is_update;
    is_update.Init(graph_->InnerVertices()); // is inner vertieces
    actives_nodes.reserve(del_pair_num+add_pair_num);
    LOG(INFO) << " actives_nodes.size=" << actives_nodes.size();
    for(vid_t i = 0; i < del_pair_num; i++) {
      auto pair = deleted_edges[i];
      vid_t u_gid = pair.first;

      #if defined(DISTRIBUTED)
        if (local_gid_set.find(u_gid) != local_gid_set.end()) {
          vertex_t u;
          CHECK(graph_->Gid2Vertex(u_gid, u));
          actives_nodes.insert(u);
          is_update[u] = true;
        }
      #else
        vertex_t u;
        CHECK(graph_->Gid2Vertex(u_gid, u));
        actives_nodes.insert(u);
        is_update[u] = true;
      #endif
    }
    LOG(INFO) << " actives_nodes.size=" << actives_nodes.size();
    for(vid_t i = 0; i < add_pair_num; i++) {
      auto pair = added_edges[i];
      vid_t u_gid = pair.first;

      #if defined(DISTRIBUTED)
        if (local_gid_set.find(u_gid) != local_gid_set.end()) {
          vertex_t u;
          CHECK(graph_->Gid2Vertex(u_gid, u));
          actives_nodes.insert(u);
          is_update[u] = true;
        }
      #else
        vertex_t u;
        CHECK(graph_->Gid2Vertex(u_gid, u));
        actives_nodes.insert(u);
        is_update[u] = true;
      #endif
    }
    // recycled value on the old graph
    if(FLAGS_compress){
      AmendValue_active(-1, actives_nodes);
    } else {
      resivition_time_0 = GetCurrentTime();
      AmendValue(-1);
    }
    LOG(INFO) << "#resivition_time_0: " << (GetCurrentTime() - resivition_time_0);

    VertexArray<value_t, vid_t> values, deltas;
    auto iv = graph_->InnerVertices();
    {
      // Backup values on old graph
      values.Init(iv);
      deltas.Init(iv);

      for (auto v : iv) {
        values[v] = app_->values_[v];
        deltas[v] = app_->deltas_[v];
      }
    }

    app_->rebuild_graph(*graph_);
    const std::shared_ptr<fragment_t>& new_graph = inc_fragment_builder.Build();
    app_->iterate_begin(*new_graph);

    print_active_edge("#AmendValue-1");
    if(FLAGS_compress){
      cpr_->inc_run(deleted_edges, added_edges, new_graph, is_update);
      print_active_edge("#inc_run_cmpIndex");
    }
    graph_ = new_graph;

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "New graph loaded" << " new_graph_edgeNum=" << (graph_->GetEdgeNum()/2);
    }
    app_->Init(comm_spec_, *graph_, false);
    if (FLAGS_compress) {
      app_->reInit(cpr_->all_node_num, *graph_); // for mirror node
    }
    {
      // Copy values to new graph
      for (auto v : iv) {
        app_->values_[v] = values[v];
        app_->deltas_[v] = deltas[v];
      }
    }

    // reissue value on the new graph
    double resivition_time_1 = GetCurrentTime();
    if(FLAGS_compress){
      AmendValue_active(1, actives_nodes);
    } else {
      AmendValue(1);
    }
    LOG(INFO) << "#resivition_time_1: " << (GetCurrentTime() - resivition_time_1);
    print_active_edge("#AmendValue+1");
  }


  void first_step(bool is_inc) {
    auto inner_vertices = graph_->InnerVertices();
    vid_t inner_node_num = inner_vertices.end().GetValue() 
                            - inner_vertices.begin().GetValue();
    auto new_node_range = VertexRange<vid_t>(0, cpr_->all_node_num);
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    auto& is_e_ = cpr_->is_e_;
    auto& is_e_offset_ = cpr_->is_e_offset_;
    auto& ib_e_ = cpr_->ib_e_;
    auto& ib_e_offset_ = cpr_->ib_e_offset_;
    auto& sync_e_ = cpr_->sync_e_;
    auto& sync_e_offset_ = cpr_->sync_e_offset_;

    bound_node_values.clear();
    bound_node_values.Init(new_node_range, app_->default_v());
    spnode_datas.clear();
    spnode_datas.Init(new_node_range, app_->default_v());
    
    cpr_->get_nodetype_mirror(inner_node_num, node_type);

    all_nodes.clear();
    all_nodes.resize(7);
    for(vid_t i = inner_vertices.begin().GetValue(); 
      i < inner_vertices.end().GetValue(); i++) {
        all_nodes[node_type[i]].emplace_back(vertex_t(i));
    }

    // cpr_->sketch2csr_divide(node_type);
    cpr_->sketch2csr_mirror(node_type);

    /* precompute supernode */
    // timer_next("pre compute");
    double pre_compute = GetCurrentTime();
    cpr_->precompute_spnode_one(this->graph_, is_inc);
    parallel_for(vid_t j = 0; j < cpr_->cluster_ids.size(); j++){
      /* send to out nodes by bound edges */
      std::vector<vertex_t> &node_set = cpr_->supernode_ids[j];
      for(auto u : node_set) {
        if (cpr_->supernode_out_bound[u.GetValue()]) {
          value_t value = values[u];
          vid_t i = u.GetValue();
          {
            /* 3. bound node: some node in is_spnode_in and supernode_out_bound at the same time. */
            auto oes = graph_->GetOutgoingAdjList(u);
            adj_list_t adj = adj_list_t(ib_e_offset_[i], 
                                        ib_e_offset_[i+1]);
            app_->g_function(*graph_, u, value, value, oes, adj);  // out degree neq now adjlist.size
            /* in-master */
            if (value != app_->default_v()) {
              adj_list_t sync_adj = adj_list_t(sync_e_offset_[i], 
                                                sync_e_offset_[i+1]);
              for (auto e : sync_adj) {
                vertex_t v = e.neighbor;
                // sync to mirror v
                app_->accumulate_atomic(deltas[v], value);
                // active mirror v
                value_t& old_delta = deltas[v];
                auto delta = atomic_exch(old_delta, app_->default_v());
                auto& value = values[v];
                adj_list_index_t adj = adj_list_index_t(is_e_offset_[v.GetValue()], 
                                                        is_e_offset_[v.GetValue()+1]);
                app_->g_index_function(*graph_, v, value, delta, adj, 
                                        bound_node_values);
                app_->accumulate_atomic(spnode_datas[v], delta);
              }
            }
          }
        }
      }
      /* out-mirror to master */
      for (auto u : cpr_->cluster_out_mirror_ids[j]) {
        vertex_t v = cpr_->mirrorid2vid[u];
        auto delta = atomic_exch(deltas[u], app_->default_v());
        this->app_->accumulate_atomic(deltas[v], delta);
      }
    }
    cpr_->precompute_spnode_two();
    pre_compute = GetCurrentTime() - pre_compute;
    LOG(INFO) << "#pre_compute_" << int(is_inc) << ": " << pre_compute;
    print_active_edge("#pre_compute");
  }

  void Query() {
    MPI_Barrier(comm_spec_.comm());

    #ifdef COUNT_ACTIVE_EDGE_NUM
      LOG(INFO) << "============== open count active edge num ================";
    #endif

    if (FLAGS_debug) {
      volatile int i = 0;
      char hostname[256];
      gethostname(hostname, sizeof(hostname));
      printf("PID %d on %s ready for attach\n", getpid(), hostname);
      fflush(stdout);
      while (0 == i) {
        sleep(1);
      }
    }

    auto inner_vertices = graph_->InnerVertices();
    auto outer_vertices = graph_->OuterVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    VertexArray<value_t, vid_t> last_values;

    auto& is_e_ = cpr_->is_e_;
    auto& is_e_offset_ = cpr_->is_e_offset_;
    auto& ib_e_ = cpr_->ib_e_;
    auto& ib_e_offset_ = cpr_->ib_e_offset_;
    auto& sync_e_ = cpr_->sync_e_;
    auto& sync_e_offset_ = cpr_->sync_e_offset_;

    int step = 1;
    bool batch_stage = true;
    short int convergence_id = 0;
    bool compr_stage = FLAGS_compress; // true: supernode send


    last_values.Init(inner_vertices);
    value_t init_value_sum = 0;
    value_t init_delta_sum = 0;

    parallel_for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
      vertex_t v(i);
      last_values[v] = values[v];
    }

    if(compr_stage){
      first_step(false); // batch
    }

    LOG(INFO) << "compr_stage=" << compr_stage;

    double exec_time = 0;
    double corr_time = 0;
    double one_step_time = 0;
    value_t pri = 0;

    messages_.Start();

    // Run an empty round, otherwise ParallelProcess will stuck
    messages_.StartARound();
    messages_.InitChannels(thread_num());
    messages_.FinishARound();

    while (true) {
      ++step;
      exec_time -= GetCurrentTime();

      messages_.StartARound();
      auto& channels = messages_.Channels();

      {
        messages_.ParallelProcess<fragment_t, value_t>(
            thread_num(), *graph_,
            [this](int tid, vertex_t v, value_t received_delta) {
              app_->accumulate_atomic(app_->deltas_[v], received_delta);
            });
      }

      {
        if (FLAGS_cilk) {
#ifdef INTERNAL_PARALLEL
          LOG(FATAL) << "Layph is not compiled with -DUSE_CILK";
#endif
          if(compr_stage == false){
            parallel_for(vid_t i = inner_vertices.begin().GetValue();
                       i < inner_vertices.end().GetValue(); i++) {
              vertex_t u(i);
              value_t& old_delta = deltas[u];
              if (isChange(old_delta)) {
                auto& value = values[u];
                auto delta = atomic_exch(deltas[u], app_->default_v());
                auto oes = graph_->GetOutgoingAdjList(u);

                app_->g_function(*graph_, u, value, delta, oes);
                app_->accumulate_atomic(value, delta);
              }
            }
          }

          if(compr_stage){
            parallel_for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
              vertex_t u(i);
              switch (node_type[i]){
                case NodeType::SingleNode:
                  /* 1. out node */
                  {
                    value_t& old_delta = deltas[u];
                    if (isChange(old_delta)) {
                      auto delta = atomic_exch(old_delta, app_->default_v());
                      auto& value = values[u];
                      adj_list_t oes = adj_list_t(ib_e_offset_[i], 
                                                  ib_e_offset_[i+1]); 
                      app_->g_function(*graph_, u, value, delta, oes);
                      app_->accumulate_atomic(value, delta);
                    }
                  }
                  break;
                case NodeType::OnlyInNode:
                  /* 2. source node: source send message to inner_bound_node by inner_bound_index */
                  {
                    value_t& old_delta = deltas[u];
                    if (isChange(old_delta)) {
                      auto delta = atomic_exch(old_delta, app_->default_v());
                      auto& value = values[u];
                      adj_list_index_t adj = adj_list_index_t(is_e_offset_[i], 
                                                              is_e_offset_[i+1]);
                      app_->g_index_function(*graph_, u, value, delta, adj, 
                                                                bound_node_values);
                      app_->accumulate_atomic(spnode_datas[u], delta);
                    }
                  }
                  break;
                case NodeType::OnlyOutNode:
                  /* 3. bound node: some node in is_spnode_in and supernode_out_bound at the same time. */
                  {
                    value_t& old_delta = bound_node_values[u];
                    if (isChange(old_delta)) {
                      auto delta = atomic_exch(old_delta, app_->default_v());
                      auto& value = values[u];
                      auto oes = graph_->GetOutgoingAdjList(u);
                      adj_list_t adj = adj_list_t(ib_e_offset_[i], 
                                                  ib_e_offset_[i+1]);
                      app_->g_function(*graph_, u, value, delta, oes, adj);  // out degree neq now adjlist.size
                      app_->accumulate_atomic(value, delta);
                    }
                  }
                  break;
                case NodeType::BothOutInNode:
                  /* 2. source node: source send message to inner_bound_node by inner_bound_index */
                  {
                    value_t& old_delta = deltas[u];
                    if (isChange(old_delta)) {
                      auto delta = atomic_exch(old_delta, app_->default_v());
                      auto& value = values[u];
                      adj_list_index_t adj = adj_list_index_t(is_e_offset_[i], 
                                                                is_e_offset_[i+1]);
                      app_->g_index_function(*graph_, u, value, delta, adj, 
                                                                bound_node_values);
                      app_->accumulate_atomic(spnode_datas[u], delta);
                    }
                  }
                  /* 3. bound node: some node in is_spnode_in and supernode_out_bound at the same time. */
                  {
                    value_t& old_delta = bound_node_values[u];
                    if (isChange(old_delta)) {
                      auto delta = atomic_exch(old_delta, app_->default_v());
                      auto& value = values[u];
                      auto oes = graph_->GetOutgoingAdjList(u);
                      adj_list_t adj = adj_list_t(ib_e_offset_[i], 
                                                  ib_e_offset_[i+1]);
                      app_->g_function(*graph_, u, value, delta, oes, adj);  // out degree neq now adjlist.size
                      app_->accumulate_atomic(value, delta);
                    }
                  }
                  break;
                case NodeType::OutMaster:
                  {
                    /* 3. bound node: some node in is_spnode_in and supernode_out_bound at the same time. */
                    value_t& old_delta = bound_node_values[u];
                    if (isChange(old_delta)) {
                      auto delta = atomic_exch(old_delta, app_->default_v());
                      auto& value = values[u];
                      auto oes = graph_->GetOutgoingAdjList(u);
                      adj_list_t adj = adj_list_t(ib_e_offset_[i], 
                                                  ib_e_offset_[i+1]);
                      app_->g_function(*graph_, u, value, delta, oes, adj);  // out degree neq now adjlist.size
                      app_->accumulate_atomic(value, delta);
                      /* in-master */
                      if (delta != app_->default_v()) {
                        adj_list_t sync_adj = adj_list_t(sync_e_offset_[i], 
                                                        sync_e_offset_[i+1]);
                        for (auto e : sync_adj) {
                          vertex_t v = e.neighbor;
                          // sync to mirror v
                          app_->accumulate_atomic(deltas[v], delta);
                          // active mirror v
                          value_t& old_delta = deltas[v];
                          auto delta = atomic_exch(old_delta, app_->default_v());
                          auto& value = values[v];
                          adj_list_index_t adj = adj_list_index_t(
                                                  is_e_offset_[v.GetValue()], 
                                                  is_e_offset_[v.GetValue()+1]);
                          app_->g_index_function(*graph_, v, value, delta, adj, 
                                                  bound_node_values);
                          app_->accumulate_atomic(spnode_datas[v], delta);
                        }
                      }
                    }
                  }
                  break;
                case NodeType::BothOutInMaster:
                  /* 2. source node: source send message to inner_bound_node by inner_bound_index */
                  {
                    value_t& old_delta = deltas[u];
                    if (isChange(old_delta)) {
                      auto delta = atomic_exch(old_delta, app_->default_v());
                      auto& value = values[u];
                      adj_list_index_t adj = adj_list_index_t(is_e_offset_[i], 
                                                              is_e_offset_[i+1]);
                      app_->g_index_function(*graph_, u, value, delta, adj, 
                                                                bound_node_values);
                      app_->accumulate_atomic(spnode_datas[u], delta);
                    }
                  }
                  {
                    /* 3. bound node: some node in is_spnode_in and supernode_out_bound at the same time. */
                    value_t& old_delta = bound_node_values[u];
                    if (isChange(old_delta)) {
                      auto delta = atomic_exch(old_delta, app_->default_v());
                      auto& value = values[u];
                      auto oes = graph_->GetOutgoingAdjList(u);
                      adj_list_t adj = adj_list_t(ib_e_offset_[i], 
                                                  ib_e_offset_[i+1]);
                      app_->g_function(*graph_, u, value, delta, oes, adj);  // out degree neq now adjlist.size
                      app_->accumulate_atomic(value, delta);
                      /* in-master */
                      if (delta != app_->default_v()) {
                        adj_list_t sync_adj = adj_list_t(sync_e_offset_[i], 
                                                        sync_e_offset_[i+1]);
                        for (auto e : sync_adj) {
                          vertex_t v = e.neighbor;
                          // sync to mirror v
                          app_->accumulate_atomic(deltas[v], delta);
                          // active mirror v
                          value_t& old_delta = deltas[v];
                          auto delta = atomic_exch(old_delta, app_->default_v());
                          auto& value = values[v];
                          adj_list_index_t adj = adj_list_index_t(
                                                  is_e_offset_[v.GetValue()], 
                                                  is_e_offset_[v.GetValue()+1]);
                          app_->g_index_function(*graph_, v, value, delta, adj, 
                                                  bound_node_values);
                          app_->accumulate_atomic(spnode_datas[v], delta);
                        }
                      }
                    }
                  }
                  break;
              }
            }
            /* out-mirror sync to master */
            vid_t size = cpr_->all_out_mirror.size();
            parallel_for (vid_t i = 0; i < size; i++) {
              vertex_t u =cpr_->all_out_mirror[i];
              value_t& old_delta = bound_node_values[u];
              if (isChange(old_delta)) {
                vertex_t v = cpr_->mirrorid2vid[u];
                auto delta = atomic_exch(bound_node_values[u], app_->default_v()); // send spnode_datas
                this->app_->accumulate_atomic(deltas[v], delta);
              }
            }
          }
        }
      }

      {
        // send local delta to remote
        ForEach(outer_vertices, [this, &deltas, &channels](int tid,
                                                           vertex_t v) {
          auto& delta_to_send = deltas[v];

          if (delta_to_send != app_->default_v()) {
            channels[tid].template SyncStateOnOuterVertex<fragment_t, value_t>(
                *graph_, v, delta_to_send);
            delta_to_send = app_->default_v();
          }
        });
      }

      #ifdef DEBUG
        VLOG(1) << "[Worker " << comm_spec_.worker_id()
              << "]: Finished IterateKernel - " << step;
      #endif
      messages_.FinishARound();

      exec_time += GetCurrentTime();
      if (termCheck(last_values, values, compr_stage) || step > FLAGS_pr_mr) {
        app_->touch_nodes.clear();
        if(compr_stage){
          print_active_edge("#globalCompt");
          // timer_next("correct deviation");
          corr_time -= GetCurrentTime();
          parallel_for(vid_t i = 0; i < cpr_->all_node_num; i++) {
            vertex_t u(i);
            auto& delta = spnode_datas[u];
            if(delta != app_->default_v()){
              vid_t cid = cpr_->id2spids[u];
              vid_t c_node_num = cpr_->supernode_ids[cid].size();
              if(isChange(delta, c_node_num)){
                vid_t sp_id = cpr_->Fc_map[u];
                supernode_t &spnode = cpr_->supernodes[sp_id];
                auto& value = values[spnode.id];
                auto& oes_d = spnode.inner_delta;
                auto& oes_v = spnode.inner_value;
                app_->g_index_func_delta(*graph_, spnode.id, value, delta, oes_d); //If the threshold is small enough when calculating the index, it can be omitted here
                app_->g_index_func_value(*graph_, spnode.id, value, delta, oes_v);
                delta = app_->default_v();
              }
            }
          }
          corr_time += GetCurrentTime();
          LOG(INFO) << "#assignment_time: " << corr_time;
          print_active_edge("#localAss");
          compr_stage = false;
          // continue; // theoretically convergent
        }

        if (batch_stage) {
          batch_stage = false;

          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "#iter step: " << step;
            LOG(INFO) << "#Batch time: " << exec_time << " sec";
            print_active_edge("#Batch");
          }
          exec_time = 0;
          corr_time = 0;
          step = 0;
          convergence_id = 0;

          if (!FLAGS_efile_update.empty()) {
            LOG(INFO) << "----------------------------------------------------";
            LOG(INFO) << "------------------INC COMPUTE-----------------------";
            LOG(INFO) << "----------------------------------------------------";
            compr_stage = FLAGS_compress; // use supernode
            // timer_next("reloadGraph");
            reloadGraph();
            LOG(INFO) << "start inc...";
            // timer_next("inc algorithm");
            CHECK_EQ(inner_vertices.size(), graph_->InnerVertices().size());
            inner_vertices = graph_->InnerVertices();
            outer_vertices = graph_->OuterVertices();
            CHECK_EQ(values.size(), app_->values_.size());
            CHECK_EQ(deltas.size(), app_->deltas_.size());
            values = app_->values_;
            deltas = app_->deltas_;

            if(compr_stage){
              first_step(true);  // inc is true
            }
            continue;
          }
        } else {
          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "#Inc iter step: " << step;
            LOG(INFO) << "#Inc time: " << exec_time << " sec";
            print_active_edge("#curr");
          }
          break;
        }
      }
    }

    // Analysis result
    double d_sum = 0;
    vertex_t source;
    bool native_source =
        graph_->GetInnerVertex(FLAGS_sssp_source, source);
    vid_t max_id = native_source ? source.GetValue() : 0;
    for (auto v : graph_->InnerVertices()) {
      d_sum += app_->values_[v];
      if (app_->values_[v] > app_->values_[vertex_t(max_id)]) {
        max_id = v.GetValue();
      }
    }
    LOG(INFO) << "max_d[" << graph_->GetId(vertex_t(max_id)) << "]=" << app_->values_[vertex_t(max_id)];
    LOG(INFO) << "d_sum=" << d_sum;

    MPI_Barrier(comm_spec_.comm());
    if(compr_stage){
      delete cpr_;
    }
  }

  void Output(std::ostream& os) {
    auto inner_vertices = graph_->InnerVertices();
    auto& values = app_->values_;

    for (auto v : inner_vertices) {
      os << graph_->GetId(v) << " " << values[v] << std::endl;
    }
  }

  void print_result(){
    auto inner_vertices = graph_->InnerVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    LOG(INFO) << "-----------result---s------------";
    for (auto v : inner_vertices) {
      vertex_t p;
      LOG(INFO) << "oid=" << graph_->GetId(v) << " id=" << v.GetValue() 
                << ": value=" << values[v] << " delta=" << deltas[v];
    }
    LOG(INFO) << "-----------result---e------------";
  }

  void check_result(std::string position = ""){
    auto inner_vertices = graph_->InnerVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    double value_sum = 0;
    double delta_sum = 0;
    LOG(INFO) << "check_result in " << position;
    for (auto v : inner_vertices) {
      if (values[v] != app_->default_v()) {
        value_sum += values[v];
      }
      if (deltas[v] != app_->default_v()) {
        delta_sum += deltas[v];
      }
    }
    printf("value_sum=%.10lf\n", value_sum);
    printf("delta_sum=%.10lf\n", delta_sum);
  }

  void Finalize() { messages_.Finalize(); }

 private:
  bool termCheck(VertexArray<value_t, vid_t>& last_values,
                 VertexArray<value_t, vid_t>& values, bool compr_stage) {
    terminate_checking_time_ -= GetCurrentTime();
    auto vertices = graph_->InnerVertices();
    double diff_sum = 0, global_diff_sum;

    if (FLAGS_portion >= 1) {
      for (auto u : vertices) {
        diff_sum += fabs(app_->deltas_[u]);
      }
      LOG(INFO) << " use priority...";
    } else {
      for (auto u : vertices) {
        diff_sum += fabs(last_values[u] - values[u]);
        last_values[u] = values[u];
      }
    }

    communicator_.template Sum(diff_sum, global_diff_sum);

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Diff: " << global_diff_sum;
    }

    double bound_value = 0;
    if (FLAGS_compress == true) {
      for (auto u : vertices) {
        bound_value += bound_node_values[u];
      }
    }
    if (global_diff_sum < FLAGS_termcheck_threshold 
        && bound_value > FLAGS_termcheck_threshold) {
      LOG(INFO) << "---------------------------------";
      LOG(INFO) << "  bound_value=" << bound_value;
      LOG(INFO) << "---------------------------------";
    }

    terminate_checking_time_ += GetCurrentTime();
    return global_diff_sum < FLAGS_termcheck_threshold;
  }

  bool isChange(value_t delta, vid_t c_node_num=1) {
    if (FLAGS_portion >= 1) {
      if (std::fabs(delta) * c_node_num 
            > FLAGS_termcheck_threshold/graph_->GetVerticesNum()) {
        return true;
      } else {
        return false;
      }
    } else {
      return true;
    }
  }

  void print_active_edge(std::string position = "") {
    #ifdef COUNT_ACTIVE_EDGE_NUM
      LOG(INFO) << position << "_f_index_count_num: " << app_->f_index_count_num;
      LOG(INFO) << position << "_f_send_delta_num: " << app_->f_send_delta_num;
      app_->f_index_count_num = 0;
      app_->f_send_delta_num = 0;
    #endif
  }

  // VertexArray<vid_t, vid_t> spnode_ids;
  std::vector<char> node_type; // all node's types, 0:out node, 1:bound node, 2:source node, 3:belong 1 and 2 at the same time, 4:inner node that needn't send message.
  VertexArray<value_t, vid_t> bound_node_values;
  std::shared_ptr<APP_T> app_;
  std::shared_ptr<fragment_t>& graph_;
  message_manager_t messages_;
  Communicator communicator_;
  CommSpec comm_spec_;
  double terminate_checking_time_;
  IterCompressor<APP_T, supernode_t>* cpr_;
  // std::vector<value_t> spnode_datas;
  VertexArray<value_t, vid_t> spnode_datas{}; 
  /* each type of vertices */
  std::vector<std::vector<vertex_t>> all_nodes;

  class compare_priority {
   public:
    VertexArray<value_t, vid_t>& parent;

    explicit compare_priority(VertexArray<value_t, vid_t>& inparent)
        : parent(inparent) {}

    bool operator()(const vid_t a, const vid_t b) {
      return abs(parent[Vertex<unsigned int>(a)]) >
             abs(parent[Vertex<unsigned int>(b)]);
    }
  };
};

}  // namespace grape

#endif  // GRAPE_WORKER_SUM_SYNC_ITER_WORKER_H_
