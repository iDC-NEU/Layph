#ifndef GRAPE_WORKER_SUM_BATCH_WORKER_H_
#define GRAPE_WORKER_SUM_BATCH_WORKER_H_

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
class SumBatchWorker : public ParallelEngine {
  static_assert(std::is_base_of<IterateKernel<typename APP_T::fragment_t,
                                              typename APP_T::value_t>,
                                APP_T>::value,
                "SumBatchWorker should work with App");

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

  SumBatchWorker(std::shared_ptr<APP_T> app,
                        std::shared_ptr<fragment_t>& graph)
      : app_(app), graph_(graph) {}

  void Init(const CommSpec& comm_spec,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()) {
    graph_->PrepareToRunApp(APP_T::message_strategy, APP_T::need_split_edges);

    comm_spec_ = comm_spec;

    // 等待所有worker执行完毕
    MPI_Barrier(comm_spec_.comm());

    // 初始化发消息相关的buffer
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
      timer_next("statistic");
      cpr_->statistic();
    }
  }

  /**
   * 通过采样确定阈值来筛选数据
   * sample_size: 采样大小
   * return 阈值
   */
   value_t Scheduled(int sample_size) {
  //    auto& priority = app_->priority_;
     vid_t all_size = graph_->GetInnerVerticesNum();
     if (all_size <= sample_size) {
       return 0;
     } else {
       std::unordered_set<int> id_set;
       // random number generator
       std::mt19937 gen(time(0));
       std::uniform_int_distribution<> dis(0, all_size - 1);  // 给定范围 // 构造符合要求的随机数生成器
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
       int cut_index = sample_size * (1 - FLAGS_portion);  // 选择阈值位置 
       return sample[cut_index];
     }
   }

  /**
   * 用于图变换前后值的修正
   * type: -1表示回收(在旧图上)， 1表示重发(在新图上)
   *
   */
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
        auto delta = type * value;  // 发送回收值/发送补发值
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
            app_->accumulate(app_->deltas_[v], received_delta);
          });
      // default_work,同步一轮
      messages_.FinishARound();
    }
    MPI_Barrier(comm_spec_.comm());  // 同步
  }

  /**
   * 用于重新加载图，并完成图变换前后值的校正
   *
   */
  void reloadGraph() {
    // recycled value on the old graph
    AmendValue(-1);

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
    
    // graph_ = inc_fragment_builder.Build();
    app_->rebuild_graph(*graph_);
    const std::shared_ptr<fragment_t>& new_graph = inc_fragment_builder.Build();
    app_->iterate_begin(*graph_);
    app_->iterate_begin(*new_graph);

    if(FLAGS_compress){
      cpr_->inc_run(deleted_edges, added_edges, new_graph);
    }
    graph_ = new_graph;

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "New graph loaded" << " new_graph_edgeNum=" << (graph_->GetEdgeNum()/2);
    }
    app_->Init(comm_spec_, *graph_, false);
    {
      // Copy values to new graph
      for (auto v : iv) {
        app_->values_[v] = values[v];
        app_->deltas_[v] = deltas[v];
      }
    }
    // app_->iterate_begin(*graph_);
    // reissue value on the new graph
    AmendValue(1);

    if(FLAGS_compress){
      cpr_->inc_precompute_supernode();
    }
  }

  void Query() {
    MPI_Barrier(comm_spec_.comm());
    // app_->Init(comm_spec_, *graph_, false);

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
    //    auto& prioritys = app_->priority_;
    VertexArray<value_t, vid_t> last_values;


    int step = 1;
    bool batch_stage = true;
    bool compr_stage = FLAGS_compress; // true: supernode send


    last_values.Init(inner_vertices);
    value_t init_value_sum = 0;
    value_t init_delta_sum = 0;

    // for (auto v : inner_vertices) {
    parallel_for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
      vertex_t v(i);
      last_values[v] = values[v];
    }

    if(compr_stage){
      #ifdef DEBUG
        for(auto v : graph_->Vertices()){
          init_delta_sum += deltas[v];
          init_value_sum += values[v];
        }
        LOG(INFO) << "init_value_sum=" << init_value_sum << " init_delta_sum=" << init_delta_sum;
      #endif

      double node_type_time = GetCurrentTime();
      vid_t inner_node_num = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();
      node_type.clear();
      node_type.resize(inner_node_num, std::numeric_limits<char>::max());
      bound_node_values.clear();
      bound_node_values.Init(inner_vertices, app_->default_v());
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
            vertex_t u(i);
          if(cpr_->Fc[u] == cpr_->FC_default_value){
            node_type[i] = 0; // out node
          }
          else if(cpr_->Fc[u] >= 0){
            node_type[i] = 2; // source node
          }
          else if(!cpr_->supernode_out_bound[i]){
            node_type[i] = 4; // inner node
          }
          if(cpr_->supernode_out_bound[i]){
            node_type[i] = 1; // bound node
            if(cpr_->Fc[u] >= 0){
              node_type[i] = 3; // source node + bound node
            }
          }
      }
      all_nodes.clear();
      all_nodes.resize(5);
      for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
          all_nodes[node_type[i]].emplace_back(vertex_t(i));
      }
      LOG(INFO) << "node_type_time=" << (GetCurrentTime()-node_type_time); //0.313418

      // debug
      {
        vid_t all_node_size = 0;
        for(int i = 0; i <= 4; i++){
          all_node_size += all_nodes[i].size();
          LOG(INFO) << "node_" << i << "=" << all_nodes[i].size();
        }
        LOG(INFO) << "all_node_size=" << all_node_size << " inner_node_num=" << inner_node_num;
        
        LOG(INFO) << "---node1_num=" << all_nodes[1].size();
        all_nodes[1].clear();
        size_t spn_ids_num = cpr_->supernode_ids.size();
        for(vid_t j = 0; j < spn_ids_num; j++){  // parallel compute
          std::vector<vertex_t> &node_set = cpr_->supernode_ids[j]; 
          for(auto v : node_set){
            if(node_type[v.GetValue()] == 1){
              all_nodes[1].emplace_back(v);
            }
          }
        }
        LOG(INFO) << "---node1_num=" << all_nodes[1].size();
        // 统计入度
        std::vector<size_t> node_0_indegree(inner_node_num, 0);
        std::vector<size_t> node_1_indegree(inner_node_num, 0);
        {
          const std::vector<vertex_t>& nodes_0 = all_nodes[0];
          vid_t node_0_size = nodes_0.size();
          for(vid_t i = 0; i < node_0_size; i++){
            vertex_t u = nodes_0[i];
            auto oes = graph_->GetOutgoingAdjList(u);
            for(auto e : oes){
              node_0_indegree[e.neighbor.GetValue()]++;
            }
          }
          size_t max_in_degree_0 = 0;
          for(vid_t i = 0; i < inner_node_num; i++){
            max_in_degree_0 = std::max(max_in_degree_0, node_0_indegree[i]);
          }
          LOG(INFO) << " ----max_in_degree_0=" << max_in_degree_0;
        }
        {
          const std::vector<vertex_t>& nodes_1 = all_nodes[1];
          vid_t node_1_size = nodes_1.size();
          for(vid_t i = 0; i < node_1_size; i++){
            vertex_t u = nodes_1[i];
            auto oes = graph_->GetOutgoingAdjList(u);
            for(auto e : oes){
              node_1_indegree[e.neighbor.GetValue()]++;
            }
          }
          size_t max_in_degree_1 = 0;
          for(vid_t i = 0; i < inner_node_num; i++){
            max_in_degree_1 = std::max(max_in_degree_1, node_1_indegree[i]);
          }
          LOG(INFO) << " ----max_in_degree_1=" << max_in_degree_1;
        }
      }

      double  transfer_csr_time = GetCurrentTime();

      /* source to in_bound_node */
      // spnode_datas.resize(cpr_->supernodes_num);

      double  init_time_1 = GetCurrentTime();
      spnode_datas.Init(inner_vertices, 0);
      is_e_.clear();
      is_e_offset_.clear();
      ib_e_.clear();
      ib_e_offset_.clear();
      size_t source_e_num = 0;
      size_t bound_e_num = 0;
      std::vector<size_t> is_e_degree(inner_node_num+1, 0);
      std::vector<size_t> ib_e_degree(inner_node_num+1, 0);
      LOG(INFO) << "init_time_1=" << (GetCurrentTime()-init_time_1); //1.05179


      double csr_time_1 = GetCurrentTime();
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        char type = node_type[i];
        if(type == 2 || type == 3){
          supernode_t &spnode = cpr_->supernodes[cpr_->Fc_map[u]];
          // source_e_num += spnode.bound_delta.size();
          is_e_degree[i+1] = spnode.bound_delta.size();
          atomic_add(source_e_num, spnode.bound_delta.size());
        }
        if(type == 1 || type == 3){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = cpr_->id2spids[u];
          size_t temp_cnt = 0;
          for(vid_t j = 0; j < out_degree; j++){  // 关闭多线程
            auto& e = *(it + j);
            if(ids_id != cpr_->id2spids[e.neighbor]){
              // bound_e_num += 1;
              temp_cnt += 1;
            }
          }
          ib_e_degree[i+1] += temp_cnt;
          atomic_add(bound_e_num, temp_cnt);
        }
      }
      LOG(INFO) << "csr_time_1=" << (GetCurrentTime()-csr_time_1); //1.25987

      /* get index start */
      double index_time = GetCurrentTime();
      for(vid_t i = 1; i < inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
      }
      LOG(INFO) << "index_time=" << (GetCurrentTime()-index_time); //0.226317

      LOG(INFO) << "inner_node_num=" << inner_node_num;
      LOG(INFO) << "inner_node_num=" << graph_->GetVerticesNum();

      double init_time_2 = GetCurrentTime();
      is_e_.resize(source_e_num);
      ib_e_.resize(bound_e_num);
      is_e_offset_.resize(inner_node_num+1);
      ib_e_offset_.resize(inner_node_num+1);
      LOG(INFO) << "init_time_2=" << (GetCurrentTime()-init_time_2); //1.13601

      // vid_t index_s = 0;
      // vid_t index_b = 0;
      // build edge
      double csr_time_2 = GetCurrentTime();
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        /* source node */
        vid_t index_s = is_e_degree[i];
        is_e_offset_[u.GetValue()] = &is_e_[index_s];
        char type = node_type[u.GetValue()];
        if(type == 2 || type == 3){
          supernode_t &spnode = cpr_->supernodes[cpr_->Fc_map[u]];
          for(auto& oe : spnode.bound_delta){
            // is_e_[index_s] = nbr_index_t(oe.first.GetValue(), oe.second);
            is_e_[index_s].neighbor = oe.first;
            is_e_[index_s].data = oe.second;
            index_s++;
          }
        }
        /* inner_bound node */
        vid_t index_b = ib_e_degree[i];
        ib_e_offset_[u.GetValue()] = &ib_e_[index_b];
        if(type == 1 || type == 3){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = cpr_->id2spids[u];
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            if(ids_id != cpr_->id2spids[e.neighbor]){
              ib_e_[index_b] = e;
              index_b++;
            }
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      LOG(INFO) << "csr_time_2=" << (GetCurrentTime()-csr_time_2); //0.207281

      //debug
      {
        const std::vector<vertex_t>& nodes_0 = all_nodes[0];
        vid_t node_0_size = nodes_0.size();
        size_t max_edge_0_num = 0;
        size_t edge_0_num = 0;
        for(vid_t i = 0; i < node_0_size; i++){
          vertex_t u(nodes_0[i]);
          auto oes = graph_->GetOutgoingAdjList(u);
          max_edge_0_num = std::max(max_edge_0_num, oes.Size());
          edge_0_num += oes.Size();
        }
        LOG(INFO) << "---max_edge_0_num=" << max_edge_0_num << " edge_0_num=" << edge_0_num << " node_0_size=" << node_0_size;
        const std::vector<vertex_t>& nodes_1 = all_nodes[1];
        vid_t node_1_size = nodes_1.size();
        size_t max_edge_1_num = 0;
        size_t edge_1_num = 0;
        for(vid_t i = 0; i < node_1_size; i++){
          vertex_t u(nodes_1[i]);
          adj_list_t adj = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
          max_edge_1_num = std::max(max_edge_1_num, adj.Size());
          edge_1_num += adj.Size();
        }
        LOG(INFO) << "---max_edge_1_num=" << max_edge_1_num << " edge_1_num=" << edge_1_num << " node_1_size=" << node_1_size;
      }

      LOG(INFO) << " transfer_csr_time=" << (GetCurrentTime()- transfer_csr_time); //3.88149
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

    app_->g_num = 0;

    exec_time -= GetCurrentTime();
    double for_time = 0.d;
    while (true) {
      ++step;
      messages_.StartARound();
      app_->next_modified_.ParallelClear(thread_num());

      {
        messages_.ParallelProcess<fragment_t, value_t>(
            thread_num(), *graph_,
            [this](int tid, vertex_t v, value_t received_delta) {
              app_->accumulate_atomic(app_->deltas_[v], received_delta);
            });
      }

      {
        for_time -= GetCurrentTime();
        {
          if (FLAGS_cilk) {
            ForEachCilkOfBitset(
              app_->curr_modified_, inner_vertices, [this](int tid, vertex_t u) {
                auto& value = app_->values_[u];
                auto& delta = app_->deltas_[u];
                app_->accumulate(value, delta);
                app_->g_function(*graph_, u, value, delta, app_->next_modified_);
            });
          } else {
            ForEach(
              app_->curr_modified_, inner_vertices, [this](int tid, vertex_t u) {
                // auto& delta = app_->deltas_[u];
                // auto oes = graph_->GetOutgoingAdjList(u);
                // for (auto & e : oes) {
                //   // vertex_t v = e.get_neighbor();
                //   vertex_t v = e.neighbor;
                //   // float ndistu = delta + e.get_data();
                //   float ndistu = delta + e.data;
                //   if (ndistu < app_->deltas_[v]) {
                //     atomic_min(app_->deltas_[v], ndistu);
                //     app_->next_modified_.Insert(v);
                //   }
                // }
                auto& value = app_->values_[u];
                auto& delta = app_->deltas_[u];
                app_->accumulate(value, delta);
                app_->g_function(*graph_, u, value, delta, app_->next_modified_);
            });
          }

        }
        for_time += GetCurrentTime();
      }


      {
        auto& channels = messages_.Channels();
        // send local delta to remote
        ForEach(app_->curr_modified_, outer_vertices, [this, &deltas, &channels](int tid,
                                                           vertex_t v) {
          auto& delta_to_send = deltas[v];

          // if (delta_to_send != app_->default_v()) {
            channels[tid].template SyncStateOnOuterVertex<fragment_t, value_t>(
                *graph_, v, delta_to_send);
            delta_to_send = app_->default_v();
          // }
        });

        if (!app_->next_modified_.PartialEmpty(0, graph_->GetInnerVerticesNum())) {
          messages_.ForceContinue();
        }
      }

      messages_.FinishARound();


      if (messages_.ToTerminate()) {
        break;
      }

      app_->next_modified_.Swap(app_->curr_modified_);
    }
    exec_time += GetCurrentTime();
    LOG(INFO) << "exec_time=" << exec_time;
    LOG(INFO) << "step=" << step;
    LOG(INFO) << "for_time=" << for_time;

    // Analysis result
    double d_sum = 0;
    vertex_t source;
    bool native_source =
        graph_->GetInnerVertex(FLAGS_sssp_source, source);
    vid_t max_id = native_source ? source.GetValue() : 0;
    auto& result = app_->values_;
    // auto& result = app_->deltas_;
    for (auto v : graph_->InnerVertices()) {
      if (!(std::fabs(result[v] - app_->default_v()) < 1e-10)) {
        d_sum += result[v];
      }
      if (result[v] > result[vertex_t(max_id)] && !(std::fabs(result[v] - app_->default_v()) < 1e-10)) {
        max_id = v.GetValue();
      }
    }
    LOG(INFO) << "max_d[" << graph_->GetId(vertex_t(max_id)) << "]=" << result[vertex_t(max_id)];
    LOG(INFO) << "d_sum=" << d_sum;
    printf("d_sum=%.10lf\n", d_sum);

    MPI_Barrier(comm_spec_.comm());
    if(compr_stage){
      delete cpr_;
    }
  }

  void Output(std::ostream& os) {
    auto inner_vertices = graph_->InnerVertices();
    auto& values = app_->values_;
    // auto& values = app_->deltas_;

    for (auto v : inner_vertices) {
      os << graph_->GetId(v) << " " << values[v] << std::endl;
    }
  }

  void Finalize() { messages_.Finalize(); }

 private:
  bool termCheck(VertexArray<value_t, vid_t>& last_values,
                 VertexArray<value_t, vid_t>& values, bool compr_stage) {
    terminate_checking_time_ -= GetCurrentTime();
    auto vertices = graph_->InnerVertices();
    double diff_sum = 0, global_diff_sum;

    for (auto u : vertices) {
      diff_sum += fabs(last_values[u] - values[u]);
      last_values[u] = values[u];
    }

    communicator_.template Sum(diff_sum, global_diff_sum);

    // if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
    //   LOG(INFO) << "Diff: " << global_diff_sum;
    // }

    terminate_checking_time_ += GetCurrentTime();
    return global_diff_sum < FLAGS_termcheck_threshold;
  }

  // VertexArray<vid_t, vid_t> spnode_ids;
  std::vector<char> node_type; // all node's types, 0:out node, 1:bound node, 2:source node, 3:belong 1 and 2 at the same time, 127:inner node that needn't send message.
  VertexArray<value_t, vid_t> bound_node_values;
  std::shared_ptr<APP_T> app_;
  std::shared_ptr<fragment_t>& graph_;
  message_manager_t messages_;
  Communicator communicator_;
  CommSpec comm_spec_;
  double terminate_checking_time_;
  IterCompressor<APP_T, supernode_t>* cpr_;
  // std::vector<value_t> spnode_datas;
  VertexArray<value_t, vid_t> spnode_datas{};;
  /* source to in_bound_node */
  Array<nbr_index_t, Allocator<nbr_index_t>> is_e_;
  Array<nbr_index_t*, Allocator<nbr_index_t*>> is_e_offset_;
  /* in_bound_node to out_bound_node */
  Array<nbr_t, Allocator<nbr_t>> ib_e_;
  Array<nbr_t*, Allocator<nbr_t*>> ib_e_offset_;
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

#endif  // GRAPE_WORKER_SUM_BATCH_WORKER_H_

