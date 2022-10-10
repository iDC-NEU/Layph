#ifndef GRAPE_FRAGMENT_ITER_COMPRESSOR_H_
#define GRAPE_FRAGMENT_ITER_COMPRESSOR_H_

#include "grape/graph/super_node.h"
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include "timer.h"
#include "flags.h"
#include <iomanip>
#include "grape/fragment/compressor_base.h"
#include <atomic>

// #define NUM_THREADS 52

namespace grape {

template <typename APP_T, typename SUPERNODE_T>
class IterCompressor : public CompressorBase <APP_T, SUPERNODE_T> {
    public:
    using fragment_t = typename APP_T::fragment_t;
    using value_t = typename APP_T::value_t;
    using vertex_t = typename APP_T::vertex_t;
    using vid_t = typename APP_T::vid_t;
    using supernode_t = SUPERNODE_T;
    using fc_t = int32_t;
    using nbr_t = typename fragment_t::nbr_t;
    using adj_list_t = typename fragment_t::adj_list_t;
    double supernode_termcheck_threshold = FLAGS_termcheck_threshold/10000; // to build index
    double supernode_termcheck_threshold_2 = FLAGS_termcheck_threshold/1000; // to pre compute
    const bool iter_compressor_flags_cilk = false;
    // int thread_num = FLAGS_build_index_concurrency;
    int thread_num = 52; // just test increment computation

    IterCompressor(std::shared_ptr<APP_T>& app,
                        std::shared_ptr<fragment_t>& graph):CompressorBase<APP_T, SUPERNODE_T>(app, graph){}

    void init_array() {
        /* init */
        test_time.resize(thread_num);
        values_array.resize(thread_num);
        deltas_array.resize(thread_num);
        bounds_array.resize(thread_num);
        double s = GetCurrentTime();
        parallel_for(int tid = 0; tid < thread_num; tid++){
            auto all_nodes = VertexRange<vid_t>(0, this->all_node_num);
            values_array[tid].Init(all_nodes);
            deltas_array[tid].Init(all_nodes);
            bounds_array[tid].Init(all_nodes);
            test_time[tid].resize(4); // for debug
        }
        this->app_->Init(this->comm_spec_, *(this->graph_), false);
    }

    void run(){
        this->supernode_out_bound.clear();
        this->supernode_out_bound.resize(this->graph_->GetVerticesNum(), 0);

        // 设置每个点收敛的阈值：
        supernode_termcheck_threshold = FLAGS_termcheck_threshold / this->graph_->GetVerticesNum();
        supernode_termcheck_threshold_2 = FLAGS_termcheck_threshold / this->graph_->GetVerticesNum();

        /* find supernode */
        timer_next("find supernode");
        {  
            this->compress();
            init_array();
            this->judge_out_bound_node(this->graph_);
            /* build subgraph of supernode */
            // build_subgraph(this->graph_);
            this->build_subgraph_mirror(this->graph_);
        }

        /* calculate index for each structure */
        timer_next("calculate index");
        double calculate_index = GetCurrentTime();
        {
            // /* parallel */
            // /* Simulate thread pool */
            std::atomic<vid_t> spnode_id(0);
            std::atomic<vid_t> active_thread_num(thread_num);
            this->ForEach(this->supernodes_num, [this, &spnode_id, &active_thread_num](int tid) {
                int i = 0, cnt = 0, step = 1;  // step need to be adjusted
                while(i < this->supernodes_num){
                    // i = __sync_fetch_and_add(&spnode_id, step);
                    i = spnode_id.fetch_add(step);
                    for(int j = i; j < i + step; j++){
                        if(j < this->supernodes_num){
                            build_iter_index_mirror(j, this->graph_, tid);
                            cnt++;
                        }
                        else{
                            break;
                        }
                    }
                }
                }, thread_num
            );
        }
        calculate_index = GetCurrentTime() - calculate_index;
        LOG(INFO) << "#calculate_index: " << calculate_index;
    }

    void clean_deltas(){
        vid_t node_num = this->graph_->Vertices().end().GetValue();
        for(vid_t tid = 0; tid < FLAGS_build_index_concurrency; tid++){
            VertexArray<value_t, vid_t>& self_deltas = deltas_array[tid];
            parallel_for(vid_t i = 0; i < node_num; i++){
                vertex_t v(i);                                                                                          
                self_deltas[v] = this->app_->default_v();
            }
        }
    }

    void build_subgraph(const std::shared_ptr<fragment_t>& new_graph){
        const auto& inner_vertices = new_graph->InnerVertices();
        const vid_t spn_ids_num = this->supernode_ids.size();
        vid_t inner_node_num = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();
        std::vector<size_t> ia_oe_degree(inner_node_num+1, 0);
        std::vector<size_t> ib_oe_degree(inner_node_num+1, 0);
        vid_t ia_oe_num = 0; 
        vid_t ib_oe_num = 0; 
        parallel_for(vid_t i = 0; i < spn_ids_num; i++){
        // for(vid_t i = 0; i < spn_ids_num; i++){
            std::vector<vertex_t> &node_set = this->supernode_ids[i];
            vid_t temp_a = 0;
            vid_t temp_b = 0;
            for(auto v : node_set){
                auto ids_id = this->id2spids[v];
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                for(auto& oe : oes){
                    if(this->id2spids[oe.neighbor] == ids_id){ // inner edge
                        ia_oe_degree[v.GetValue()+1]++;
                        temp_a++;
                    }
                    else{
                        ib_oe_degree[v.GetValue()+1]++;
                        temp_b++;
                    }
                }
            }
            atomic_add(ia_oe_num, temp_a);
            atomic_add(ib_oe_num, temp_b);
        }
        ia_oe_.clear();
        ia_oe_.resize(ia_oe_num);
        ia_oe_offset_.clear();
        ia_oe_offset_.resize(inner_node_num+1);
        ib_oe_.clear();
        ib_oe_.resize(ib_oe_num);
        ib_oe_offset_.clear();
        ib_oe_offset_.resize(inner_node_num+1);

        for(vid_t i = 1; i < inner_node_num; i++) {
            ia_oe_degree[i] += ia_oe_degree[i-1];
            ib_oe_degree[i] += ib_oe_degree[i-1];
        }

        parallel_for(vid_t i = 0; i < inner_node_num; i++) {
        // for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
            vertex_t u(i);
            vid_t index_a = ia_oe_degree[i];
            ia_oe_offset_[i] = &ia_oe_[index_a];
            vid_t index_b = ib_oe_degree[i];
            ib_oe_offset_[i] = &ib_oe_[index_b];
            if(this->Fc[u] != this->FC_default_value){
                auto ids_id = this->id2spids[u];
                const auto& oes = new_graph->GetOutgoingAdjList(u);
                for(auto& oe : oes){
                    if(this->id2spids[oe.neighbor] == ids_id){ // inner edge
                        ia_oe_[index_a] = oe;
                        index_a++;
                    }
                    else{
                        ib_oe_[index_b] = oe;
                        index_b++;
                    }
                }
            }
            // CHECK_EQ(index_s, ia_oe_degree[i+1]);
        }
        ia_oe_offset_[inner_node_num] = &ia_oe_[ia_oe_num-1] + 1;
        ib_oe_offset_[inner_node_num] = &ib_oe_[ib_oe_num-1] + 1;
    }


    /**
     * To compute indexes in parallel, use a VertexArray
    */
    void build_iter_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, vid_t tid){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<value_t, vid_t>& deltas = deltas_array[tid];
        supernode_t& spnode = this->supernodes[spid];
        const auto& source = spnode.id;
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];

        /* init values/deltas */
        test_time[tid][0] -= GetCurrentTime();
        for(auto v : node_set){
            this->app_->init_c(v, deltas[v], *new_graph, source);
            this->app_->init_v(v, values[v]);
        }
        test_time[tid][0] += GetCurrentTime();
        /* iterative calculation */
        test_time[tid][1] -= GetCurrentTime();
        double b = GetCurrentTime();
        inc_run_to_convergence(tid, new_graph, node_set, source);
        test_time[tid][3] = std::max(test_time[tid][3], GetCurrentTime()-b);
        test_time[tid][1] += GetCurrentTime();
        /* build new index in supernodes */
        test_time[tid][2] -= GetCurrentTime();
        fianl_build_iter_index_for_mode2(tid, new_graph, node_set, spnode);
        test_time[tid][2] += GetCurrentTime();
    }

    /**
     * To compute indexes in parallel, use a VertexArray
    */
    void build_iter_index_mirror(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, vid_t tid){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<value_t, vid_t>& deltas = deltas_array[tid];
        supernode_t& spnode = this->supernodes[spid];
        // spnode.data = this->app_->default_v();
        const auto& source = spnode.id;
        std::vector<vertex_t> &node_set = this->cluster_ids[spnode.ids]; // inner node id + mirror id
        std::vector<vertex_t> &inner_node_set = this->supernode_ids[spnode.ids];
        // std::vector<vertex_t> &out_mirror_node_set = this->cluster_out_mirror_ids[spnode.ids]; // include mirror

        /* init values/deltas */
        // test_time[tid][0] -= GetCurrentTime();
        for(auto v : node_set){
            this->app_->init_c(v, deltas[v], *new_graph, source);
            this->app_->init_v(v, values[v]);
        }
        // test_time[tid][0] += GetCurrentTime();
        /* iterative calculation */
        // test_time[tid][1] -= GetCurrentTime();
        // double b = GetCurrentTime();
        inc_run_to_convergence_mirror(tid, new_graph, inner_node_set, source);
        // test_time[tid][3] = std::max(test_time[tid][3], GetCurrentTime()-b);
        // test_time[tid][1] += GetCurrentTime();
        /* build new index in supernodes */
        // test_time[tid][2] -= GetCurrentTime();
        // fianl_build_iter_index2(tid, new_graph, node_set, spnode); 
        fianl_build_iter_index_for_mirror(tid, new_graph, node_set, spnode); 
        // test_time[tid][2] += GetCurrentTime();
    }

    /* use a VertexArray */
    void inc_run_to_convergence_mirror(vid_t tid, 
                                const std::shared_ptr<fragment_t>& new_graph, 
                                const std::vector<vertex_t> &inner_node_set, 
                                vertex_t source){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<value_t, vid_t>& deltas = deltas_array[tid];
        // const vid_t ids_id = this->id2spids[source];
        int step = 0;
        float Diff = 0;
        //double now_termcheck_threshold = supernode_termcheck_threshold / node_set.size();
        size_t inner_node_set_size = inner_node_set.size();
        double threshold_full = FLAGS_termcheck_threshold / this->old_node_num 
                                * inner_node_set_size;
        // double threshold_full = FLAGS_termcheck_threshold 
        //                         / this->indegree[this->GetClusterSize()]
        //                         * this->indegree[this->id2spids[source]];

        /* in_mirror_source vertex first send a message to its neighbors */
        if (source.GetValue() >= this->old_node_num) {
            auto to_send = deltas[source];
            if(to_send != this->app_->default_v()){
                auto& value = values[source];
                vertex_t v = this->mirrorid2vid[source];
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                const auto& inner_oes = this->subgraph[source.GetValue()];
                deltas[source] = this->app_->default_v();
                #ifdef COUNT_ACTIVE_EDGE_NUM
                  atomic_add(this->app_->f_index_count_num, (long long)inner_oes.size());
                #endif
                for(auto e : inner_oes){
                    value_t outv = this->app_->default_v();
                    this->app_->g_function(*new_graph, v, value, to_send, oes, 
                                           e, outv);
                    // this->app_->accumulate_atomic(deltas[e.neighbor], outv);
                    this->app_->accumulate(deltas[e.neighbor], outv); 
                }
                this->app_->accumulate(value, to_send); 
            }
        }

        while (true) {
            step++;
            Diff = 0;
            // receive & send
            for(vid_t i = 0; i < inner_node_set_size; i++){
                const vertex_t& v = inner_node_set[i];
                // auto to_send = atomic_exch(deltas[v], this->app_->default_v());
                auto to_send = deltas[v];
                auto last_value = values[v];
                deltas[v] = this->app_->default_v();
                if(to_send != this->app_->default_v()){
                    const auto& oes = new_graph->GetOutgoingAdjList(v);
                    const auto& inner_oes = this->subgraph[v.GetValue()];

                    auto& value = values[v];
                    value_t outv = this->app_->default_v();
                    #ifdef COUNT_ACTIVE_EDGE_NUM
                      atomic_add(this->app_->f_index_count_num, (long long)inner_oes.size());
                    #endif
                    for(auto& e : inner_oes){
                        this->app_->g_function(*new_graph, v, value, to_send, 
                                                oes, e, outv);
                        this->app_->accumulate(deltas[e.neighbor], outv); // 
                    }
                    this->app_->accumulate(value, to_send); // 
                }
                Diff += fabs(last_value - values[v]);
            }
            // check convergence
            if(Diff <= threshold_full || step > 100){
                break;
            }
        }
    }

    /* use a bounds_array */
    void fianl_build_iter_index_for_mirror(vid_t tid, 
                                const std::shared_ptr<fragment_t>& new_graph, 
                                const std::vector<vertex_t> &node_set, 
                                supernode_t& spnode){
        std::vector<vertex_t> &local_node_set = this->supernode_ids[spnode.ids];
        std::vector<vertex_t> &mirror_node_set = this->cluster_out_mirror_ids[spnode.ids];
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<value_t, vid_t>& deltas = deltas_array[tid];
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        spnode.inner_value.clear();
        spnode.inner_delta.clear();
        spnode.bound_delta.clear();
        spnode.status = true;  

        for(auto v : local_node_set){
            auto& value = values[v];
            auto& delta = deltas[v];
            if(value != this->app_->default_v()){
                value_t rt_value;
                this->app_->g_revfunction(value, rt_value);
                if(this->supernode_out_bound[v.GetValue()]){
                    spnode.bound_delta.emplace_back(std::pair<vertex_t, value_t>(v, rt_value));
                }
                else {
                    spnode.inner_value.emplace_back(std::pair<vertex_t, value_t>(v, rt_value));
                }
            }
            /* inner_delta index, if there is absolute internal convergence, this index may not be established */
            if(delta != this->app_->default_v()){
                value_t rt_value;
                this->app_->g_revfunction(delta, rt_value);
                spnode.inner_delta.emplace_back(std::pair<vertex_t, value_t>(v, rt_value));
            }
        }
        for(auto m : mirror_node_set){
            auto& value = values[m];
            auto& delta = deltas[m];
            this->app_->accumulate(value, delta);
            if(value != this->app_->default_v()){  
                value_t rt_value;
                this->app_->g_revfunction(value, rt_value);
                spnode.bound_delta.emplace_back(std::pair<vertex_t, value_t>(m, rt_value));
            }
        }
    }

    /* use a bounds_array */
    void fianl_build_iter_index_for_mode2(vid_t tid, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, supernode_t& spnode){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<value_t, vid_t>& deltas = deltas_array[tid];
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[spnode.id];
        spnode.inner_value.clear();
        spnode.inner_delta.clear();
        spnode.bound_delta.clear();
        spnode.status = true; 

        for(auto v : node_set){
            auto& value = values[v];
            auto& delta = deltas[v];
            if(value != this->app_->default_v()){
                value_t rt_value;
                this->app_->g_revfunction(value, rt_value);
                if(this->supernode_out_bound[v.GetValue()]){
                    spnode.bound_delta.emplace_back(std::pair<vertex_t, value_t>(v, rt_value));
                }
                else {
                    spnode.inner_value.emplace_back(std::pair<vertex_t, value_t>(v, rt_value));
                }
            }
            /* inner_delta index, if there is absolute internal convergence, this index may not be established */
            if(delta != this->app_->default_v()){
                value_t rt_value;
                this->app_->g_revfunction(delta, rt_value);
                spnode.inner_delta.emplace_back(std::pair<vertex_t, value_t>(v, rt_value));
            }
        }
    }

    void precompute_spnode_one(const std::shared_ptr<fragment_t>& new_graph,
                               const bool is_inc){
        /* if the source vertex is within the supernode and isn't the entry point. */
        double pre_compute = GetCurrentTime();
        auto& old_values = deltas_array[0];
        auto& values = this->app_->values_;
        auto& deltas = this->app_->deltas_;

        if (is_inc == false) { // batch
            /* copy value */
            parallel_for(vid_t j = 0; j < this->old_node_num; j++){
                vertex_t v(j);
                old_values[v] = values[v];
                values[v] = this->app_->default_v();
            }
            /* internal iteration */
            parallel_for(vid_t j = 0; j < this->cluster_ids.size(); j++){
                run_to_convergence_for_precpt(j);
            }
        } else {  // inc
            /* copy value */
            parallel_for(vid_t j = 0; j < this->old_node_num; j++){
                vertex_t v(j);
                old_values[v] = values[v];
                values[v] = this->app_->default_v();
            }
            /* internal iteration */
            size_t update_ids_num = this->update_cluster_ids.size();
            parallel_for(vid_t j = 0; j < update_ids_num; j++){
                vid_t ids_id = this->update_cluster_ids[j];
                run_to_convergence_for_precpt(ids_id);
            }
        }
    }

    void precompute_spnode_two() {
        /* merge old values */
        auto& old_values = deltas_array[0];
        auto& values = this->app_->values_;
        parallel_for(vid_t j = 0; j < this->old_node_num; j++){
            vertex_t v(j);
            this->app_->accumulate(values[v], old_values[v]);
        }
    }

    /* Used for precomputing */
    void run_to_convergence_for_precpt(const vid_t ids_id){
        std::vector<vertex_t> &node_set = this->supernode_ids[ids_id]; 
        int step = 0;
        float Diff = 0;
        auto& values = this->app_->values_;
        auto& deltas = this->app_->deltas_;
        size_t node_set_size = node_set.size();

        // double threshold_full = supernode_termcheck_threshold_2  * node_set_size;
        double threshold_full = FLAGS_termcheck_threshold / this->old_node_num 
                                * node_set_size;

        while (true) {
            step++;
            Diff = 0;
            // receive & send
            for(auto v : node_set){
                auto to_send = deltas[v];
                if(to_send != this->app_->default_v()){
                    auto last_value = values[v];
                    deltas[v] = this->app_->default_v();
                    auto& value = values[v];
                    const auto& oes = this->graph_->GetOutgoingAdjList(v);
                    /* inner edges */
                    const auto& inner_oes = this->subgraph[v.GetValue()];
                    #ifdef COUNT_ACTIVE_EDGE_NUM
                      atomic_add(this->app_->f_index_count_num, (long long)inner_oes.size());
                    #endif
                    for(auto e : inner_oes){
                        value_t outv = 0;
                        this->app_->g_function(*(this->graph_), v, value, to_send, oes, e, outv);
                        this->app_->accumulate(deltas[e.neighbor], outv); // inner nodes
                    }
                    this->app_->accumulate(value, to_send);
                    Diff += fabs(last_value - values[v]);
                }
            }
            // check convergence
            if(Diff <= threshold_full || step > 100){
                break;
            }
        }
    }

    void init_node(const std::vector<vertex_t> &node_set, const vertex_t& source){
        auto& values = this->app_->values_;
        auto& deltas = this->app_->deltas_;
        for(auto v : node_set){
            this->app_->init_c(v, deltas[v], *(this->graph_), source);
            this->app_->init_v(v, values[v]);
        }
    }

    void inc_run(std::vector<std::pair<vid_t, vid_t>>& deleted_edges, 
                  std::vector<std::pair<vid_t, vid_t>>& added_edges, 
                  const std::shared_ptr<fragment_t>& new_graph,
                  const VertexArray<bool, vid_t>& temp_is_update 
                                                  = VertexArray<bool, vid_t>()){
        is_update = temp_is_update;
          
        /* Switch data to support add new nodes */
        VertexArray<fc_t, vid_t> old_Fc;
        auto old_vertices = this->graph_->Vertices();
        old_Fc.Init(old_vertices, this->FC_default_value);
        parallel_for(vid_t i = old_vertices.begin().GetValue(); i < old_vertices.end().GetValue(); i++) {
            vertex_t v(i);
            old_Fc[v] = this->Fc[v];
        }
        this->Fc.Init(new_graph->Vertices(), this->FC_default_value);
        // copy to new graph
        parallel_for(vid_t i = new_graph->Vertices().begin().GetValue(); i < new_graph->Vertices().end().GetValue(); i++) {
            vertex_t v(i);
            this->Fc[v] = old_Fc[v];
        }

        auto old_inner_vertices = this->graph_->InnerVertices();

        /* find supernode */
        double inc_compress = GetCurrentTime();
        this->inc_compress_mirror(deleted_edges, added_edges, new_graph);
        inc_compress = GetCurrentTime()-inc_compress;
        LOG(INFO) << "#inc_compress: " << inc_compress;

        this->clean_no_used(this->app_->values_, this->app_->default_v());

        /* init supernode_out_bound*/
        double begin = GetCurrentTime();
        // this->supernode_out_bound.clear();
        this->supernode_out_bound.resize(this->graph_->GetVerticesNum(), 0);
        this->inc_judge_out_bound_node(new_graph);
        /* build subgraph of supernode */
        this->inc_build_subgraph_mirror(new_graph);
        LOG(INFO) << "#inc_build_subgraph_mirror: " << (GetCurrentTime() - begin);

        /* calculate index for each structure */
        double inc_calculate_index = GetCurrentTime();
        // inc_compute_index_mirror_cid(new_graph);  // to inc_trav_compress_mirror
        inc_compute_index_mirror_spid(new_graph); // to inc_compress_mirror
        inc_calculate_index = GetCurrentTime() - inc_calculate_index;
        LOG(INFO) << "#inc_calculate_index: " << inc_calculate_index;
    }

    void inc_compute_index_mirror_cid(const std::shared_ptr<fragment_t>& new_graph) {
      /* 1: cluster */
      // parallel_for(vid_t i = 0; i < this->update_cluster_ids.size(); i++) { // 并行
      //   vid_t tid = 0; // 后面用并行
      //   vid_t ids_id = this->update_cluster_ids[i];
      //   for(auto ms : this->cluster_in_mirror_ids[ids_id]) {
      //     vid_t spid = this->Fc_map[ms];
      //     inc_build_iter_index_mirror(spid, new_graph, tid);
      //     // build_iter_index_mirror(spid, new_graph, tid);
      //   }
      //   for(auto vs : this->supernode_source[ids_id]) {
      //     vid_t spid = this->Fc_map[vs];
      //     inc_build_iter_index_mirror(spid, new_graph, tid);
      //   }
      // }

      /* 2: supernode */
      double inc_calculate_index_1 = GetCurrentTime();
        std::vector<vid_t> spnodeidset;
        for(vid_t i = 0; i < this->update_cluster_ids.size(); i++) {
          vid_t tid = 0;
          vid_t ids_id = this->update_cluster_ids[i];
          for(auto ms : this->cluster_in_mirror_ids[ids_id]) {
            vid_t spid = this->Fc_map[ms];
            spnodeidset.emplace_back(spid);
          }
          for(auto vs : this->supernode_source[ids_id]) {
            vid_t spid = this->Fc_map[vs];
            spnodeidset.emplace_back(spid);
          }
        }

        vid_t update_num = spnodeidset.size();
        std::atomic<vid_t> spnode_id(0);
        std::atomic<vid_t> active_thread_num(thread_num);
        this->ForEach(update_num, [this, &spnode_id, &active_thread_num, &spnodeidset, &new_graph](int tid) {
            double time = GetCurrentTime();
            int i = 0, cnt = 0, step = 1;  // step need to be adjusted
            vid_t update_num = spnodeidset.size();
            while(i < update_num){
                i = spnode_id.fetch_add(step);
                for(int j = i; j < i + step; j++){
                    if(j < update_num){
                      vid_t id = spnodeidset[j];
                      build_iter_index_mirror(id, new_graph, tid);
                      cnt++;
                    }
                    else{
                        break;
                    }
                }
            }
          }, FLAGS_build_index_concurrency
        );
      inc_calculate_index_1 = GetCurrentTime() - inc_calculate_index_1;
      LOG(INFO) << "#inc_calculate_index_1: " << inc_calculate_index_1;
    }

    void inc_compute_index_mirror_spid(const std::shared_ptr<fragment_t>& new_graph) {
      vid_t update_num = this->update_source_id.size();
      std::atomic<vid_t> spnode_id(0);
      std::atomic<vid_t> active_thread_num(thread_num);
      this->ForEach(update_num, [this, &spnode_id, &active_thread_num, &new_graph](int tid) {
          double time = GetCurrentTime();
          int i = 0, cnt = 0, step = 1;  // step need to be adjusted
          vid_t update_num = this->update_source_id.size();
          while(i < update_num){
              i = spnode_id.fetch_add(step);
              for(int j = i; j < i + step; j++){
                  if(j < update_num){
                    vid_t spid = this->Fc_map[vertex_t(this->update_source_id[j])];
                    if (spid < this->supernodes_num) { 
                      inc_build_iter_index_mirror(spid, new_graph, tid);
                    }
                    cnt++;
                  } else{
                      break;
                  }
              }
          }
          }, FLAGS_build_index_concurrency
      );
    }

    void inc_build_iter_index_mirror(const vid_t spid, 
                                    const std::shared_ptr<fragment_t>& new_graph, 
                                    vid_t tid){
        supernode_t& spnode = this->supernodes[spid];
        if (spnode.status == false) { // no old shortcut or no benefit
          build_iter_index_mirror(spid, new_graph, tid); // rebuild shourtcut
          return ;
        }

        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<value_t, vid_t>& deltas = deltas_array[tid];
        // spnode.data = this->app_->default_v();
        const auto& source = spnode.id;
        std::vector<vertex_t> &node_set = this->cluster_ids[spnode.ids]; // inner node id + mirror id
        std::vector<vertex_t> &inner_node_set = this->supernode_ids[spnode.ids];

        /* init values/deltas */
        for(auto v : node_set){
            values[v] = this->app_->default_v();
            deltas[v] = this->app_->default_v();
        }

        /* get init value from old shortcuts, 
            inner_delta is null if this is a new source.
        */
        for(auto e : spnode.inner_value){
            values[e.first] = e.second; 
        }
        for(auto e : spnode.inner_delta){
            deltas[e.first] = e.second;
        } 
        // If you are using cos model two, you need to copy the following values.
        for(auto e : spnode.bound_delta){
            values[e.first] = e.second; 
        }
        // If source is a in-mirror, set value to 1. 
        if (source.GetValue() >= this->old_node_num) {
          values[source] = 1;
        }

        /* recycled value on the old graph */
        AmendValue_mirror(-1, tid, this->graph_, inner_node_set, 
                                                    source, this->subgraph_old);
        /* reissue value on the new graph */
        AmendValue_mirror(1, tid, new_graph, inner_node_set, source, 
                                                                this->subgraph);
        /* iterative calculation */
        inc_run_to_convergence_mirror(tid, new_graph, inner_node_set, source);
        /* build new index in supernodes */
        fianl_build_iter_index_for_mirror(tid, new_graph, node_set, spnode);
    }

    void AmendValue_mirror(int type, vid_t tid, 
                        const std::shared_ptr<fragment_t>& graph, 
                        const std::vector<vertex_t> &inner_node_set, 
                        vertex_t source,
                        std::vector<std::vector<nbr_t>>& temp_subgraph){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<value_t, vid_t>& deltas = deltas_array[tid];

        /* in_mirror_source vertex first send a message to its neighbors */
        if (source.GetValue() >= this->old_node_num) {
          auto& value = values[source];
          auto to_send = value * type;
          vertex_t v = this->mirrorid2vid[source];
          if (is_update[v]) {
            const auto& oes = graph->GetOutgoingAdjList(v);
            const auto& inner_oes = temp_subgraph[source.GetValue()]; 
            #ifdef COUNT_ACTIVE_EDGE_NUM
              atomic_add(this->app_->f_index_count_num, (long long)inner_oes.size());
            #endif
            for(auto e : inner_oes){
              value_t outv = this->app_->default_v();
              this->app_->g_function(*graph, v, value, to_send, oes, 
                                      e, outv);
              // this->app_->accumulate_atomic(deltas[e.neighbor], outv);
              this->app_->accumulate(deltas[e.neighbor], outv); 
            }
          }
        }

        size_t inner_node_set_size = inner_node_set.size();
        for(vid_t i = 0; i < inner_node_set_size; i++){
          const vertex_t& v = inner_node_set[i];
          if (is_update[v]) {
            auto& value = values[v];
            auto to_send = value * type;
            const auto& oes = graph->GetOutgoingAdjList(v);
            const auto& inner_oes = temp_subgraph[v.GetValue()];
            #ifdef COUNT_ACTIVE_EDGE_NUM
              atomic_add(this->app_->f_index_count_num, (long long)inner_oes.size());
            #endif
            for(auto e : inner_oes){
              value_t outv = this->app_->default_v();
              this->app_->g_function(*graph, v, value, to_send, oes, 
                                      e, outv);
              // this->app_->accumulate_atomic(deltas[e.neighbor], outv);
              this->app_->accumulate(deltas[e.neighbor], outv);
            }
          }
        }
    }


    void inc_precompute_supernode(){
        /* precompute supernode */
        timer_next("inc pre compute");
        double inc_pre_compute = GetCurrentTime();
        vid_t spn_ids_num = this->supernode_ids.size();
        if (iter_compressor_flags_cilk) {
#ifdef INTERNAL_PARALLEL
          LOG(FATAL) << "Layph is not compiled with -DUSE_CILK";
#endif
            parallel_for(vid_t j = 0; j < spn_ids_num; j++){  // parallel compute
                run_to_convergence_for_precpt(j);
            }
        }
        auto& deltas = this->app_->deltas_;
        vid_t node_num = this->graph_->Vertices().end().GetValue();
        for(vid_t tid = 0; tid < FLAGS_build_index_concurrency; tid++){
            VertexArray<value_t, vid_t>& self_deltas = deltas_array[tid];
            parallel_for(vid_t i = 0; i < node_num; i++) {
                vertex_t v(i);
                this->app_->accumulate(deltas[v], self_deltas[v]);
            }
        }
        inc_pre_compute = GetCurrentTime()-inc_pre_compute;
        LOG(INFO) << "#inc_pre_compute: " << inc_pre_compute;
        LOG(INFO) << "finish inc_precompute_supernode...";
    }

    void inc_compute_index(const std::shared_ptr<fragment_t>& new_graph){
        {
            double test_init_vector = GetCurrentTime();
            std::vector<vid_t> ids;
            // ids.reserve(this->inccalculate_spnode_ids.size());
            ids.insert(ids.begin(), this->inccalculate_spnode_ids.begin(), this->inccalculate_spnode_ids.end());
            int len = ids.size();
            /* Simulate thread pool */
            std::atomic<vid_t> spnode_id(0);
            int thread_num = FLAGS_build_index_concurrency;
            this->ForEach(len, [this, &spnode_id, &ids, &len, &new_graph](int tid) {
                int i = 0, cnt = 0;
                while(i < len){
                    // i = __sync_fetch_and_add(&spnode_id, 1);
                    i = spnode_id.fetch_add(1);
                    if(i < len){
                        vertex_t u(ids[i]);
                        if(this->Fc_map[u] != this->ID_default_value){
                            inc_build_iter_index(this->Fc_map[u], new_graph, tid); // inc
                            // build_iter_index(this->Fc_map[u], new_graph, tid); // re compute
                        }
                        cnt++;
                    }
                }
                }, thread_num
            );
        }
        // The newly added index must be recalculated
        {
            std::vector<vid_t> ids;
            // ids.reserve(this->inccalculate_spnode_ids.size());
            ids.insert(ids.begin(), this->recalculate_spnode_ids.begin(), this->recalculate_spnode_ids.end());
            int len = ids.size();
            /* parallel */
            /* Simulate thread pool */
            std::atomic<vid_t> spnode_id(0);
            int thread_num = FLAGS_build_index_concurrency;
            this->ForEach(len, [this, &spnode_id, &ids, &len, &new_graph](int tid) {
                int i = 0, cnt = 0;
                while(i < len){
                    // i = __sync_fetch_and_add(&spnode_id, 1);
                    i = spnode_id.fetch_add(1);
                    if(i < len){
                        vertex_t u(ids[i]);
                        if(this->Fc_map[u] != this->ID_default_value){
                            build_iter_index(this->Fc_map[u], new_graph, tid);
                        }
                        cnt++;
                    }
                }
                }, thread_num
            );
        }
    }

    /* use a VertexArray */
    void inc_build_iter_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, vid_t tid){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<value_t, vid_t>& deltas = deltas_array[tid];
        supernode_t& spnode = this->supernodes[spid];
        const vertex_t& source = spnode.id;
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];

        /* init values/deltas by inner_delta and inner_value, 
           Note that the inner_value and inner_delta at this time have been converted into weights.
           g_revfunction()
        */
        for(auto v : node_set){
            values[v] = this->app_->default_v();
            deltas[v] = this->app_->default_v();
        }
        for(auto e : spnode.inner_value){
            values[e.first] = e.second; 
        }
        for(auto e : spnode.inner_delta){
            deltas[e.first] = e.second;
        }        
        // If you are using cos model two, you need to copy the following values.
        for(auto e : spnode.bound_delta){
            values[e.first] = e.second; 
        }
        
        /* recycled value on the old graph */
        AmendValue(-1, tid, this->graph_, node_set, source);
        /* reissue value on the new graph */
        AmendValue(1, tid, new_graph, node_set, source);
        /* iterative calculation */
        inc_run_to_convergence(tid, new_graph, node_set, source);
        /* build new index in supernodes */
        // fianl_build_iter_index2(tid, new_graph, node_set, spnode);
        fianl_build_iter_index_for_mode2(tid, new_graph, node_set, spnode);
    }

    /* use a VertexArray */
    void AmendValue(int type, vid_t tid, const std::shared_ptr<fragment_t>& graph, const std::vector<vertex_t> &node_set, vertex_t source){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<value_t, vid_t>& deltas = deltas_array[tid];
        const vid_t ids_id = this->id2spids[source];
        // send
        for(auto v : node_set){
            const auto& oes = graph->GetOutgoingAdjList(v);
            auto& value = values[v];
            value_t to_send = value * type;
            if(value != this->app_->default_v()){
                #ifdef COUNT_ACTIVE_EDGE_NUM
                  atomic_add(this->app_->f_index_count_num, (long long)oes.Size());
                #endif
                for(auto& e : oes){
                    value_t outv = 0;
                    if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                        this->app_->g_function(*graph, v, value, to_send, oes, e, outv);
                        // this->app_->accumulate_atomic(deltas[e.neighbor], outv);
                        this->app_->accumulate(deltas[e.neighbor], outv);
                    }
                }
            }
        }
    }

    /* use a VertexArray */
    void inc_run_to_convergence(vid_t tid, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, vertex_t source){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<value_t, vid_t>& deltas = deltas_array[tid];
        // const vid_t ids_id = this->id2spids[source];
        int step = 0;
        float Diff = 0;
        //double now_termcheck_threshold = supernode_termcheck_threshold / node_set.size();
        size_t node_set_size = node_set.size();
        while (true) {
            step++;
            Diff = 0;
            // receive & send
            // for(auto v : node_set){
            for(vid_t i = 0; i < node_set_size; i++){
                const vertex_t& v = node_set[i];
                // auto to_send = atomic_exch(deltas[v], this->app_->default_v());
                auto to_send = deltas[v];
                auto last_value = values[v];
                deltas[v] = this->app_->default_v();
                if(to_send != this->app_->default_v()){
                    const auto& old_oes = new_graph->GetOutgoingAdjList(v);
                    const adj_list_t& oes = adj_list_t(ia_oe_offset_[v.GetValue()], ia_oe_offset_[v.GetValue()+1]);
                    auto& value = values[v];
                    value_t outv = 0;
                    #ifdef COUNT_ACTIVE_EDGE_NUM
                      atomic_add(this->app_->f_index_count_num, (long long)oes.Size());
                    #endif
                    for(auto& e : oes){
                    // for(auto& e : old_oes){
                    //     if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                            this->app_->g_function(*new_graph, v, value, to_send, old_oes, e, outv);
                            // this->app_->accumulate_atomic(deltas[e.neighbor], outv);
                            this->app_->accumulate(deltas[e.neighbor], outv);
                        // }
                    }
                    // this->app_->accumulate_atomic(value, to_send);
                    this->app_->accumulate(value, to_send);
                }
                Diff += fabs(last_value - values[v]);
            }
            // check convergence
            // if(Diff * node_set_size <= supernode_termcheck_threshold || step > 100){
            if(Diff <= supernode_termcheck_threshold * node_set_size || step > 100){
                break;
            }
        }
    }

public:
    // VertexArray<value_t, vid_t> init_deltas;
    std::vector<VertexArray<value_t, vid_t>> values_array; // use to calulate indexes in parallel
    std::vector<VertexArray<value_t, vid_t>> deltas_array;
    std::vector<VertexArray<value_t, vid_t>> bounds_array; // delta of bound vertex
    std::vector<std::vector<double>> test_time; // test time
    /* inner all nodes */
    Array<nbr_t, Allocator<nbr_t>> ia_oe_;
    Array<nbr_t*, Allocator<nbr_t*>> ia_oe_offset_;
    /* in_bound_node to out_bound_node */
    Array<nbr_t, Allocator<nbr_t>> ib_oe_;
    Array<nbr_t*, Allocator<nbr_t*>> ib_oe_offset_;
    VertexArray<bool, vid_t> is_update; // inc updated nodes
};

}  // namespace grape
#endif  // GRAPE_FRAGMENT_ITER_COMPRESSOR_H_