/*
    add mirror in cluster.
*/
#ifndef GRAPE_FRAGMENT_TRAV_COMPRESSOR_H_
#define GRAPE_FRAGMENT_TRAV_COMPRESSOR_H_

#include "grape/graph/super_node.h"
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include "timer.h"
#include "flags.h"
#include <iomanip>
#include "grape/fragment/compressor_base.h"

// #define NUM_THREADS 52

namespace grape {

template <typename APP_T, typename SUPERNODE_T>
class TravCompressor : public CompressorBase <APP_T, SUPERNODE_T> {
    public:
    using fragment_t = typename APP_T::fragment_t;
    using value_t = typename APP_T::value_t;
    using vertex_t = typename APP_T::vertex_t;
    using vid_t = typename APP_T::vid_t;
    using supernode_t = SUPERNODE_T;
    using delta_t = typename APP_T::delta_t;
    using fc_t = int32_t;
    using nbr_t = typename fragment_t::nbr_t;
    using adj_list_t = typename fragment_t::adj_list_t;
    size_t total_node_set = 0;
    const bool trav_compressor_flags_cilk = true;
    // int thread_num = FLAGS_build_index_concurrency;
    int thread_num = 52; // test increment computation

    TravCompressor(std::shared_ptr<APP_T>& app,
                        std::shared_ptr<fragment_t>& graph):CompressorBase<APP_T, SUPERNODE_T>(app, graph){}

    void init_array() {
        /* init */
        active_clusters.resize(this->cluster_ids.size(), 0);
        test_time.resize(thread_num);
        values_array.resize(thread_num);
        deltas_array.resize(thread_num);
        double s = GetCurrentTime();
        parallel_for(int tid = 0; tid < thread_num; tid++){
            auto all_nodes = VertexRange<vid_t>(0, this->all_node_num);
            values_array[tid].Init(all_nodes);
            deltas_array[tid].Init(all_nodes);
            test_time[tid].resize(4); // for debug
        }
    }

    void run(){
        this->supernode_out_bound.clear();
        this->supernode_out_bound.resize(this->graph_->GetVerticesNum(), 0);

        /* find supernode */
        {  
            this->compress();
            init_array();
            this->judge_out_bound_node(this->graph_);
            /* build subgraph of supernode */
            // build_subgraph(this->graph_);
            this->build_subgraph_mirror(this->graph_);
        }

        /* calculate index for each structure */
        double calculate_index = GetCurrentTime();
        {
            /* parallel */
            /* Simulate thread pool */
            std::atomic<vid_t> spnode_id(0);
            std::atomic<vid_t> active_thread_num(thread_num);
            this->ForEach(this->supernodes_num, [this, &spnode_id, &active_thread_num](int tid) {
                int i = 0, cnt = 0, step = 1;  // step need to be adjusted
                while(i < this->supernodes_num){
                    i = spnode_id.fetch_add(step);
                    for(int j = i; j < i + step; j++){
                        if(j < this->supernodes_num){
                            // build_trav_index(i, this->graph_, tid);
                            build_trav_index_mirror(i, this->graph_, tid);
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
        double start = GetCurrentTime();
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
        double subgraph_time = GetCurrentTime();
        const auto& inner_vertices = new_graph->InnerVertices();
        const vid_t spn_ids_num = this->supernode_ids.size();
        vid_t inner_node_num = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();
        std::vector<size_t> ia_oe_degree(inner_node_num+1, 0);
        vid_t ia_oe_num = 0; 
        parallel_for(vid_t i = 0; i < spn_ids_num; i++){
        // for(vid_t i = 0; i < spn_ids_num; i++){
            std::vector<vertex_t> &node_set = this->supernode_ids[i];
            vid_t temp_a = 0;
            for(auto v : node_set){
                auto ids_id = this->id2spids[v];
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                for(auto& oe : oes){
                    if(this->id2spids[oe.neighbor] == ids_id){ // inner edge
                        ia_oe_degree[v.GetValue()+1]++;
                        temp_a++;
                    }
                }
            }
            atomic_add(ia_oe_num, temp_a);
        }
        ia_oe_.clear();
        ia_oe_.resize(ia_oe_num);
        ia_oe_offset_.clear();
        ia_oe_offset_.resize(inner_node_num+1);

        for(vid_t i = 1; i < inner_node_num; i++) {
            ia_oe_degree[i] += ia_oe_degree[i-1];
        }

        parallel_for(vid_t i = 0; i < inner_node_num; i++) {
            vertex_t u(i);
            vid_t index_a = ia_oe_degree[i];
            ia_oe_offset_[i] = &ia_oe_[index_a];
            if(this->Fc[u] != this->FC_default_value){
                auto ids_id = this->id2spids[u];
                const auto& oes = new_graph->GetOutgoingAdjList(u);
                for(auto& oe : oes){
                    if(this->id2spids[oe.neighbor] == ids_id){ // inner edge
                        ia_oe_[index_a] = oe;
                        index_a++;
                    }
                }
            }
        }
        ia_oe_offset_[inner_node_num] = &ia_oe_[ia_oe_num-1] + 1;
    }

    /* use a VertexArray */
    void build_trav_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, vid_t tid){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        supernode_t& spnode = this->supernodes[spid];
        const auto& source = spnode.id;
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];

        /* init values/deltas */
        test_time[tid][0] -= GetCurrentTime();
        for(auto v : node_set){
            deltas[v] = this->app_->GetInitDelta(v, source);
            values[v] = this->app_->GetInitValue(v);
        }
        test_time[tid][0] += GetCurrentTime();
        /* iterative calculation */
        test_time[tid][1] -= GetCurrentTime();
        double b = GetCurrentTime();
        std::unordered_set<vertex_t> next_modified;
        next_modified.insert(node_set.begin(), node_set.end());
        run_to_convergence(tid, new_graph, node_set, next_modified, source);
        test_time[tid][3] = std::max(test_time[tid][3], GetCurrentTime()-b);
        test_time[tid][1] += GetCurrentTime();
        /* build new index in supernodes */
        test_time[tid][2] -= GetCurrentTime();
        fianl_build_trav_index(tid, new_graph, node_set, spnode);
        test_time[tid][2] += GetCurrentTime();
    }

    void build_trav_index_mirror(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, vid_t tid){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        supernode_t& spnode = this->supernodes[spid];
        // spnode.data = this->app_->GetIdentityElement();
        const auto& source = spnode.id;
        std::vector<vertex_t> &node_set = this->cluster_ids[spnode.ids]; // inner node id + mirror id
        std::vector<vertex_t> &inner_node_set = this->supernode_ids[spnode.ids];
        std::vector<vertex_t> &out_mirror_node_set = this->cluster_out_mirror_ids[spnode.ids]; // include mirror

        /* init values/deltas
            Note: In order to ensure the original dependencies, 
            mirror vertices must be converted to original vertices 
            for initialization.
         */
        test_time[tid][0] -= GetCurrentTime();
        vertex_t real_source = source;
        if (source.GetValue() >= this->old_node_num) {
            vertex_t v = this->mirrorid2vid[source];
            real_source = v;
            deltas[source] = this->app_->GetInitDelta(v, real_source);
            values[source] = this->app_->GetInitValue(v);
        }
        for(auto v : inner_node_set){
            deltas[v] = this->app_->GetInitDelta(v, real_source);
            values[v] = this->app_->GetInitValue(v);
        }
        for(auto m : out_mirror_node_set){
            vertex_t v = this->mirrorid2vid[m];
            deltas[m] = this->app_->GetInitDelta(v, real_source);
            values[m] = this->app_->GetInitValue(v);
        }
        test_time[tid][0] += GetCurrentTime();
        /* iterative calculation */
        test_time[tid][1] -= GetCurrentTime();
        double b = GetCurrentTime();
        std::unordered_set<vertex_t> next_modified;
        next_modified.insert(source); 
        run_to_convergence_mirror(tid, new_graph, node_set, next_modified, source);
        test_time[tid][3] = std::max(test_time[tid][3], GetCurrentTime()-b);
        test_time[tid][1] += GetCurrentTime();
        /* build new index in supernodes */
        test_time[tid][2] -= GetCurrentTime();
        fianl_build_trav_index_mirror(tid, new_graph, node_set, spnode);
        test_time[tid][2] += GetCurrentTime();
    }

    void run_to_convergence_mirror(vid_t tid, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, std::unordered_set<vertex_t>& next_modified_, vertex_t source){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        std::unordered_set<vertex_t> curr_modified_;
        int step = 0;
        // const vid_t ids_id = this->id2spids[source];

        /* in_mirror_source vertex first send a message to its neighbors */
        if (source.GetValue() >= this->old_node_num) {
            // When only the source vertex can be cleared. It cannot be 
            // completely cleared, otherwise it will cause the loss of 
            // active vertices in the correction phase.
            next_modified_.erase(source); 
            vertex_t v = this->mirrorid2vid[source];
            const auto& oes = new_graph->GetOutgoingAdjList(v);
            const auto& inner_oes = this->subgraph[source.GetValue()];
            auto& value = values[source];
            auto& to_send = deltas[source];
            if (this->app_->CombineValueDelta(value, to_send)) {
                auto out_degree = inner_oes.size();
                #ifdef COUNT_ACTIVE_EDGE_NUM
                  atomic_add(this->app_->f_index_count_num, (long long)inner_oes.size());
                #endif
                if(out_degree > 0){
                    // vertex_t v = this->mirrorid2vid[source];
                    for(auto e : inner_oes){
                        delta_t outv;
                        this->app_->Compute(v, value, to_send, oes, e, outv); 
                        if(this->app_->AccumulateDelta(deltas[e.neighbor], outv)){
                            next_modified_.insert(e.neighbor);
                        }
                    }
                }
            }
        }
        
        while (true) {
            step++;
            curr_modified_.clear();
            // receive & send
            for(auto v : next_modified_){
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                const auto& inner_oes = this->subgraph[v.GetValue()];
                auto& value = values[v];
                auto& to_send = deltas[v];
                if (this->app_->CombineValueDelta(value, to_send)) {
                    auto out_degree = inner_oes.size();
                    if(out_degree > 0){
                        #ifdef COUNT_ACTIVE_EDGE_NUM
                          atomic_add(this->app_->f_index_count_num, (long long)inner_oes.size());
                        #endif
                        for(auto e : inner_oes){
                            delta_t outv;
                            this->app_->Compute(v, value, to_send, oes, e, outv);
                            if(this->app_->AccumulateDelta(deltas[e.neighbor], outv) 
                                && e.neighbor.GetValue() < this->old_node_num){ // exclude mirror
                                curr_modified_.insert(e.neighbor);
                            }
                        }
                    }
                }
            }
            if(curr_modified_.size() == 0){
                break;
            }
            next_modified_.swap(curr_modified_);
        }
    }

    /* use VertexArray */
    void fianl_build_trav_index_mirror(vid_t tid, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, supernode_t& spnode){
        std::vector<vertex_t> &local_node_set = this->supernode_ids[spnode.ids];
        std::vector<vertex_t> &mirror_node_set = this->cluster_out_mirror_ids[spnode.ids];
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        // std::unordered_map<vid_t, delta_t> bound_map;
        spnode.inner_delta.clear();
        spnode.bound_delta.clear();
        spnode.status = true; 
        if (FLAGS_application == "cc") {
            vid_t co_id = values[source];
            for(auto v : node_set){
                auto& value = values[v];
                auto& delta = deltas[v];
                if(delta.value == co_id){
                    delta_t rt_delta;
                    this->app_->revCompute(delta, rt_delta);
                    if(this->supernode_out_bound[v.GetValue()]){
                        spnode.bound_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta));
                    }
                    else {
                        spnode.inner_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta)); 
                    }
                }
            }
        }
        else {
            for(auto v : local_node_set){
                auto& value = values[v];
                auto& delta = deltas[v];
                if(value != this->app_->GetIdentityElement()){
                    delta_t rt_delta;
                    this->app_->revCompute(delta, rt_delta);
                    if(this->supernode_out_bound[v.GetValue()]){
                        spnode.bound_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta));
                    }
                    else {
                        spnode.inner_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta)); 
                    }
                }
            }
            for(auto m : mirror_node_set){
                auto& value = values[m];
                auto& delta = deltas[m];
                if(delta.value != this->app_->GetIdentityElement()){ 
                    delta_t rt_delta;
                    vertex_t v = this->mirrorid2vid[m];
                    this->app_->revCompute(delta, rt_delta);
                    spnode.bound_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta));
                }
            }
        }
    }


    /* use VertexArray */
    void run_to_convergence(vid_t tid, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, std::unordered_set<vertex_t>& next_modified_, vertex_t source){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        std::unordered_set<vertex_t> curr_modified_;
        int step = 0;
        const vid_t ids_id = this->id2spids[source];
        
        while (true) {
            step++;
            curr_modified_.clear();
            // receive & send
            for(auto v : next_modified_){
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                auto& value = values[v];
                auto& to_send = deltas[v];
                if (this->app_->CombineValueDelta(value, to_send)) {
                    auto out_degree = oes.Size();
                    if(out_degree > 0){
                        #ifdef COUNT_ACTIVE_EDGE_NUM
                          atomic_add(this->app_->f_index_count_num, (long long)out_degree);
                        #endif
                        for(auto e : oes){
                            if(ids_id == this->id2spids[e.neighbor]){  // Only sent to internal vertices
                                delta_t outv;
                                this->app_->Compute(v, value, to_send, oes, e, outv);
                                if(this->app_->AccumulateDelta(deltas[e.neighbor], outv)){
                                    curr_modified_.insert(e.neighbor);
                                }
                            }
                        }
                    }
                }
            }
            // check convergence
            if(curr_modified_.size() == 0 || step > 2000){
                break;
            }
            next_modified_.swap(curr_modified_);
        }
    }

    /* use VertexArray */
    void fianl_build_trav_index(vid_t tid, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, supernode_t& spnode){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::unordered_map<vid_t, delta_t> bound_map;
        spnode.inner_delta.clear();
        spnode.bound_delta.clear();
        spnode.status = true; 
        if (FLAGS_application == "cc") {
            vid_t co_id = values[source];
            for(auto v : node_set){
                auto& value = values[v];
                auto& delta = deltas[v];
                if(delta.value == co_id){
                    delta_t rt_delta;
                    this->app_->revCompute(delta, rt_delta);
                    if(this->supernode_out_bound[v.GetValue()]){
                        spnode.bound_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta));
                    }
                    else {
                        spnode.inner_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta)); 
                    }
                }
            }
        }
        else {
            for(auto v : node_set){
                auto& value = values[v];
                auto& delta = deltas[v];
                if(value != this->app_->GetIdentityElement()){
                    delta_t rt_delta;
                    this->app_->revCompute(delta, rt_delta);
                    if(this->supernode_out_bound[v.GetValue()]){
                        spnode.bound_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta));
                    }
                    else {
                        spnode.inner_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta)); 
                    }
                }
            }
            // this->print();
        }
    }

    void precompute_spnode(const std::shared_ptr<fragment_t>& new_graph){
        /* if the source vertex is within the supernode and isn't the entry point. */
        double pre_compute = GetCurrentTime();
        if(FLAGS_application == "sssp" || FLAGS_application == "sswp" 
           || FLAGS_application == "bfs"){
            vertex_t source;
            bool native_source = new_graph->GetInnerVertex(FLAGS_sssp_source, source);
            if(native_source && this->Fc[source] < 0){ // inner node, must include bound node
                vid_t spids = this->id2spids[source];
                run_to_convergence_for_precpt(spids, new_graph);
                /* get active node from out_mirror */
                auto out_mirror_nodes = this->cluster_out_mirror_ids[spids];
                if (!out_mirror_nodes.empty()) {
                    vid_t size = out_mirror_nodes.size();
                    auto& deltas = this->app_->deltas_;
                    auto& values = this->app_->values_;
                    parallel_for(vid_t i = 0; i < size; i++) {
                        vertex_t mid = out_mirror_nodes[i];
                        vertex_t u = this->mirrorid2vid[mid];
                        bool is_update = this->app_->AccumulateDeltaAtomic(deltas[u], deltas[mid]);
                        deltas[mid] = this->app_->GetInitDelta(u); // reset using master
                        values[mid] = this->app_->GetInitValue(u); // reset using master
                        if (is_update) {
                            this->app_->curr_modified_.Insert(u);
                        }
                    }
                }
                /* get active node from out_bound_node of active supernode */
                std::vector<vertex_t> &node_set = this->supernode_ids[spids];
                for (auto u : node_set) {
                    if (this->supernode_out_bound[u.GetValue()]) {
                        this->app_->curr_modified_.Insert(u);
                    }
                }
            }
        } else if (FLAGS_application == "cc") {
            const vid_t spn_ids_num = this->supernode_ids.size();
            LOG(INFO) << "application cc spn_ids_num=" << spn_ids_num;
            #pragma cilk grainsize = 1
            parallel_for(vid_t i = 0; i < spn_ids_num; i++){
                run_to_convergence_for_precpt(i, new_graph);
            }
        } else {
            LOG(INFO) << "No this application.";
            exit(0);
        }
        pre_compute = GetCurrentTime() - pre_compute;
        LOG(INFO) << "#pre_compute: " << pre_compute;
    }

    void inc_precompute_spnode_mirror(const std::shared_ptr<fragment_t>& new_graph, 
                                      std::vector<char>& node_type) {
      this->ForEachCilkOfBitset(
        this->app_->curr_modified_, new_graph->InnerVertices(), [this, &new_graph,
                                              &node_type](int tid, vertex_t u) {
        if (node_type[u.GetValue()] != NodeType::SingleNode
            && node_type[u.GetValue()] != NodeType::OnlyInNode
            && node_type[u.GetValue()] != NodeType::BothOutInNode) {
          this->active_clusters[this->id2spids[u]] = 1; 
        }
      });

      auto& deltas = this->app_->deltas_;
      auto& values = this->app_->values_;
      size_t active_cluster_num = this->active_clusters.size();
      parallel_for(vid_t i = 0; i < active_cluster_num; i++) {
        if (this->active_clusters[i] == 1) {
          vid_t spids = i;
          // reset out-mirror /
          std::vector<vertex_t> &out_mirror_node_set = this->cluster_out_mirror_ids[spids]; // include mirror
          for(auto m : out_mirror_node_set){
            vertex_t v = this->mirrorid2vid[m];
            deltas[m] = this->app_->GetInitDelta(v); 
            values[m] = this->app_->GetInitValue(v); 
          }
          run_to_convergence_for_precpt(spids, new_graph);
        }
      }
      // get active node from out_mirror /
      parallel_for(vid_t i = 0; i < active_cluster_num; i++) {
        if (this->active_clusters[i] == 1) {
          vid_t spids = i;
          auto out_mirror_nodes = this->cluster_out_mirror_ids[spids];
          if (!out_mirror_nodes.empty()) {
            vid_t size = out_mirror_nodes.size();
            for(vid_t i = 0; i < size; i++) {
              vertex_t mid = out_mirror_nodes[i];
              vertex_t u = this->mirrorid2vid[mid];
              bool is_update = this->app_->AccumulateDeltaAtomic(deltas[u], 
                                                                deltas[mid]);
              deltas[mid] = this->app_->GetInitDelta(u); // reset using master
              values[mid] = this->app_->GetInitValue(u); // reset using master
              if (is_update) {
                this->app_->curr_modified_.Insert(u);
              }
            }
          }
          // get active node from out_bound_node of active supernode /
          std::vector<vertex_t> &node_set = this->supernode_ids[spids];
          for (auto u : node_set) {
            if (this->supernode_out_bound[u.GetValue()]) {
              this->app_->curr_modified_.Insert(u);
            }
          }
        }
      }
    }

    void precompute_spnode_all(const std::shared_ptr<fragment_t>& new_graph){
        const vid_t spn_ids_num = this->supernode_ids.size();
        #pragma cilk grainsize = 1
        parallel_for(vid_t i = 0; i < spn_ids_num; i++){
            run_to_convergence_for_precpt(i, new_graph);
        }
    }

    void Output() {
        auto& values = this->app_->values_;
        auto& deltas = this->app_->deltas_;
        LOG(INFO) << "-------print now result-------";
        // for (auto v : inner_vertices) {
        for (int i = 0; i < this->all_node_num; i++) {
            vertex_t v(i);
            LOG(INFO) << " oid=" << this->v2Oid(v) << " value=" << values[v] << " delta=" << deltas[v].value << std::endl;
        }
        LOG(INFO) << "==============================";
    }

    /**
     * spids: index of supernode_ids[]
    */
    void run_to_convergence_for_precpt(const vid_t spids, const std::shared_ptr<fragment_t>& new_graph){
        std::vector<vertex_t> &node_set = this->supernode_ids[spids];
        const vid_t ids_id = spids;

        // std::unordered_map<vid_t, delta_t> send_delta;
        std::unordered_set<vertex_t> curr_modified_; 
        std::unordered_set<vertex_t> next_modified_;
        int step = 0;
        next_modified_.insert(node_set.begin(), node_set.end());

        auto& values = this->app_->values_;
        auto& deltas = this->app_->deltas_;

        // Output(new_graph);
        while (true) {
            step++;
            curr_modified_.clear();
            // receive & send
            for(auto v : next_modified_){
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                const auto& inner_oes = this->subgraph[v.GetValue()];
                auto& value = values[v];
                auto& to_send = deltas[v];

                if (this->app_->CombineValueDelta(value, to_send)) {
                    this->app_->active_entry_node_[v] = 1;
                    // auto out_degree = oes.Size();
                    auto out_degree = inner_oes.size();
                    if(out_degree > 0){
                        // for(auto e : oes){
                        #ifdef COUNT_ACTIVE_EDGE_NUM
                          atomic_add(this->app_->f_index_count_num, (long long)out_degree);
                        #endif
                        for(auto e : inner_oes){ // Only sent to internal points
                            delta_t outv;
                            this->app_->Compute(v, value, to_send, oes, e, outv);
                            bool is_update = this->app_->AccumulateDelta(deltas[e.neighbor], outv);
                            if(is_update 
                                && e.neighbor.GetValue() < this->old_node_num){ // exclude mirror
                                curr_modified_.insert(e.neighbor);
                            }
                        }
                    }
                }
            }
            // check convergence
            if(curr_modified_.size() == 0){ 
                break;
            }
            next_modified_.swap(curr_modified_);
        }
    }

    void inc_run(std::vector<std::pair<vid_t, vid_t>>& deleted_edges, std::vector<std::pair<vid_t, vid_t>>& added_edges, const std::shared_ptr<fragment_t>& new_graph){
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

        /* find supernode */
        double inc_compress = GetCurrentTime();
        // this->inc_trav_compress_mirror(deleted_edges, added_edges, new_graph);
        this->inc_compress_mirror(deleted_edges, added_edges, new_graph);
        inc_compress = GetCurrentTime()-inc_compress;
        LOG(INFO) << "#inc_compress: " << inc_compress;

        /* init supernode_out_bound*/
        double begin = GetCurrentTime();
        // this->supernode_out_bound.clear();
        this->supernode_out_bound.resize(this->graph_->GetVerticesNum(), 0);
        this->inc_judge_out_bound_node(new_graph);
        /* build subgraph of supernode */
        this->inc_build_subgraph_mirror(new_graph);
        LOG(INFO) << "#inc_build_subgraph_mirror: " << (GetCurrentTime() - begin);

        this->get_reset_edges(deleted_edges, new_graph);
        /* calculate index for each structure */
        double inc_calculate_index = GetCurrentTime();
        // inc_compute_index_mirror_cid(new_graph);  // inc_trav_compress_mirror
        inc_compute_index_mirror_spid(new_graph);    // inc_compress_mirror
        inc_calculate_index = GetCurrentTime() - inc_calculate_index;
        LOG(INFO) << "#inc_calculate_index: " << inc_calculate_index;
    }

    void get_reset_edges(
            std::vector<std::pair<vid_t, vid_t>>& deleted_edges, 
            const std::shared_ptr<fragment_t>& new_graph){

        fid_t fid = this->graph_->fid();
        auto vm_ptr = this->graph_->vm_ptr();
        reset_spnode_edges.clear();
        vid_t cluster_size = this->cluster_ids.size();
        reset_spnode_edges.resize(cluster_size);

        for(auto& pair : deleted_edges) {
          auto u_gid = pair.first;
          auto v_gid = pair.second;
          fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                v_fid = vm_ptr->GetFidFromGid(v_gid);
          // u -> v
          vertex_t u;
          CHECK(this->graph_->Gid2Vertex(u_gid, u));
          vertex_t v;
          CHECK(this->graph_->Gid2Vertex(v_gid, v));
          if (u_fid != v_fid) {
              continue;
          }
          vid_t u_ids = this->id2spids[u];
          vid_t v_ids = this->id2spids[v];
          if (u_ids == v_ids && u_ids != this->ID_default_value) {
              reset_spnode_edges[u_ids].emplace_back(
                                        std::pair<vid_t, vid_t>(u_gid, v_gid));
          } else{ 
            if (v_ids != this->ID_default_value) {
              auto vc_in_mirrors = this->supernode_in_mirror[v_ids]; 
              if (vc_in_mirrors.find(u) != vc_in_mirrors.end()) {
                  reset_spnode_edges[v_ids].emplace_back(
                                        std::pair<vid_t, vid_t>(u_gid, v_gid));
              }
            }
          } 
        }
    }

    void inc_compute_index_mirror_cid(const std::shared_ptr<fragment_t>& new_graph) {
      /* 1 */
      // parallel_for(vid_t i = 0; i < this->update_cluster_ids.size(); i++) { 
      //   vid_t tid = 0;
      //   vid_t ids_id = this->update_cluster_ids[i];
      //   for(auto ms : this->cluster_in_mirror_ids[ids_id]) {
      //     vid_t spid = this->Fc_map[ms];
      //     inc_build_trav_index_mirror(spid, new_graph, tid);
      //     // build_iter_index_mirror(spid, new_graph, tid);
      //   }
      //   for(auto vs : this->supernode_source[ids_id]) {
      //     vid_t spid = this->Fc_map[vs];
      //     inc_build_trav_index_mirror(spid, new_graph, tid);
      //   }
      // }

      /* 2 supernode */
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
                inc_build_trav_index_mirror(id, new_graph, tid);
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
                inc_build_trav_index_mirror(spid, new_graph, tid);
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

    void re_build_trav_index_mirror(const vid_t spid,
                     const std::shared_ptr<fragment_t>& new_graph, vid_t tid) {
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        supernode_t& spnode = this->supernodes[spid];
        vid_t ids_id = spnode.ids;
        const auto& source = spnode.id;
        std::vector<vertex_t> &node_set = this->cluster_ids[ids_id]; // inner node id + mirror id
        std::vector<vertex_t> &inner_node_set = this->supernode_ids[ids_id];
        std::vector<vertex_t> &out_mirror_node_set = 
                                        this->cluster_out_mirror_ids[ids_id]; // include mirror

        /* init values/deltas
            Note: In order to ensure the original dependencies, 
            mirror vertices must be converted to original vertices 
            for initialization.
         */
        vertex_t real_source = source;
        if (source.GetValue() >= this->old_node_num) {
            vertex_t v = this->mirrorid2vid[source];
            real_source = v;
            deltas[source] = this->app_->GetInitDelta(v, real_source);
            values[source] = this->app_->GetInitValue(v);
        }
        for(auto v : inner_node_set){
            deltas[v] = this->app_->GetInitDelta(v, real_source);
            values[v] = this->app_->GetInitValue(v);
        }
        for(auto m : out_mirror_node_set){
            vertex_t v = this->mirrorid2vid[m];
            deltas[m] = this->app_->GetInitDelta(v, real_source);
            values[m] = this->app_->GetInitValue(v);
        }

        /* iterative calculation */
        std::unordered_set<vertex_t> next_modified;
        next_modified.insert(source);
        run_to_convergence_mirror(tid, new_graph, node_set, next_modified, source);
        /* build new index in supernodes */
        fianl_build_trav_index_mirror(tid, new_graph, node_set, spnode);
    }

    void inc_build_trav_index_mirror(const vid_t spid,
                     const std::shared_ptr<fragment_t>& new_graph, vid_t tid) {
        supernode_t& spnode = this->supernodes[spid];
        if (spnode.status == false 
            || this->cluster_ids[spnode.ids].size() < 10) {
          re_build_trav_index_mirror(spid, new_graph, tid); // rebuild shourtcut
          return ;
        }
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        vid_t ids_id = spnode.ids;
        const auto& source = spnode.id;
        std::vector<vertex_t> &node_set = this->cluster_ids[ids_id]; // inner node id + mirror id
        std::vector<vertex_t> &inner_node_set = this->supernode_ids[ids_id];
        std::vector<vertex_t> &out_mirror_node_set = 
                                        this->cluster_out_mirror_ids[ids_id]; // out-mirror mid
        /* init values/deltas
            Note: In order to ensure the original dependencies, 
            mirror vertices must be converted to original vertices 
            for initialization.
         */
        vertex_t real_source = source;
        for(auto v : inner_node_set){
            deltas[v] = this->app_->GetInitDelta(v, real_source);
            values[v] = this->app_->GetInitValue(v);
        }
        for(auto m : out_mirror_node_set){
            vertex_t v = this->mirrorid2vid[m];
            deltas[m] = this->app_->GetInitDelta(v, real_source);
            values[m] = this->app_->GetInitValue(v);
        }
        if (source.GetValue() >= this->old_node_num) {
            vertex_t v = this->mirrorid2vid[source];
            real_source = v;
            deltas[source] = this->app_->GetInitDelta(v, real_source);
            values[source] = this->app_->GetInitValue(v);
        } else {
          deltas[source] = this->app_->GetInitDelta(source, source);
          values[source] = this->app_->GetInitValue(source);
        }

        /* get init value from old shortcuts, 
            inner_delta is null if this is a new source.
            In the old shortcut, the mirror node is converted to 
            the corresponding master node. So filter out Mirror's shortcut,
            out-mirror's shortcut will be rebuilded. In-mirror's old value 
            will be lost. CC will error????
        */
        for(auto e : spnode.inner_delta){
            if (this->id2spids[e.first] == ids_id) {
                values[e.first] = e.second.value;  // The value of delta is used directly.
                deltas[e.first] = e.second;
            }
        }
        for(auto e : spnode.bound_delta){
            if (this->id2spids[e.first] == ids_id) {
                values[e.first] = e.second.value;  // The value of delta is used directly.
                deltas[e.first] = e.second;
            }
        }

        /* reset by dependency */
        std::unordered_set<vertex_t> curr_modified, next_modified;
        for (auto& pair : reset_spnode_edges[ids_id]) {
            vid_t u_gid = pair.first, v_gid = pair.second;
            vertex_t v, u;
            CHECK(this->graph_->Gid2Vertex(u_gid, u)); 
            CHECK(this->graph_->Gid2Vertex(v_gid, v));

            auto parent_gid = deltas[v].parent_gid;

            if (parent_gid == u_gid && v != source) {
                curr_modified.insert(v);
            }
        }
        while (curr_modified.size() > 0){
          for(auto u : curr_modified){
            // LOG(INFO) << " reset: u.oid=" << this->v2Oid(u);
            auto u_gid = this->graph_->Vertex2Gid(u);
            // auto& oes = this->graph_->GetOutgoingAdjList(u); // old graph //
            auto& oes = this->subgraph_old[u.GetValue()];
            // const auto& inner_oes = this->subgraph[v.GetValue()]; // new graph
            for (auto e : oes) {
              auto v = e.neighbor;
              // if (this->id2spids[v] == ids_id  
              //     && v.GetValue() < this->old_node_num) { // id2spids[] 
              if (v.GetValue() < this->old_node_num) {
                auto parent_gid = deltas[v].parent_gid;
                if (parent_gid == u_gid && u != v) {
                    next_modified.insert(v);
                }
              }
            }
            values[u] = this->app_->GetInitValue(u);
            deltas[u] = this->app_->GetInitDelta(u); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
            this->app_->CombineValueDelta(values[u], deltas[u]);
          }
          curr_modified.clear();
          curr_modified.swap(next_modified);
        }
        // LOG(INFO) << "---------test-------------------";
        // Start a round without any condition on new_graph
        for(auto v : inner_node_set){
          const auto& oes = new_graph->GetOutgoingAdjList(v);
          const auto& inner_oes = this->subgraph[v.GetValue()]; // new subgraph
          auto& value = values[v];
          auto& delta = deltas[v];

          auto out_degree = inner_oes.size();
          if(out_degree > 0 && delta.value != this->app_->GetIdentityElement()){
            #ifdef COUNT_ACTIVE_EDGE_NUM
              atomic_add(this->app_->f_index_count_num, (long long)out_degree);
            #endif
            for(auto e : inner_oes){
              delta_t outv;
              this->app_->Compute(v, value, delta, oes, e, outv); // pagerank is oes.
              if(this->app_->AccumulateDelta(deltas[e.neighbor], outv)
                && e.neighbor.GetValue() < this->old_node_num){
                next_modified.insert(e.neighbor);
              }
            } 
          }
        }

        /* iterative calculation */
        next_modified.insert(source);
        run_to_convergence_mirror(tid, new_graph, node_set, next_modified, source);
        /* build new index in supernodes */
        fianl_build_trav_index_mirror(tid, new_graph, node_set, spnode);
    }

    void inc_trav_compress(std::vector<std::pair<vid_t, vid_t>>& deleted_edges, std::vector<std::pair<vid_t, vid_t>>& added_edges){
        fid_t fid = this->graph_->fid();
        auto vm_ptr = this->graph_->vm_ptr();
        this->inccalculate_spnode_ids.clear();
        this->recalculate_spnode_ids.clear();
        reset_edges.clear();
        for(auto& pair : deleted_edges) {
            auto u_gid = pair.first;
            auto v_gid = pair.second;
            fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                  v_fid = vm_ptr->GetFidFromGid(v_gid);
            // u -> v
            vertex_t u;
            CHECK(this->graph_->Gid2Vertex(u_gid, u));
            vertex_t v;
            CHECK(this->graph_->Gid2Vertex(v_gid, v));
            if(u_fid == fid && this->Fc[u] != this->FC_default_value){
                reset_edges.template emplace_back(u.GetValue(), v.GetValue());
            }
            if(v_fid == fid && this->Fc[v] != this->FC_default_value){
                vid_t src_id = this->Fc[v] < 0 ? (-this->Fc[v]-1) : this->Fc[v];
                std::vector<vertex_t>& src = this->supernode_source[src_id];
                vid_t del_id = this->Fc_map[src[0]];
                supernode_t& spnode = this->supernodes[del_id];
                const vid_t ids_id = this->id2spids[v];
                if(ids_id != this->id2spids[u] && src.size() > 1){
                    CHECK(this->Fc[v] >= 0);
                    const auto& ies = this->graph_->GetIncomingAdjList(v);
                    bool hava_out_inadj = false;
                    for(auto& e : ies){
                        auto& nb = e.neighbor;
                        if(nb != u && ids_id != this->id2spids[nb]){
                            hava_out_inadj = true;
                            break;
                        }
                    }
                    if(hava_out_inadj == false){
                        this->delete_supernode(v);
                    }
                }
            }
        }
        for(auto& pair : added_edges){
            auto u_gid = pair.first;
            auto v_gid = pair.second;
            fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                  v_fid = vm_ptr->GetFidFromGid(v_gid);
            // u -> v
            // LOG(INFO) << u_gid << "->" << v_gid;
            vertex_t u;
            CHECK(this->graph_->Gid2Vertex(u_gid, u));
            if(u_fid == fid && this->Fc[u]!= this->FC_default_value){
                // LOG(INFO) << graph_->GetId(u);
                vid_t src_id = this->Fc[u] < 0 ? (-this->Fc[u]-1) : this->Fc[u];
                for(auto source : this->supernode_source[src_id]){
                    this->inccalculate_spnode_ids.insert(source.GetValue());
                }
            }
            vertex_t v;
            CHECK(this->graph_->Gid2Vertex(v_gid, v));
            if(v_fid == fid && this->Fc[v] != this->FC_default_value){
                vid_t src_id = this->Fc[v] < 0 ? (-this->Fc[v]-1) : this->Fc[v];
                std::vector<vertex_t>& src = this->supernode_source[src_id];
                supernode_t& spnode = this->supernodes[this->Fc_map[src[0]]];
                auto& spids = this->supernode_ids[spnode.ids];
                const vid_t ids_id = this->id2spids[spnode.id];
                if(this->Fc[v] < 0 && ids_id != this->id2spids[u]){ // not a source, build a new spnode
                    this->Fc[v] = src_id;
                    this->supernode_source[src_id].emplace_back(v);
                    // build a new spnode idnex
                    vid_t supernoed_id = this->supernodes_num;
                    this->Fc_map[v] = supernoed_id;
                    this->supernodes[supernoed_id].id = v;
                    this->supernodes[supernoed_id].ids = spnode.ids;
                    this->supernodes_num++;
                    this->recalculate_spnode_ids.insert(v.GetValue());
                }
            }
        }
    }


    void inc_compute_index(const std::shared_ptr<fragment_t>& new_graph){
        /* case 1: Reset the index value according to the reset_edges */
        reset_spnode_edges.resize(this->supernodes_num);
        had_reset.clear();
        had_reset.resize(this->supernodes_num, 0);
        if_touch.clear();
        if_touch.resize(this->supernodes_num, 0);
        for(auto pair : reset_edges){
            auto u_id = pair.first;
            auto v_id = pair.second;
            vertex_t u(u_id);
            vid_t src_id = this->Fc[u] < 0 ? (-this->Fc[u]-1) : this->Fc[u];
            auto& srcs = this->supernode_source[src_id];
            for(auto source: srcs){
                reset_spnode_edges[this->Fc_map[source]].push_back(std::pair<vid_t, vid_t>(u_id, v_id));
            }
        }
        {
            std::vector<vid_t> ids;
            vid_t i = 0;
            for(auto e : reset_spnode_edges){
                if(int(e.size()) > 0){
                    ids.emplace_back(i);
                }
                i++;
            }
            int len = ids.size();
            if (trav_compressor_flags_cilk) {
#ifdef INTERNAL_PARALLEL
          LOG(FATAL) << "Layph is not compiled with -DUSE_CILK";
#endif
                parallel_for(vid_t i = 0; i < len; i++){
                    int tid = __cilkrts_get_worker_number();
                    reset_inc_build_tarv_index(ids[i], new_graph, tid); // iterative: pr, php
                    if(i % 1000000 == 0){
                        LOG(INFO) << "----id=" << i << " computing index" << std::endl;
                    }
                }
            }
            else{
                this->ForEachIndex(len, [this, &ids, &new_graph](int tid, vid_t begin, vid_t end) {
                    for(vid_t i = begin; i < end; i++){
                        reset_inc_build_tarv_index(ids[i], new_graph, tid);
                    }
                    }, FLAGS_build_index_concurrency
                );
                MPI_Barrier(this->comm_spec_.comm()); 
            }
            LOG(INFO) << "finish reset_inc_build_tarv_index... len=" << len;
        }

        /* case 2: inc-recalculate the index value according to the inccalculate_spnode_ids */
        {
            std::vector<vid_t> ids;
            ids.insert(ids.begin(), this->inccalculate_spnode_ids.begin(), this->inccalculate_spnode_ids.end());
            int len = ids.size();
            if (trav_compressor_flags_cilk) {
#ifdef INTERNAL_PARALLEL
          LOG(FATAL) << "Layph is not compiled with -DUSE_CILK";
#endif
                parallel_for(vid_t i = 0; i < len; i++){
                    vertex_t u(ids[i]);
                    int tid = __cilkrts_get_worker_number();
                    // Note: It needs to be judged here, because the index of u as the entry may have been deleted.
                    if(this->Fc_map[u] != this->ID_default_value){
                        inc_build_trav_index(this->Fc_map[u], new_graph, tid);
                    }
                }
            }
            else{
                this->ForEachIndex(len, [this, &ids, &new_graph](int tid, vid_t begin, vid_t end) {
                    for(vid_t i = begin; i < end; i++){
                        vertex_t u(ids[i]);
                        if(this->Fc_map[u] != this->ID_default_value && !had_reset[this->Fc_map[u]]){
                            inc_build_trav_index(this->Fc_map[u], new_graph, tid);
                        }
                    }
                    }, FLAGS_build_index_concurrency
                );
                MPI_Barrier(this->comm_spec_.comm());
            }
        }
        
        /* case 3: recalculate the newly created index according to the recalculate_spnode_ids */
        {
            std::vector<vid_t> ids;
            ids.insert(ids.begin(), this->recalculate_spnode_ids.begin(), this->recalculate_spnode_ids.end());
            int len = ids.size();
            if (trav_compressor_flags_cilk) {
#ifdef INTERNAL_PARALLEL
          LOG(FATAL) << "Layph is not compiled with -DUSE_CILK";
#endif
                parallel_for(vid_t i = 0; i < len; i++){
                    vertex_t u(ids[i]);
                    if(this->Fc_map[u] != this->ID_default_value){
                        int tid = __cilkrts_get_worker_number();
                        build_trav_index(this->Fc_map[u], new_graph, tid);
                    }
                    if(i % 1000000 == 0){
                        LOG(INFO) << "----id=" << i << " computing index" << std::endl;
                    }
                }
            }
            else{
                this->ForEachIndex(len, [this, &ids, &new_graph](int tid, vid_t begin, vid_t end) {
                    // LOG(INFO) << "build index, tid=" << tid << " range: "<< begin << "->" << end;
                    for(vid_t i = begin; i < end; i++){
                        vertex_t u(ids[i]);
                        // build_trav_index(i, this->graph_, values_array[tid], deltas_array[tid]);
                        if(this->Fc_map[u] != this->ID_default_value){
                            build_trav_index(this->Fc_map[u], new_graph, tid);
                        }
                    }
                    // LOG(INFO) << "tid=" << tid << " finish inc_build_trav_index!";
                    }, FLAGS_build_index_concurrency
                );
                MPI_Barrier(this->comm_spec_.comm());
            }
            LOG(INFO) << "finish recalculate build_trav_index... len=" << len;
        }
    }

    /* use a VertexArray */
    void reset_inc_build_tarv_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, vid_t tid){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        supernode_t& spnode = this->supernodes[spid];
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];

        /* 
            init values/deltas by inner_delta and inner_value, 
            Note that the inner_value and inner_delta at this time have been converted into weights 
        */
        for(auto v : node_set){
            values[v] = this->app_->GetInitValue(v);
            deltas[v] = this->app_->GetInitDelta(v); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
            this->app_->CombineValueDelta(values[v], deltas[v]);
        }
        for(auto e : spnode.inner_delta){
            values[e.first] = e.second.value;  // The value of delta is used directly.
            deltas[e.first] = e.second;
        }
        for(auto e : spnode.bound_delta){
            values[e.first] = e.second.value;  // The value of delta is used directly.
            deltas[e.first] = e.second;
        }
        
        /* reset by dependency */
        std::unordered_set<vertex_t> curr_modified, next_modified;
        for (auto& pair : reset_spnode_edges[spid]) {
            vid_t u_id = pair.first, v_id = pair.second;

            vertex_t u(u_id), v(v_id);
            auto parent_gid = deltas[v].parent_gid;

            if (parent_gid == this->graph_->Vertex2Gid(u)) {
                curr_modified.insert(v);
            }
        }
        if(curr_modified.size() == 0){ // bound_delta maybe have been chaged, so, can't return.
            return;  
        }
        while (curr_modified.size() > 0){
            for(auto u : curr_modified){
                auto u_gid = this->graph_->Vertex2Gid(u);
                auto oes = this->graph_->GetOutgoingAdjList(u);

                for (auto e : oes) {
                    auto v = e.neighbor;
                    auto parent_gid = deltas[v].parent_gid;
                    if (parent_gid == u_gid) {
                        next_modified.insert(v);
                    }
                }

                values[u] = this->app_->GetInitValue(u);
                deltas[u] = this->app_->GetInitDelta(u); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
                this->app_->CombineValueDelta(values[u], deltas[u]);
            }

            curr_modified.clear();
            curr_modified.swap(next_modified);
        }

        // Start a round without any condition on new_graph
        for(auto v : node_set){
            const auto& oes = new_graph->GetOutgoingAdjList(v);
            auto& value = values[v];
            auto& delta = deltas[v];

            auto out_degree = oes.Size();
            if(out_degree > 0 && delta.value != this->app_->GetIdentityElement()){
                #ifdef COUNT_ACTIVE_EDGE_NUM
                  atomic_add(this->app_->f_index_count_num, (long long)out_degree);
                #endif
                for(auto e : oes){
                    if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                        delta_t outv;
                        this->app_->Compute(v, value, delta, oes, e, outv);
                        if(this->app_->AccumulateDelta(deltas[e.neighbor], outv)){
                            next_modified.insert(e.neighbor);
                        }
                    }
                } 
            }
        }
        
        had_reset[spid] = 1; // Corresponds to the conditions of stage 2
        if_touch[spid] = 1;
        /* iterative calculation */
        run_to_convergence(tid, new_graph, node_set, next_modified, source);
        /* build new index in supernodes */
        fianl_build_trav_index(tid, new_graph, node_set, spnode);
    }

    /* use a VertexArray */
    void inc_build_trav_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, vid_t tid){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        supernode_t& spnode = this->supernodes[spid];
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];

        /* 
            init values/deltas by inner_delta and inner_value, 
            Note that the inner_value and inner_delta at this time have been converted into weights 
        */
        for(auto v : node_set){
            // values[v] = this->app_->GetIdentityElement();
            // deltas[v].Reset(this->app_->GetIdentityElement());
            values[v] = this->app_->GetInitValue(v);
            deltas[v] = this->app_->GetInitDelta(v); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
            this->app_->CombineValueDelta(values[v], deltas[v]);
        }

        for(auto e : spnode.inner_delta){
            values[e.first] = e.second.value;  // The value of delta is used directly.
            deltas[e.first] = e.second;
        }
        for(auto e : spnode.bound_delta){
            values[e.first] = e.second.value;  // The value of delta is used directly.
            deltas[e.first] = e.second;
        }

        // Start a round without any condition on new_graph
        std::unordered_set<vertex_t> next_modified;
        for(auto v : node_set){
            const auto& oes = new_graph->GetOutgoingAdjList(v);
            auto& value = values[v];
            auto& delta = deltas[v];

            auto out_degree = oes.Size();
            if(out_degree > 0 && delta.value != this->app_->GetIdentityElement()){
                #ifdef COUNT_ACTIVE_EDGE_NUM
                  atomic_add(this->app_->f_index_count_num, (long long)out_degree);
                #endif
                for(auto e : oes){
                    if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                        delta_t outv;
                        this->app_->Compute(v, value, delta, oes, e, outv);
                        if(this->app_->AccumulateDelta(deltas[e.neighbor], outv)){
                            next_modified.insert(e.neighbor);
                        }
                    }
                }
            }
        }

        /* iterative calculation */
        if_touch[spid] = 1;
        run_to_convergence(tid, new_graph, node_set, next_modified, source);
        fianl_build_trav_index(tid, new_graph, node_set, spnode);
    }

    /* use a VertexArray */
    void inc_build_trav_index2(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, vid_t tid){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        supernode_t& spnode = this->supernodes[spid];
        // spnode.data = this->app_->GetIdentityElement(); // ??
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];

        /* 
            init values/deltas by inner_delta and inner_value, 
            Note that the inner_value and inner_delta at this time have been converted into weights 
        */
        for(auto v : node_set){
            values[v] = this->app_->GetInitValue(v);
            deltas[v] = this->app_->GetInitDelta(v); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
            this->app_->CombineValueDelta(values[v], deltas[v]);
        }
        for(auto e : spnode.inner_delta){
            values[e.first] = e.second.value;  // The value of delta is used directly.
            deltas[e.first] = e.second;
        }
        for(auto e : spnode.bound_delta){
            values[e.first] = e.second.value;  // The value of delta is used directly.
            deltas[e.first] = e.second;
        }

        // // Start a round without any condition on new_graph
        std::unordered_set<vertex_t> next_modified;
        for(auto v : node_set){
            const auto& oes = new_graph->GetOutgoingAdjList(v);
            auto& value = values[v];
            auto& delta = deltas[v];
            auto out_degree = oes.Size();
            if(out_degree > 0 && delta.value != this->app_->GetIdentityElement()){
                #ifdef COUNT_ACTIVE_EDGE_NUM
                  atomic_add(this->app_->f_index_count_num, (long long)out_degree);
                #endif
                for(auto e : oes){
                    if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                        delta_t outv;
                        this->app_->Compute(v, value, delta, oes, e, outv);
                        if(this->app_->AccumulateDelta(deltas[e.neighbor], outv)){
                            next_modified.insert(e.neighbor);
                        }
                    }
                }
            }
        }

        if_touch[spid] = 1;
        run_to_convergence(tid, new_graph, node_set, next_modified, source);
        /* build new index in supernodes */
        fianl_build_trav_index(tid, new_graph, node_set, spnode);
    }

public:
    std::vector<std::pair<vid_t, vid_t>> reset_edges;
    std::vector<std::vector<std::pair<vid_t, vid_t>> > reset_spnode_edges;
    std::vector<short int> had_reset; // Mark whether the super point is reset and calculated
    std::vector<short int> if_touch; // Mark whether to update the inside of the superpoint
    std::vector<char> active_clusters;
    std::vector<VertexArray<value_t, vid_t>> values_array; // use to calulate indexes in parallel
    std::vector<VertexArray<delta_t, vid_t>> deltas_array;
    std::vector<vertex_t> active_nodes; // recode active master nodes of out_mirror.

    // VertexArray<delta_t, vid_t> init_deltas;
    std::vector<std::vector<double>> test_time; // test time
    /* inner all nodes */
    Array<nbr_t, Allocator<nbr_t>> ia_oe_;
    Array<nbr_t*, Allocator<nbr_t*>> ia_oe_offset_;
    /* in_bound_node to out_bound_node */
    // Array<nbr_t, Allocator<nbr_t>> ib_oe_;
    // Array<nbr_t*, Allocator<nbr_t*>> ib_oe_offset_;
};

}  // namespace grape
#endif  // GRAPE_FRAGMENT_TRAV_COMPRESSOR_H_