#ifndef GRAPE_FRAGMENT_COMPRESSOR_BASE_H_
#define GRAPE_FRAGMENT_COMPRESSOR_BASE_H_

#include "grape/graph/super_node.h"
#include "grape/utils/Queue.h"
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include "timer.h"
#include "flags.h"
#include <iomanip>
#include "omp.h"
// #include <metis.h>

#define NUM_THREADS omp_get_max_threads()

namespace grape {

template <typename APP_T, typename SUPERNODE_T>
class CompressorBase : public ParallelEngine{
    public:
    using fragment_t = typename APP_T::fragment_t;
    using value_t = typename APP_T::value_t;
    using delta_t = typename APP_T::delta_t;
    using vertex_t = typename APP_T::vertex_t;
    using vid_t = typename APP_T::vid_t;
    using supernode_t = SUPERNODE_T;
    using fc_t = int32_t;
    using nbr_t = typename fragment_t::nbr_t;
    using nbr_index_t = Nbr<vid_t, delta_t>;
    using adj_list_t = typename fragment_t::adj_list_t;
    // using adj_list_index_t = AdjList<vid_t, value_t>;
    using adj_list_index_t = AdjList<vid_t, delta_t>; // for inc-sssp

    CompressorBase(std::shared_ptr<APP_T>& app,
                        std::shared_ptr<fragment_t>& graph)
      : app_(app), graph_(graph) {}

    void init(const CommSpec& comm_spec, const Communicator& communicator,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()){
        comm_spec_ = comm_spec;
        communicator_ = communicator;
        InitParallelEngine(pe_spec);
        /* init */
        vid_t nodes_num = graph_->GetVerticesNum();
        // Fc.resize(nodes_num, FC_default_value);
        Fc.Init(graph_->Vertices(), FC_default_value);
        // Fc_map.Init(graph_->InnerVertices(), ID_default_value);
        id2spids.Init(graph_->Vertices(), ID_default_value);
        supernodes = new supernode_t[nodes_num];
        vid2in_mirror_cluster_ids.resize(nodes_num);
        vid2in_mirror_mids.resize(nodes_num);
        vid2out_mirror_mids.resize(nodes_num);
        // out_mirror2spids.resize(nodes_num);
        shortcuts.resize(nodes_num);
        old_node_num = nodes_num;
        all_node_num = nodes_num;
    }

    void print(std::string pos=""){
        LOG(INFO) << "--------------------------------print in " << pos;
        LOG(INFO) << "supernodes_num=" << supernodes_num << " ids_num=" << supernode_ids.size();
        for(vid_t i = 0; i < supernodes_num; i++){
            supernode_t& spn = supernodes[i];
            if (spn.id.GetValue() < old_node_num) {
              std::cout << "source_oid=" << graph_->GetId(spn.id) 
                        << " vid=" << spn.id.GetValue() << std::endl;
            } else {
              std::cout << "source_mirror_vid=" << spn.id.GetValue() << std::endl;
              std::cout << "  to_master_oid=" 
                        << graph_->GetId(mirrorid2vid[spn.id])
                        << "  master_vid=" << mirrorid2vid[spn.id].GetValue() << std::endl;
            }
            std::cout << " Fc_map=" << Fc_map[spn.id] 
                      << " spid=" << i << std::endl;
            std::cout << " ids_id=" << id2spids[spn.id];
            std::cout << " size=" << supernode_ids[spn.ids].size() << std::endl;
            for(auto u : supernode_ids[spn.ids]){
                std::cout << graph_->GetId(u) << " ";
            }
            std::cout << "\nFc:" << std::endl;
            for(auto fc : Fc){
                std::cout << fc << " ";
            }
            std::cout << std::endl;
            // LOG(INFO) << "inner_value:" << std::endl;
            // for(auto edge : spn.inner_value){
            //     LOG(INFO) << graph_->GetId(edge.first) << ": " << edge.second << std::endl;
            // }
            std::cout << "inner_delta: size=" << spn.inner_delta.size() << std::endl;
            for(auto edge : spn.inner_delta){
                std::cout << "gid:" << this->v2Oid(edge.first) 
                          << ": " << edge.second << std::endl;
            }
            std::cout << "bound_delta: size=" << spn.bound_delta.size() << std::endl;
            for(auto edge : spn.bound_delta){
                std::cout << "gid:" << this->v2Oid(edge.first) 
                          << ": " << edge.second << std::endl;
            }
            std::cout << "-----------------------------------------" << std::endl;
        }
    }

    void write_spnodes(const std::string &efile){
        std::ofstream outfile(efile);
        if(!outfile){
            LOG(INFO) << "open file failed. " << efile;
            exit(0);
        }
        for(vid_t i = 0; i < supernodes_num; i++){
            supernode_t& spn = supernodes[i];
            outfile << "id=" << graph_->GetId(spn.id) << " ids=" << spn.ids << " size=" << supernode_ids[spn.ids].size() << std::endl;
            for(auto u : supernode_ids[spn.ids]){
                outfile << graph_->GetId(u) << " ";
            }
            outfile << std::endl;
            // outfile << "inner_value:" << spn.inner_value.size() << std::endl;
            // for(auto edge : spn.inner_value){
            //     outfile << std::setprecision(10) << graph_->GetId(edge.first) << ": " << edge.second << std::endl;
            // }
            outfile << "inner_delta:" << spn.inner_delta.size() << std::endl;
            for(auto edge : spn.inner_delta){
                // outfile << graph_->GetId(edge.first) << ": " << edge.second << std::endl;
                vertex_t u;
                CHECK(graph_->Gid2Vertex(edge.second.parent_gid, u));
                outfile << graph_->GetId(edge.first) << ": " << edge.second.value << " id=" << graph_->GetId(u) << std::endl;
            }
            outfile << "bound_delta:" << spn.bound_delta.size() << std::endl;
            for(auto edge : spn.bound_delta){
                // outfile << graph_->GetId(edge.first) << ": " << edge.second << std::endl;
                vertex_t u;
                CHECK(graph_->Gid2Vertex(edge.second.parent_gid, u));
                outfile << graph_->GetId(edge.first) << ": " << edge.second.value << " id=" << graph_->GetId(u) << std::endl;
            }
            outfile << "fc:" << Fc[spn.id].size() << std::endl;
            for(auto f : Fc[spn.id]){
                outfile << graph_->GetId(f) << ",";
            }
            outfile << std::endl;
            outfile << std::endl;
        }
        LOG(INFO) << "finish write_spnodes..." << efile;
    }

    void write_spnodes_binary(const std::string &spnodefile){
        std::fstream file(spnodefile, std::ios::out | std::ios::binary);
        if(!file){
            LOG(INFO) << "Error opening file. " << spnodefile;
            exit(0);
        }
        // write supernode
        file.write(reinterpret_cast<char *>(&supernodes_num), sizeof(vid_t));
        for(vid_t i = 0; i < supernodes_num; i++){
            supernode_t& spn = supernodes[i];
            vid_t id = spn.id.GetValue();
            file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
            file.write(reinterpret_cast<char *>(&spn.ids), sizeof(vid_t));
        }
        // write Fc & supernode_source
        vid_t size;
        fc_t id;
        for(auto fc : Fc){
            id = fc;
            file.write(reinterpret_cast<char *>(&id), sizeof(fc_t));
        }
        vid_t source_num = supernode_source.size();
        file.write(reinterpret_cast<char *>(&source_num), sizeof(vid_t));
        for(auto ids : supernode_source){
            size = ids.size();
            file.write(reinterpret_cast<char *>(&size), sizeof(ids.size()));
            vid_t id;
            for(auto f : ids){
                id = f.GetValue();
                file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
            }
        }
        // write supernode_ids
        vid_t ids_num = supernode_ids.size();
        file.write(reinterpret_cast<char *>(&ids_num), sizeof(vid_t));
        for(auto ids : supernode_ids){
            size = ids.size();
            file.write(reinterpret_cast<char *>(&size), sizeof(ids.size()));
            vid_t id;
            for(auto f : ids){
                id = f.GetValue();
                file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
            }
        }
        file.close ();
    }

    bool read_spnodes_binary(const std::string &spnodefile){
        std::fstream file(spnodefile, std::ios::in | std::ios::binary);
        if(!file){
            return false;
        }
        // read supernode
        file.read(reinterpret_cast<char *>(&supernodes_num), sizeof(vid_t));
        for(vid_t i = 0; i < supernodes_num; i++){
            supernode_t& spn = supernodes[i];
            vid_t id;
            file.read(reinterpret_cast<char *>(&id), sizeof(vid_t));
            spn.id = vertex_t(id);
            file.read(reinterpret_cast<char *>(&spn.ids), sizeof(vid_t));
            Fc_map[spn.id] = i;
        }
        // read Fc & supernode_source
        Fc.Init(graph_->Vertices(), FC_default_value);
        fc_t id = 0;
        vid_t i = 0;
        for(auto v : graph_->Vertices()){
            file.read(reinterpret_cast<char *>(&id), sizeof(fc_t));
            Fc[vertex_t(i)] = id;
            i++;
        }
        vid_t source_num = 0;
        file.read(reinterpret_cast<char *>(&source_num), sizeof(vid_t));
        if(source_num > 0){
            supernode_source.resize(source_num);
        }
        for(auto& ids : supernode_source){
            vid_t size = 0;
            file.read(reinterpret_cast<char *>(&size), sizeof(ids.size()));
            if(size > 0){
                ids.resize(size);
            }
            vid_t id = 0;
            for(vid_t i = 0; i < size; i++){
                file.read(reinterpret_cast<char *>(&id), sizeof(vid_t));
                ids[i] = vertex_t(id);
            }
        }
        // read supernode_ids
        vid_t ids_num = 0;
        file.read(reinterpret_cast<char *>(&ids_num), sizeof(vid_t));
        if(ids_num > 0){
            supernode_ids.resize(ids_num);
        }
        vid_t cnt = 0;
        for(auto& ids : supernode_ids){
            vid_t size = 0;
            file.read(reinterpret_cast<char *>(&size), sizeof(ids.size()));
            if(size > 0){
                ids.resize(size);
            }
            vid_t id = 0;
            for(vid_t i = 0; i < size; i++){
                file.read(reinterpret_cast<char *>(&id), sizeof(vid_t));
                ids[i] = vertex_t(id);
                id2spids[ids[i]] = cnt;
            }
            cnt++;
        }
        file.close ();
        return true;
    }

    void compress(){
        std::string prefix = "";
        if(!FLAGS_serialization_cmp_prefix.empty()){
            /*
                filename: efile + vfile + worknum + 
            */
            std::string serialize_prefix = FLAGS_serialization_cmp_prefix;
            std::string digest = FLAGS_efile + FLAGS_vfile + std::to_string(comm_spec_.worker_num());
            digest += "_" + std::to_string(comm_spec_.worker_id())
                    + "_" + std::to_string(FLAGS_max_node_num)
                    + "_" + std::to_string(FLAGS_min_node_num)
                    + "_" + std::to_string(FLAGS_compress_concurrency)
                    + "_" + std::to_string(FLAGS_directed)
                    + "_" + std::to_string(FLAGS_compress_type)
                    + "_mirror_k" + std::to_string(FLAGS_mirror_k)
                    + "_cmpthreshold" + std::to_string(FLAGS_compress_threshold);

            std::replace(digest.begin(), digest.end(), '/', '_');
            prefix = serialize_prefix + "/" + digest;
            LOG(INFO) << prefix;
            std::string key = "_w";
            std::size_t found = prefix.rfind(key);
            if (found!=std::string::npos) {
                prefix.replace (found, key.length(), "");
            }
            LOG(INFO) << "prefix: " << prefix;
        }

        /* use metis */
        if(FLAGS_compress_type == 1){
            LOG(INFO) << "FLAGS_compress_type Error.";
            exit(0);
        }
        /* use scan++/ Louvain */
        else if(FLAGS_compress_type == 2){
            // use scan++ cluster method
            // compress_by_scanpp();
            compress_by_cluster(prefix);
        }
        else{
            LOG(INFO) << "No FLAGS_compress_type.";
        }
    }
    
    void statistic(){
        long inner_edges_num = 0;
        long bound_edges_num = 0;
        // long inner_value_num = 0;
        long supernodes_comtain_num = 0;
        long max_node_num = 0;
        long min_node_num = std::numeric_limits<vid_t>::max();
        long max_inner_edges_num = 0;
        long max_bound_edges_num = 0;
        for(long i = 0; i < supernodes_num; i++){
            inner_edges_num += supernodes[i].inner_delta.size();
            // inner_value_num += supernodes[i].inner_value.size();
            bound_edges_num += supernodes[i].bound_delta.size();
            max_node_num = std::max(max_node_num, (long)supernode_ids[supernodes[i].ids].size());
            min_node_num = std::min(min_node_num, (long)supernode_ids[supernodes[i].ids].size());
            max_inner_edges_num = std::max(max_inner_edges_num, (long)supernodes[i].inner_delta.size());
            max_bound_edges_num = std::max(max_bound_edges_num, (long)supernodes[i].bound_delta.size());
        }
        for(auto ids : supernode_ids){
            supernodes_comtain_num += ids.size();
        }
        long bound_node_num = 0;
        for(auto f : supernode_out_bound){
            if(f){
                bound_node_num++;
            }
        }
        long source_node_num = 0;
        for(auto vec : supernode_source){
            source_node_num += vec.size();
        }

        long global_bound_node_num = 0;
        long global_source_node_num = 0;
        long global_spn_com_num = 0;
        long global_spn_num = 0;
        long global_inner_edges_num = 0;
        // long global_inner_value_num = 0;
        long global_bound_edges_num = 0;
        long global_max_node_num = 0;
        long global_min_node_num = 0;
        long global_max_inner_edges_num = 0;
        long global_max_bound_edges_num = 0;
        long nodes_num = graph_->GetTotalVerticesNum();
        long edges_num = 0;
        long local_edges_num = graph_->GetEdgeNum();
        long local_ids_num = supernode_ids.size();
        long global_ids_num = 0;

        communicator_.template Sum(source_node_num, global_source_node_num);
        communicator_.template Sum(bound_node_num, global_bound_node_num);
        communicator_.template Sum(supernodes_comtain_num, global_spn_com_num);
        communicator_.template Sum((long)supernodes_num, global_spn_num);
        communicator_.template Sum(inner_edges_num, global_inner_edges_num);
        // communicator_.template Sum(inner_value_num, global_inner_value_num);
        communicator_.template Sum(bound_edges_num, global_bound_edges_num);
        communicator_.template Sum(local_edges_num, edges_num);
        communicator_.template Sum(local_ids_num, global_ids_num);
        communicator_.template Max(max_node_num, global_max_node_num);
        communicator_.template Max(min_node_num, global_min_node_num);
        communicator_.template Max(max_inner_edges_num, global_max_inner_edges_num);
        communicator_.template Max(max_bound_edges_num, global_max_bound_edges_num);

        edges_num /= 2; // in/out

        LOG(INFO) << "efile=" << FLAGS_efile;
        LOG(INFO) << "work" << comm_spec_.worker_id() << " supernodes_num=" << supernodes_num;
        if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "statistic...";
            LOG(INFO) << "#graph edges_num: " << edges_num;
            LOG(INFO) << "#nodes_num: " << nodes_num << std::endl;
            LOG(INFO) << "#global_spn_num: " << global_spn_num << std::endl; 
            LOG(INFO) << "#global_ids_num: " << global_ids_num << std::endl; 
            LOG(INFO) << "#global_spn_com_num: " << global_spn_com_num  << std::endl; 
            LOG(INFO) << "#global_spn_com_num_ave: " << (global_spn_com_num/(global_ids_num+1e-3))  << std::endl; 
            LOG(INFO) << "#global_bound_node_num: " << global_bound_node_num  << std::endl; 
            LOG(INFO) << "#global_source_node_num: " << global_source_node_num  << std::endl; 
            LOG(INFO) << "#global_inner_index_num: " << global_inner_edges_num  << std::endl; 
            LOG(INFO) << "#global_bound_index_num: " << global_bound_edges_num  << std::endl; 
            LOG(INFO) << "#global_spn_com_num/nodes_num: " << (global_spn_com_num*1.0/nodes_num)  << std::endl; 
            LOG(INFO) << "#supernodes_index/edges_num: " << ((global_inner_edges_num+global_bound_edges_num)*1.0/edges_num)  << std::endl; 
            // LOG(INFO) << "global_inner_value_num=" << global_inner_value_num << std::endl;
            LOG(INFO) << "#max_node_num: " << global_max_node_num << std::endl; 
            LOG(INFO) << "#min_node_num: " << global_min_node_num << std::endl; 
            LOG(INFO) << "#max_inner_index_num: " << global_max_inner_edges_num << std::endl; 
            LOG(INFO) << "#max_bound_index_num: " << global_max_bound_edges_num << std::endl; 
            LOG(INFO) << "#MAX_NODE_NUM: " << MAX_NODE_NUM  << std::endl; 
            LOG(INFO) << "#MIN_NODE_NUM: " << MIN_NODE_NUM  << std::endl; 
        }

        //debug
        {
            long long bound_edge_num = 0;
            long long inner_edge_num = 0;
            long long filter_num = 0;
            long long filter_error = 0;
            long long filter_save = 0;
            long long inner_edge_num_ = 0;
            long long best_save = 0;
            // check P set
            for(vid_t j = 0; j < supernode_ids.size(); j++){  // parallel compute
                std::vector<vertex_t> &node_set = this->supernode_ids[j];
                long long temp_ie_num = 0;
                vertex_t s;
                // inner edge, out bound edge
                for(auto v : node_set){
                    auto spids = this->id2spids[v];
                    const auto& oes = graph_->GetOutgoingAdjList(v);
                    for(auto& e : oes){
                        if(this->id2spids[e.neighbor] == spids){
                            inner_edge_num++;
                            temp_ie_num++;
                        }
                        else{
                            bound_edge_num++;
                        }
                    }
                    s = v;
                }
                // inner bound node
                int b_num = 0;
                for(auto v : node_set){
                    auto spids = this->id2spids[v];
                    const auto& oes = graph_->GetOutgoingAdjList(v);
                    for(auto& e : oes){
                        if(this->id2spids[e.neighbor] != spids){
                            b_num++;
                            break;
                        }
                    }
                }
                // source node
                int s_num_new = 0;
                for(auto v : node_set){
                    auto spids = this->id2spids[v];
                    const auto& oes = graph_->GetIncomingAdjList(v);
                    for(auto& e : oes){
                        if(this->id2spids[e.neighbor] != spids){
                            s_num_new++;
                            break;
                        }
                    }
                }
                vid_t src_id = Fc[s] < 0 ? (-Fc[s]-1) : Fc[s];
                int s_num = supernode_source[src_id].size();
                if(s_num * b_num > temp_ie_num && s_num > 1){
                    filter_num++;
                    if (filter_num < 5) {
                        LOG(INFO) << "id=" << j << " s_num=" << s_num << " b_num=" << b_num << " temp_ie_num=" << temp_ie_num;
                    }
                    // CHECK_EQ(s_num, s_num_new);
                }
            }

            LOG(INFO) << " filter_num=" << filter_num << " filter_error=" << filter_error << " filter_save=" << filter_save;
            LOG(INFO) << "#inner_edge_num/global_bound_edges_num: " << (inner_edge_num*1.0/global_bound_edges_num) << std::endl;
            LOG(INFO) << "#save_edge_rate: " << ((inner_edge_num*1.0-global_bound_edges_num)/edges_num) << std::endl;
            LOG(INFO) << "#inner_edge_num: " << inner_edge_num << std::endl; 
            // LOG(INFO) << "#bound_edge_num: " << bound_edge_num << std::endl; 
            LOG(INFO) << "#inner_edge_num/edges_num: " << (inner_edge_num*1.0/edges_num) << std::endl; 
        }

        // Statistics superpoint size distribution data
        int step = 500;
        int max_num = 1e6;
        std::vector<int> s(max_num/step, 0);
        for (int i=0 ; i < supernode_ids.size(); i++) {
            s[supernode_ids[i].size()/step]++;
        }
        for(int i = 0; i < max_num/step; i++){
            if(s[i] > 0){
                LOG(INFO) << "   statistic: " << (i*step) << "-" << (i*step+step) << ":" << s[i];
            }
        }
    }

    float check_score(vertex_t d, std::set<vertex_t>& S, std::set<vertex_t>& P, std::set<vertex_t>& B, float ring_weight, std::vector<short int>& is_s_vec, std::vector<short int>& is_b_vec, std::vector<std::unordered_set<vertex_t>>& temp_S_vec, std::vector<std::unordered_set<vertex_t>>& temp_B_vec, std::vector<vid_t>& temp_inner_vec, int i){
        // if d add to P:
        bool is_s = false;
        bool is_b = false;
        vid_t temp_inner = temp_inner_vec[i];
        // update S/P/O
        // std::unordered_set<vertex_t> temp_B; // need to delete
        // std::unordered_set<vertex_t> temp_S; // need to delete
        std::unordered_set<vertex_t>& temp_B = temp_B_vec[i];
        std::unordered_set<vertex_t>& temp_S = temp_S_vec[i];
        temp_B.clear();
        temp_S.clear();
        vid_t now_s_size = S.size();
        vid_t now_b_size = B.size();
        for(auto& oe : graph_->GetOutgoingAdjList(d)){
            // update S
            if(S.find(oe.neighbor) != S.end()){
                bool flag = true;
                for(auto& ie : graph_->GetIncomingAdjList(oe.neighbor)){
                    if(P.find(ie.neighbor) == P.end() && ie.neighbor != d){
                        flag = false;
                        break;
                    }
                }
                if(flag){
                    // temp_S.erase(v.neighbor);
                    temp_S.insert(oe.neighbor);
                    // now_s_size--;
                }
            }
            // update temp_inner
            if(P.find(oe.neighbor) == P.end() && oe.neighbor != d){ // self-edge
                is_b = true;
            }
            else{
                // temp_inner++;
                temp_inner += ring_weight;
            }
        }
        for(auto& ie : graph_->GetIncomingAdjList(d)){
            // update B
            if(B.find(ie.neighbor) != B.end()){
                bool flag = true;
                for(auto& oe : graph_->GetOutgoingAdjList(ie.neighbor)){
                    if(P.find(oe.neighbor) == P.end() && oe.neighbor != d){
                        flag = false;
                        break;
                    }
                }
                if(flag){
                    temp_B.insert(ie.neighbor);
                    // now_b_size--;
                }
            }
            // update temp_inner
            // if(P.find(ie.neighbor) != P.end() || ie.neighbor == d){ // self-edge
            if(P.find(ie.neighbor) != P.end()){ // self-edge just only count once, it has been calculated in the out side.
                temp_inner++;
            }
            else{
                is_s = true;
            }
        }
        now_s_size -= temp_S.size();
        now_b_size -= temp_B.size();
        if(is_s){
            // temp_S.insert(d);
            now_s_size++;
        }
        if(is_b){
            // B.insert(d);
            now_b_size++;
        }

        is_s_vec[i] = is_s;
        is_b_vec[i] = is_b;
        temp_inner_vec[i] = temp_inner;
        // LOG(INFO) << "i=" << i << " is_s=" << is_s << " is_b=" << is_b;  
        // return temp_inner * 1.0 / (now_s_size * now_b_size + 1e-3);
        // return (temp_inner * 1.0 - (now_s_size * now_b_size)) / temp_inner; // Reference: Graph Summarization with Bounded Error
        return temp_inner * 1.0 - (now_s_size * now_b_size);
    }

    void inc_compress(std::vector<std::pair<vid_t, vid_t>>& deleted_edges, std::vector<std::pair<vid_t, vid_t>>& added_edges){
        fid_t fid = graph_->fid();
        auto vm_ptr = graph_->vm_ptr();
        inccalculate_spnode_ids.clear();
        recalculate_spnode_ids.clear();
        vid_t add_num = 0;
        vid_t del_num = 0;
        LOG(INFO) << "spnode_num=" << supernodes_num;
        LOG(INFO) << "deal deleted_edges...";

        for(auto& pair : deleted_edges) {
            auto u_gid = pair.first;
            auto v_gid = pair.second;
            fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                  v_fid = vm_ptr->GetFidFromGid(v_gid);
            // u -> v
            // LOG(INFO) << u_gid << "->" << v_gid;
            vertex_t u;
            CHECK(graph_->Gid2Vertex(u_gid, u));
            if(u_fid == fid && Fc[u] != FC_default_value){
                // LOG(INFO) << graph_->GetId(u);
                vid_t src_id = Fc[u] < 0 ? (-Fc[u]-1) : Fc[u];
                for(auto source : supernode_source[src_id]){
                    inccalculate_spnode_ids.insert(source.GetValue());
                }
            }
            vertex_t v;
            CHECK(graph_->Gid2Vertex(v_gid, v));
            if(v_fid == fid && Fc[v] != FC_default_value){
                // vid_t del_id = Fc_map[Fc[v][0]];
                vid_t src_id = Fc[v] < 0 ? (-Fc[v]-1) : Fc[v];
                std::vector<vertex_t>& src = supernode_source[src_id];
                vid_t del_id = Fc_map[src[0]];
                supernode_t& spnode = supernodes[del_id];
                const vid_t ids_id = this->id2spids[spnode.id];
                if(ids_id != this->id2spids[u] && src.size() > 1){
                    CHECK(Fc[v] >= 0);
                    const auto& ies = graph_->GetIncomingAdjList(v);
                    bool hava_out_inadj = false;
                    // for(auto& e : ies){
                    //     auto& nb = e.neighbor;
                    //     if(nb != u && ids_id != this->id2spids[nb]){
                    //         hava_out_inadj = true;
                    //         break;
                    //     }
                    // }
                    /*-----parallel-----*/
                        auto out_degree = ies.Size();
                        auto it = ies.begin();
                        granular_for(j, 0, out_degree, (out_degree > 1024), {
                            auto& e = *(it + j);
                            auto& nb = e.neighbor;
                            if(nb != u && ids_id != this->id2spids[nb]){
                                hava_out_inadj = true;
                                // break;
                            }
                        })
                    if(hava_out_inadj == false){
                        delete_supernode(v);
                        del_num++;
                    }
                }
            }
        }
        LOG(INFO) << "deal added_edges...";
        for(auto& pair : added_edges){
            auto u_gid = pair.first;
            auto v_gid = pair.second;
            fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                  v_fid = vm_ptr->GetFidFromGid(v_gid);
            // u -> v
            // LOG(INFO) << u_gid << "->" << v_gid;
            vertex_t u;
            CHECK(graph_->Gid2Vertex(u_gid, u));
            if(u_fid == fid && Fc[u] != FC_default_value){
                // LOG(INFO) << graph_->GetId(u);
                // for(auto source: Fc[u]){
                vid_t src_id = Fc[u] < 0 ? (-Fc[u]-1) : Fc[u];
                for(auto source : supernode_source[src_id]){
                    inccalculate_spnode_ids.insert(source.GetValue());
                }
            }
            vertex_t v;
            CHECK(graph_->Gid2Vertex(v_gid, v));
            if(v_fid == fid && Fc[v] != FC_default_value){
                // supernode_t& spnode = supernodes[Fc_map[Fc[v][0]]];
                vid_t src_id = Fc[v] < 0 ? (-Fc[v]-1) : Fc[v];
                std::vector<vertex_t>& src = supernode_source[src_id];
                supernode_t& spnode = supernodes[Fc_map[src[0]]];
                auto& spids = supernode_ids[spnode.ids];
                const vid_t ids_id = this->id2spids[spnode.id];
                
                if(Fc[v] < 0 && ids_id != this->id2spids[u]){ // not a source, build a new spnode
                    // for(auto u : spids){
                    //     Fc[u] = src_id;
                    // }
                    // std::swap(Fc[v][0], Fc[v][int(Fc[v].size())-1]);
                    Fc[v] = src_id;
                    this->supernode_source[src_id].emplace_back(v);
                    // build a new spnode idnex
                    vid_t supernode_id = supernodes_num;
                    Fc_map[v] = supernode_id;
                    supernodes[supernode_id].id = v;
                    // supernodes[supernode_id].ids.insert(supernodes[supernode_id].ids.begin(), spnode.ids.begin(), spnode.ids.end());
                    supernodes[supernode_id].ids = spnode.ids;
                    supernodes_num++;

                    recalculate_spnode_ids.insert(v.GetValue());
                    add_num++;
                }
            }
        }
    }

    void parallel_inc_compress(std::vector<std::pair<vid_t, vid_t>>& deleted_edges, std::vector<std::pair<vid_t, vid_t>>& added_edges){
        fid_t fid = graph_->fid();
        auto vm_ptr = graph_->vm_ptr();
        inccalculate_spnode_ids.clear();
        recalculate_spnode_ids.clear();
        vid_t add_num = 0;
        vid_t del_num = 0;
        {
            /* parallel */
            int thread_num = FLAGS_app_concurrency > 0 ? FLAGS_app_concurrency : 4;
            std::vector<std::unordered_set<vid_t>> inc_temp;
            std::vector<std::unordered_set<vid_t>> del_temp;
            inc_temp.resize(thread_num);
            del_temp.resize(thread_num);
            this->ForEachIndex(deleted_edges.size(), [this, &vm_ptr, &inc_temp, &del_temp, &deleted_edges](int tid, vid_t begin, vid_t end) {
                fid_t fid = graph_->fid();
                // LOG(INFO) << "build index, tid=" << tid << " range: "<< begin << "->" << end;
                double start = GetCurrentTime();
                for(vid_t i = begin; i < end; i++){
                    auto& pair = deleted_edges[i];
                    auto u_gid = pair.first;
                    auto v_gid = pair.second;
                    fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                        v_fid = vm_ptr->GetFidFromGid(v_gid);
                    // u -> v
                    // LOG(INFO) << u_gid << "->" << v_gid;
                    vertex_t u;
                    CHECK(graph_->Gid2Vertex(u_gid, u));
                    if(u_fid == fid && Fc[u] != FC_default_value){
                        vid_t src_id = Fc[u] < 0 ? (-Fc[u]-1) : Fc[u];
                        for(auto source : supernode_source[src_id]){
                            inc_temp[tid].insert(source.GetValue());
                        }
                    }
                    vertex_t v;
                    CHECK(graph_->Gid2Vertex(v_gid, v));
                    if(v_fid == fid && Fc[v] != FC_default_value){
                        vid_t src_id = Fc[v] < 0 ? (-Fc[v]-1) : Fc[v];
                        std::vector<vertex_t>& src = supernode_source[src_id];
                        vid_t del_id = Fc_map[src[0]];
                        supernode_t& spnode = supernodes[del_id];
                        const vid_t ids_id = this->id2spids[spnode.id];
                        if(Fc[v] >= 0 && ids_id != this->id2spids[u] && src.size() > 1){
                            CHECK(Fc[v] >= 0);
                            // CHECK(Fc[v][0] == v);
                            const auto& ies = graph_->GetIncomingAdjList(v);
                            bool hava_out_inadj = false;
                            /*-----parallel-----*/
                                auto out_degree = ies.Size();
                                auto it = ies.begin();
                                granular_for(j, 0, out_degree, (out_degree > 1024), {
                                    auto& e = *(it + j);
                                    auto& nb = e.neighbor;
                                    if(nb != u && ids_id != this->id2spids[nb]){
                                        hava_out_inadj = true;
                                        // break;
                                    }
                                })
                            if(hava_out_inadj == false){
                                del_temp[tid].insert(v.GetValue());
                            }
                        }
                    }
                }
                }, thread_num
            );
            double start = GetCurrentTime();
            std::vector<vid_t> del_ids;
            for(int i = 0; i < thread_num; i++){ 
                inccalculate_spnode_ids.insert(inc_temp[i].begin(), inc_temp[i].end());
                del_ids.insert(del_ids.end(), del_temp[i].begin(), del_temp[i].end());
            }
            vid_t count_id = 0;
            vid_t del_ids_num = del_ids.size();
            start = GetCurrentTime();
            this->ForEach(del_ids_num, [this, &count_id, &del_ids, &del_ids_num](int tid) {
                int i = 0, cnt = 0, step = 1;
                while(i < del_ids_num){
                    i = __sync_fetch_and_add(&count_id, 1);
                    if(i < del_ids_num){
                        vertex_t u(del_ids[i]);
                        delete_supernode(u);
                    }
                }
                }, thread_num
            );
        }

        {
            double start = GetCurrentTime();
            for(auto& pair : added_edges){
                auto u_gid = pair.first;
                auto v_gid = pair.second;
                fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                    v_fid = vm_ptr->GetFidFromGid(v_gid);
                vertex_t u;
                CHECK(graph_->Gid2Vertex(u_gid, u));
                if(u_fid == fid && Fc[u] != FC_default_value){
                    vid_t src_id = Fc[u] < 0 ? (-Fc[u]-1) : Fc[u];
                    // CHECK(src_id < supernode_source.size());
                    for(auto source : supernode_source[src_id]){
                        inccalculate_spnode_ids.insert(source.GetValue());
                    }
                }
                vertex_t v;
                CHECK(graph_->Gid2Vertex(v_gid, v));
                if(v_fid == fid && Fc[v] != FC_default_value){
                    vid_t src_id = Fc[v] < 0 ? (-Fc[v]-1) : Fc[v];
                    // CHECK(src_id < supernode_source.size());
                    std::vector<vertex_t>& src = supernode_source[src_id];
                    // CHECK(src.size() > 0);
                    // CHECK(Fc_map[src[0]] < supernodes_num);
                    supernode_t& spnode = supernodes[Fc_map[src[0]]];
                    auto& spids = supernode_ids[spnode.ids];
                    const vid_t ids_id = this->id2spids[spnode.id];
                    if(Fc[v] < 0 && ids_id != this->id2spids[u]){ // not a source, build a new spnode
                        Fc[v] = src_id;
                        this->supernode_source[src_id].emplace_back(v);
                        // build a new spnode idnex
                        vid_t supernode_id = supernodes_num;
                        Fc_map[v] = supernode_id;
                        supernodes[supernode_id].id = v;
                        supernodes[supernode_id].ids = spnode.ids;
                        supernodes_num++;

                        recalculate_spnode_ids.insert(v.GetValue());
                    }
                }
            }
        }
        LOG(INFO) << "#spn_update_rate: " << ((inccalculate_spnode_ids.size()+recalculate_spnode_ids.size())*1.0/supernodes_num);
    }

    template<typename T = vertex_t>
    bool remove_array(std::vector<T> &array, T v){
        typename std::vector<T>::iterator it = std::find(array.begin(), array.end(), v);
        if(it == array.end()){
            return false;
        }
        std::swap(*it, array.back());
        array.pop_back();
        return true;
    }

    void delete_supernode(const vertex_t source){
        vid_t del_id = Fc_map[source]; // Get the index of spnode in the array
        supernode_t &spnode_v = supernodes[del_id];
        std::unique_lock<std::mutex> lk(supernode_ids_mux_);
        vid_t src_id = Fc[source];
        if(src_id < 0 || supernode_source[src_id].size() <= 1){
            return;
        }
        Fc_map[spnode_v.id] = ID_default_value;
        spnode_v.clear();
        CHECK(src_id >= 0);
        Fc[source] = -(src_id+1);
        CHECK(remove_array(supernode_source[src_id], source));
        supernode_t &spnode_end = supernodes[supernodes_num-1];
        // updata Fc_map
        Fc_map[spnode_end.id] = del_id;
        // clear supernode 
        spnode_v.swap(spnode_end);
        supernodes_num--;
    }

    void delete_cluster(const vid_t cid) {
      // clear supnode of this cluster
      for (auto source : supernode_source[cid]) {
        vid_t del_spid = Fc_map[source]; // Get the index of spnode in the array
        delete_supernode(del_spid);
      }
      for (auto source : cluster_in_mirror_ids[cid]) {
        vid_t del_spid = Fc_map[source]; // Get the index of spnode in the array
        delete_supernode(del_spid);
      }
      // clear all info of cluster
      auto& all_nodes = cluster_ids[cid]; // include mirror
      for (auto v : all_nodes) {
        if (v.GetValue() < this->old_node_num) { // real node
          // master vertex
          Fc[v] = FC_default_value;
          Fc_map[v] = ID_default_value;
          id2spids[v] = ID_default_value;
        } else {
          // mirror
          Fc_map[v] = ID_default_value;
          id2spids[v] = ID_default_value;
        }
        supernode_out_bound[v.GetValue()] = false;
      }
      // clear mirror info in master cluster
      for (auto mid_v : cluster_in_mirror_ids[cid]) {
        auto master_vid = mirrorid2vid[mid_v];
        CHECK(remove_array<vid_t>(vid2in_mirror_mids[master_vid.GetValue()], 
                                    mid_v.GetValue()));
        CHECK(remove_array<vid_t>(
                    vid2in_mirror_cluster_ids[master_vid.GetValue()], cid));
      }
      for (auto mid_v : cluster_out_mirror_ids[cid]) {
        auto master_vid = mirrorid2vid[mid_v];
        CHECK(remove_array<vid_t>(vid2out_mirror_mids[master_vid.GetValue()], 
                                    mid_v.GetValue()));
      }

      auto& nodes_no_mirror = supernode_ids[cid];
      cluster_ids[cid].clear();
      supernode_source[cid].clear();
      cluster_out_mirror_ids[cid].clear();
      cluster_in_mirror_ids[cid].clear();
      supernode_out_mirror[cid].clear();
      supernode_in_mirror[cid].clear();
      for (auto v : nodes_no_mirror) {
        bool used = false;
        for (auto t : vid2in_mirror_mids[v.GetValue()]) {
          auto mid = vertex_t(t);
          vid_t cid = id2spids[mid];
          Fc[v] = -(cid+1); 
          id2spids[v] = cid;
          used = true;
          for (auto &u : supernode_source[cid]) {
            if (u == mid) {
              u = v;
              Fc_map[v] = Fc_map[u];
              Fc[v] = cid;
              break;
            }
          }
          for (auto &u : cluster_ids[cid]) {
            if (u == mid) {
              u = v;
              break;
            }
          }
          supernode_ids[cid].emplace_back(v);
          cluster_ids[cid].emplace_back(v);
          CHECK(remove_array(cluster_in_mirror_ids[cid], mid));
          supernode_in_mirror[cid].erase(v);
          CHECK(remove_array<vid_t>(vid2in_mirror_mids[v.GetValue()], 
                                    mid.GetValue()));
          CHECK(remove_array<vid_t>(vid2in_mirror_cluster_ids[v.GetValue()], 
                                    cid));
          for (auto out_mirror : cluster_out_mirror_ids[cid]) {
            if (mirrorid2vid[out_mirror].GetValue() == v.GetValue()) {
              CHECK(remove_array(cluster_out_mirror_ids[cid], out_mirror));
              CHECK(supernode_out_mirror[cid].erase(v));
              CHECK(remove_array<vid_t>(vid2out_mirror_mids[v.GetValue()], 
                                        out_mirror.GetValue()));
              break;
            }
          }
          break; 
        }
        if (used == true) {
          continue;
        }
        for (auto t : vid2out_mirror_mids[v.GetValue()]) {
          auto mid = vertex_t(t);
          vid_t cid = id2spids[mid];
          Fc[v] = -(cid+1); 
          id2spids[v] = cid;
          used = true;
          for (auto &u : supernode_source[cid]) {
            if (u == mid) {
              u = v;
              Fc_map[v] = Fc_map[u];
              Fc[v] = cid;
              break;
            }
          }
          for (auto &u : cluster_ids[cid]) {
            if (u == mid) {
              u = v;
              break;
            }
          }
          supernode_ids[cid].emplace_back(v);
          cluster_ids[cid].emplace_back(v);
          CHECK(remove_array(cluster_out_mirror_ids[cid], mid));
          supernode_out_mirror[cid].erase(v);
          CHECK(remove_array<vid_t>(vid2out_mirror_mids[v.GetValue()], 
                                    mid.GetValue()));
          for (auto in_mirror : cluster_in_mirror_ids[cid]) {
            if (mirrorid2vid[in_mirror].GetValue() == v.GetValue()) {
              CHECK(remove_array(cluster_in_mirror_ids[cid], in_mirror));
              CHECK(supernode_in_mirror[cid].erase(v));
              CHECK(remove_array<vid_t>(vid2in_mirror_mids[v.GetValue()], 
                                        in_mirror.GetValue()));
              CHECK(remove_array<vid_t>(vid2in_mirror_cluster_ids[v.GetValue()], 
                                        cid));
              break;
            }
          }
          break; 
        }
      }
      supernode_ids[cid].clear();
    }

    void delete_supernode(const vid_t del_spid){
        supernode_t &spnode_v = supernodes[del_spid];
        vertex_t source = spnode_v.id;
        vid_t spids_id = spnode_v.ids;
        // delete spnode
        std::unique_lock<std::mutex> lk(supernode_ids_mux_);
        if(spids_id < 0){
            LOG(INFO) << "error.";
            exit(0);
        }
        CHECK(spids_id >= 0);
        supernode_t &spnode_end = supernodes[supernodes_num-1];
        // updata Fc_map
        Fc_map[spnode_end.id] = del_spid;
        vid_t vid = spnode_end.id.GetValue();
        if (vid >= this->old_node_num) { // mirror to master
          vid = this->mirrorid2vid[spnode_end.id].GetValue();
        }
        this->shortcuts[vid][spnode_end.ids] = del_spid;
        // clear supernode 
        spnode_v.swap(spnode_end);
        supernodes_num--;
    }


    // big to small, sort
    static bool cmp_pair_b2s_cluster(std::pair<vid_t, vid_t> a, std::pair<vid_t, vid_t> b){
        if(a.second != b.second)
            return a.second > b.second;
        else return a.first < b.first;
    }

    // big to small, sort
    static bool cmp_pair_b2s(std::pair<vertex_t, vid_t> a, std::pair<vertex_t, vid_t> b){
        if(a.second != b.second)
            return a.second > b.second;
        else return a.first.GetValue() < b.first.GetValue();
    }

    // big to small, sort
    static bool cmp_pair_s2b_sort(std::pair<vertex_t, vid_t> a, std::pair<vertex_t, vid_t> b){
        if(a.second != b.second)
            return a.second < b.second;
        else return a.first.GetValue() < b.first.GetValue();
    }

    // small to big, priority_queue
    struct cmp_pair_s2b{
        bool operator()(std::pair<vertex_t, vid_t> a, std::pair<vertex_t, vid_t> b){
            return a.second < b.second;
        }
    };

    void compress_by_scanpp(){
        // std::string path = "/mnt/data/nfs/yusong/code/ppSCAN/SCANVariants/scan_plus2/result_uk-2002_base.txt.c";
        std::string path = FLAGS_efile + ".c_" 
            + std::to_string(FLAGS_max_node_num); // road_usa.e -> road_usa.e.c.1000
        std::vector<std::string> keys{"_w.", "_ud.", "_w1.", ".random."};
        while (true) {
            bool changed = false;
            for(int i = 0; i < keys.size(); i++){
                std::string key = keys[i];
                std::size_t found = path.rfind(key);
                if (found!=std::string::npos) {
                    path.replace (found, key.length(), ".");
                    changed = true;
                }
            }
            if (changed == false) {
                break;
            }
        }

        LOG(INFO) << "load cluster result file... path=" << path;

        std::ifstream inFile(path);
        if(!inFile){
            LOG(INFO) << "open file failed. " << path;
            exit(0);
        }
        size_t size;
        vid_t v_oid, v_gid;
        vid_t cluster_num = 0;
        vertex_t u;
        auto vm_ptr = graph_->vm_ptr();
        fid_t fid = this->graph_->fid();
        while(inFile >> size){
            std::set<vertex_t> P;
            for(int i = 0; i < size; i++){
                inFile >> v_oid;
                CHECK(vm_ptr->GetGid(v_oid, v_gid));
                fid_t v_fid = vm_ptr->GetFidFromGid(v_gid);
                if (v_fid == fid) {
                    vertex_t u;
                    CHECK(this->graph_->Gid2Vertex(v_gid, u));
                    P.insert(u);
                }
            }
            cluster_num++;
            if(P.size() >= MIN_NODE_NUM){
                build_supernode_by_P(P);
            }
            if(cluster_num % 100000 == 0){
                LOG(INFO) << "cluster_num=" << cluster_num << " spnodes_num=" << supernodes_num << std::endl;
            }
        }
    }

    void build_supernode_by_P(std::set<vertex_t>& P){
        std::set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
        // std::set<vertex_t> P; // inner node
        std::set<vertex_t> B; // belong to P, bound vertices
        const float obj_score = 1;
        const float ring_weight = 1;

        long long temp_ie_num = 0;
        vertex_t s;
        // inner edge, out bound edge
        for(auto v : P){
            const auto& oes = graph_->GetOutgoingAdjList(v);
            for(auto& e : oes){
                if(P.find(e.neighbor) != P.end()){
                    temp_ie_num++;
                }
            }
        }
        // inner bound node
        for(auto v : P){
            const auto& oes = graph_->GetOutgoingAdjList(v);
            for(auto& e : oes){
                if(P.find(e.neighbor) == P.end()){
                    B.insert(v);
                    break;
                }
            }
        }
        // source node
        for(auto v : P){
            const auto& oes = graph_->GetIncomingAdjList(v);
            for(auto& e : oes){
                if(P.find(e.neighbor) == P.end()){
                    S.insert(v);
                    break;
                }
            }
        }
        int b_num = B.size();
        int s_num = S.size();
        float score = temp_ie_num * 1.0 / (s_num * b_num + 1e-3); // must >= 0
        float obj = FLAGS_compress_threshold; // 1
        if(score >= obj){
        // if(1){ // not fiter
            if(S.size() == 0){
                S.insert(*(P.begin()));
            }
            int ids_id = -1;
            {
                std::unique_lock<std::mutex> lk(supernode_ids_mux_);
                supernode_ids.emplace_back(P.begin(), P.end());
                ids_id = int(supernode_ids.size()) - 1; // root_id
                // supernode_bound_ids.emplace_back(O.begin(), O.end());
                supernode_source.emplace_back(S.begin(), S.end());
            }
            // CHECK(ids_id >= 0);
            for(auto u : P){
                Fc[u] = -(ids_id+1);
                id2spids[u] = ids_id;
            }
            for(auto src : S){
                Fc[src] = ids_id;
                /* build a supernode */
                // vid_t supernode_id = supernodes_num;
                // supernodes_num++;
                vid_t supernode_id = __sync_fetch_and_add(&supernodes_num, 1);
                Fc_map[src] = supernode_id;
                supernodes[supernode_id].id = src;
                supernodes[supernode_id].ids = ids_id; // root_id
            }
        }
    }

    bool de_serialize_cluster(const std::string prefix, vid_t& init_mirror_num) {
      std::fstream file(prefix, std::ios::in | std::ios::binary);
      if(!file){
        LOG(INFO) << "Can't opening file, refind cluster... " << prefix;
        return false;
      }
      LOG(INFO) << "Deserializing cluster to " << prefix;
      // read cluster
      // vid_t init_mirror_num = 0;
      file.read(reinterpret_cast<char *>(&init_mirror_num), sizeof(vid_t));
      vid_t supernode_ids_num = 0;
      file.read(reinterpret_cast<char *>(&supernode_ids_num), sizeof(vid_t));

      LOG(INFO) << " init_mirror_num=" << init_mirror_num;
      LOG(INFO) << " supernode_ids_num=" << supernode_ids_num;

      // init 
      this->supernode_ids.resize(supernode_ids_num);
      this->cluster_ids.resize(supernode_ids_num);
      this->supernode_source.resize(supernode_ids_num);
      this->supernode_in_mirror.resize(supernode_ids_num);
      this->supernode_out_mirror.resize(supernode_ids_num);

      for (vid_t i = 0; i < supernode_ids_num; i++) {
        std::vector<vertex_t>& sp_ids = this->supernode_ids[i];
        vid_t sp_ids_num = 0;
        file.read(reinterpret_cast<char *>(&sp_ids_num), sizeof(vid_t));
        sp_ids.resize(sp_ids_num);
        for (vid_t i = 0; i < sp_ids_num; i++) {
          vid_t vid = 0;
          file.read(reinterpret_cast<char *>(&vid), sizeof(vid_t));
          sp_ids[i] = vertex_t(vid);
        }

        std::vector<vertex_t>& cs_ids = this->cluster_ids[i];
        vid_t cs_ids_num = 0;
        file.read(reinterpret_cast<char *>(&cs_ids_num), sizeof(vid_t));
        cs_ids.resize(cs_ids_num);
        for (vid_t i = 0; i < cs_ids_num; i++) {
          vid_t vid = 0;
          file.read(reinterpret_cast<char *>(&vid), sizeof(vid_t));
          cs_ids[i] = vertex_t(vid);
        }

        std::vector<vertex_t>& sp_source = this->supernode_source[i];
        vid_t sp_source_num = 0;
        file.read(reinterpret_cast<char *>(&sp_source_num), sizeof(vid_t));
        sp_source.resize(sp_source_num);
        for (vid_t i = 0; i < sp_source_num; i++) {
          vid_t vid = 0;
          file.read(reinterpret_cast<char *>(&vid), sizeof(vid_t));
          sp_source[i] = vertex_t(vid);
        }

        std::unordered_set<vertex_t>& sp_in_mirror = this->supernode_in_mirror[i];
        vid_t sp_in_mirror_num = 0;
        file.read(reinterpret_cast<char *>(&sp_in_mirror_num), sizeof(vid_t));
        for (vid_t i = 0; i < sp_in_mirror_num; i++) {
          vid_t vid = 0;
          file.read(reinterpret_cast<char *>(&vid), sizeof(vid_t));
          sp_in_mirror.insert(vertex_t(vid));
        }

        std::unordered_set<vertex_t>& sp_out_mirror = this->supernode_out_mirror[i];
        vid_t sp_out_mirror_num = 0;
        file.read(reinterpret_cast<char *>(&sp_out_mirror_num), sizeof(vid_t));
        for (vid_t i = 0; i < sp_out_mirror_num; i++) {
          vid_t vid = 0;
          file.read(reinterpret_cast<char *>(&vid), sizeof(vid_t));
          sp_out_mirror.insert(vertex_t(vid));
        }
      }

      fc_t id = 0;
      for(auto v : graph_->Vertices()){
        file.read(reinterpret_cast<char *>(&id), sizeof(fc_t));
        Fc[v] = id;
      }

      vid_t spid;
      for(auto v : graph_->Vertices()){
        file.read(reinterpret_cast<char *>(&spid), sizeof(vid_t));
        id2spids[v] = spid;
      }
      int eof = 0;
      file.read(reinterpret_cast<char *>(&eof), sizeof(int));
      file.close();
      CHECK_EQ(eof, -1);
      return true;
    }

    void serialize_cluster(const std::string prefix, vid_t init_mirror_num) {
      std::fstream file(prefix, std::ios::out | std::ios::binary);
      if(!file){
        LOG(INFO) << "Error opening file. " << prefix;
        exit(0);
      }
      LOG(INFO) << "Serializing cluster to " << prefix;
      // write cluster
      file.write(reinterpret_cast<char *>(&init_mirror_num), sizeof(vid_t));
      vid_t supernode_ids_num = supernode_ids.size();
      file.write(reinterpret_cast<char *>(&supernode_ids_num), sizeof(vid_t));

      LOG(INFO) << " init_mirror_num=" << init_mirror_num;
      LOG(INFO) << " supernode_ids_num=" << supernode_ids_num;

      for (vid_t i = 0; i < supernode_ids_num; i++) {
        std::vector<vertex_t>& sp_ids = this->supernode_ids[i];
        vid_t sp_ids_num = sp_ids.size();
        file.write(reinterpret_cast<char *>(&sp_ids_num), sizeof(vid_t));
        for (auto& v : sp_ids) {
          vid_t id = v.GetValue();
          file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
        }

        std::vector<vertex_t>& cs_ids = this->cluster_ids[i];
        vid_t cs_ids_num = cs_ids.size();
        file.write(reinterpret_cast<char *>(&cs_ids_num), sizeof(vid_t));
        for (auto& v : cs_ids) {
          vid_t id = v.GetValue();
          file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
        }

        std::vector<vertex_t>& sp_source = supernode_source[i];
        vid_t sp_source_num = sp_source.size();
        file.write(reinterpret_cast<char *>(&sp_source_num), sizeof(vid_t));
        for (auto& v : sp_source) {
          vid_t id = v.GetValue();
          file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
        }

        std::unordered_set<vertex_t>& sp_in_mirror = supernode_in_mirror[i];
        vid_t sp_in_mirror_num = sp_in_mirror.size();
        file.write(reinterpret_cast<char *>(&sp_in_mirror_num), sizeof(vid_t));
        for (auto& v : sp_in_mirror) {
          vid_t id = v.GetValue();
          file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
        }

        std::unordered_set<vertex_t>& sp_out_mirror = supernode_out_mirror[i];
        vid_t sp_out_mirror_num = sp_out_mirror.size();
        file.write(reinterpret_cast<char *>(&sp_out_mirror_num), sizeof(vid_t));
        for (auto& v : sp_out_mirror) {
          vid_t id = v.GetValue();
          file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
        }
      }
      fc_t id;
      for (auto fc : Fc) {
          id = fc;
          file.write(reinterpret_cast<char *>(&id), sizeof(fc_t));
      }

      vid_t spid;
      for (auto spid : id2spids) {
          file.write(reinterpret_cast<char *>(&spid), sizeof(vid_t));
      }
      int eof = -1;
      file.write(reinterpret_cast<char *>(&eof), sizeof(int));
      file.close();
    }

    void compress_by_cluster(const std::string prefix){
        LOG(INFO) << "compress_by_cluster...";

        if (prefix != "") {
          vid_t init_mirror_num = 0;
          bool find = de_serialize_cluster(prefix, init_mirror_num);
          if (find == true) {
            final_build_supernode(init_mirror_num);
            return ;
          }
        }

        // std::string path = "/mnt/data/nfs/yusong/code/ppSCAN/SCANVariants/scan_plus2/result_uk-2002_base.txt.c";
        // get cluster file name
        std::string path = FLAGS_efile + ".c_" 
            + std::to_string(FLAGS_max_node_num); // road_usa.e -> road_usa.e.c.1000
        std::vector<std::string> keys{"_w.", "_ud.", "_w1.", ".random."};
        while (true) {
            bool changed = false;
            for(int i = 0; i < keys.size(); i++){
                std::string key = keys[i];
                std::size_t found = path.rfind(key);
                if (found!=std::string::npos) {
                    path.replace (found, key.length(), ".");
                    changed = true;
                }
            }
            if (changed == false) {
                break;
            }
        }

        LOG(INFO) << "load cluster result file... path=" << path;

        /* read cluster file */
        std::ifstream inFile(path);
        if(!inFile){
            LOG(INFO) << "open file failed. " << path;
            exit(0);
        }
        size_t size;
        vid_t v_oid, v_gid;
        vid_t cluster_num = 0;
        vertex_t u;
        auto vm_ptr = graph_->vm_ptr();
        VertexArray<vid_t, vid_t> id2clusterid; // map: vid -> clusterid
        id2clusterid.Init(graph_->Vertices(), ID_default_value);
        std::vector<std::vector<vertex_t> > clusters;
        fid_t fid = this->graph_->fid();
        vid_t max_v_id = graph_->GetVerticesNum();
        vid_t load_cnt = 0;
        while(inFile >> size){
            std::set<vertex_t> P;
            for(int i = 0; i < size; i++){
                inFile >> v_oid;
                CHECK_GE(max_v_id, v_oid);
                CHECK(vm_ptr->GetGid(v_oid, v_gid));
                fid_t v_fid = vm_ptr->GetFidFromGid(v_gid);
                if (v_fid == fid) {
                    vertex_t u;
                    CHECK(this->graph_->Gid2Vertex(v_gid, u));
                    // LOG(INFO) << v_oid << " " << v_gid << " " << u.GetValue();
                    P.insert(u);
                    id2clusterid[u] = cluster_num;
                    // for (auto e : graph_->GetOutgoingAdjList(u)) {
                    //     LOG(INFO) << "v_gid=" << v_gid << " " << e.neighbor.GetValue();
                    // }
                }
            }
            if(P.size() >= MIN_NODE_NUM){
              clusters.emplace_back(P.begin(), P.end());
              cluster_num++;
                // build_supernode_by_P(P);
            }
            if(load_cnt % 100000 == 0){
                LOG(INFO) << "load_cnt=" << load_cnt 
                          << " cluster_num=" << cluster_num << std::endl;
            }
            load_cnt++;
        }
        LOG(INFO) << "cluster_num=" << cluster_num 
                  << " clusters.size=" << clusters.size();

        vid_t init_mirror_num = get_init_supernode_by_clusters(clusters, 
                                                                id2clusterid);
        /* 将cluster序列化到文件: 注意必须放在build_supernode之前 */
        if (prefix != "") {
          serialize_cluster(prefix, init_mirror_num);
        }
        final_build_supernode(init_mirror_num);
    }

    vid_t get_init_supernode_by_clusters (std::vector<std::vector<vertex_t>> 
                                        &clusters, VertexArray<vid_t, vid_t> 
                                        &id2clusterid) {
        LOG(INFO) << "build supernode by out/in-mirror...";
        double init_supernode_by_clusters_time = GetCurrentTime();
        typedef long long count_t;
        count_t k = FLAGS_mirror_k;
        count_t mirror_num = 0;
        count_t reduce_edge_num = 0;
        count_t new_index_num = 0;
        count_t old_index_num = 0;
        count_t old_inner_edge = 0;
        count_t new_inner_edge = 0;
        count_t spnids_num = 0;
        count_t add_spnids_num = 0;
        count_t all_old_exit_node_num = 0;
        count_t all_new_exit_node_num = 0;
        count_t all_old_entry_node_num = 0;
        count_t all_new_entry_node_num = 0;
        count_t abandon_node_num = 0;
        count_t abandon_edge_num = 0;
        const vid_t spn_ids_num = clusters.size(); 
        float obj = FLAGS_compress_threshold; // 1
        count_t no_a_entry_node_num = 0;
        count_t mirror_node_num = 0;
        LOG(INFO) << " FLAGS_compress_threshold=" << FLAGS_compress_threshold;

        bool is_use_mirror = true;
        if (FLAGS_mirror_k == 1e8) {
          is_use_mirror = false;
          LOG(INFO) << "Close function of Using Mirror!";
        } else {
          LOG(INFO) << "Open function of Using Mirror!";
        }
        
        for (vid_t j = 0; j < spn_ids_num; j++) {
            std::vector<vertex_t> &node_set = clusters[j];
            std::unordered_map<vid_t, vid_t> in_frequent;
            std::unordered_map<vid_t, vid_t> out_frequent;
            std::set<vertex_t> old_entry_node;
            std::set<vertex_t> old_exit_node;
            count_t temp_old_inner_edge = 0;
            for (auto u : node_set) {
              for (auto e : this->graph_->GetIncomingAdjList(u)) {
                vid_t to_ids = id2clusterid[e.neighbor];
                if (to_ids != j) {
                  if (is_use_mirror == true) {
                    vid_t newids = id2spids[e.neighbor];
                    if (newids != ID_default_value) { // new cluster
                      auto out_mirror = supernode_out_mirror[newids];
                      if (out_mirror.find(u) == out_mirror.end()) {
                        in_frequent[e.neighbor.GetValue()] += 1;
                      } else {
                      }
                    } else {
                      in_frequent[e.neighbor.GetValue()] += 1;
                    }
                  }
                  old_entry_node.insert(u);
                } else {
                    temp_old_inner_edge++;
                }
              }
              for (auto e : this->graph_->GetOutgoingAdjList(u)) {
                vid_t to_ids = id2clusterid[e.neighbor];
                if (to_ids != j) { 
                  if (is_use_mirror == true) {
                    vid_t newids = id2spids[e.neighbor];
                    if (newids != ID_default_value) { // new cluster
                      auto in_mirror = supernode_in_mirror[newids];
                      if (in_mirror.find(u) == in_mirror.end()) {
                        out_frequent[e.neighbor.GetValue()] += 1;
                      } else {
                      }
                    } else {
                      out_frequent[e.neighbor.GetValue()] += 1;
                    }
                  }
                  old_exit_node.insert(u);
                }
              }
            }
            count_t in_edge_num = 0;
            count_t out_edge_num = 0;
            count_t in_mirror_node_num = 0;
            count_t out_mirror_node_num = 0;
            count_t old_exit_node_num = old_exit_node.size();
            count_t old_entry_node_num = old_entry_node.size();
            std::set<vertex_t> old_P;
            old_P.insert(node_set.begin(), node_set.end());
            std::set<vertex_t> in_P;
            in_P.insert(node_set.begin(), node_set.end());
            std::set<vertex_t> out_P;
            out_P.insert(node_set.begin(), node_set.end());
            std::set<vertex_t> in_mirror;
            std::set<vertex_t> out_mirror;
            
            if (is_use_mirror == true) {
              for (const auto& fre : in_frequent) {
                  if (fre.second > k) {
                      in_edge_num += fre.second;
                      in_mirror_node_num += 1;
                      in_P.insert(vertex_t(fre.first));  
                      in_mirror.insert(vertex_t(fre.first));
                  }
                  // LOG(INFO) << "in: " << fre.first << ": " << fre.second << std::endl;
              }
              for (const auto& fre : out_frequent) {
                  if (fre.second > k) {
                      out_edge_num += fre.second;
                      out_mirror_node_num += 1;
                      out_P.insert(vertex_t(fre.first));  
                      out_mirror.insert(vertex_t(fre.first));
                  }
                  // LOG(INFO) << "out: " << fre.first << ": " << fre.second << std::endl;
              }
            }

            std::set<vertex_t> B; // belong to P, bound vertices
            for(auto v : node_set){
            // parallel_for(int i = 0; i < node_set.size(); i++){
            //   vertex_t v = node_set[i];
              const auto& oes = this->graph_->GetOutgoingAdjList(v);
              for(auto& e : oes){
                if(out_P.find(e.neighbor) == out_P.end()){
                  {
                    // std::unique_lock<std::mutex> lk(set_mux_);
                    B.insert(v);
                  }
                  break;
                }
              }
            }
            std::set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
            for(auto v : node_set){
              const auto& oes = this->graph_->GetIncomingAdjList(v);
              for(auto& e : oes){
                if(in_P.find(e.neighbor) == in_P.end()){ // Mirror
                  {
                    // std::unique_lock<std::mutex> lk(set_mux_);
                    S.insert(v);
                  }
                  break;
                }
              }
            }
            count_t new_exit_node_num = B.size() + out_mirror_node_num;
            count_t new_entry_node_num = S.size() + in_mirror_node_num;
            count_t temp_old_index_num = old_exit_node_num * old_entry_node_num;
            count_t temp_entry_index_num = old_exit_node_num * new_entry_node_num;
            count_t temp_exit_index_num = new_exit_node_num * old_entry_node_num;
            count_t temp_new_index_num = new_exit_node_num * new_entry_node_num;

            const bool original_compress_condition = 
                (temp_old_index_num < temp_old_inner_edge);

            std::vector<float> benefit;
            benefit.resize(4, 0);
            benefit[0] = temp_old_inner_edge * 1.0 - temp_old_index_num; 
            benefit[1] = (temp_old_inner_edge + in_edge_num) * 1.0 
                         - (temp_entry_index_num + in_mirror_node_num);
            benefit[2] = (temp_old_inner_edge + out_edge_num) * 1.0
                         - (temp_exit_index_num + out_mirror_node_num);
            benefit[3] = (temp_old_inner_edge + in_edge_num + out_edge_num) * 1.0
                         - (temp_new_index_num + in_mirror_node_num + out_mirror_node_num);

            int max_i = 0;
            for (int i = 0; i < benefit.size(); i++) {
                if (benefit[max_i] < benefit[i]) {
                    max_i = i;
                }
            }
            float max_benefit = benefit[max_i];

            if (max_benefit <= obj) {
                abandon_edge_num += temp_old_inner_edge;
                abandon_node_num += old_P.size();
            }

            if (original_compress_condition == true) {
                spnids_num++;
                old_inner_edge += temp_old_inner_edge;
                old_index_num += temp_old_index_num;
                all_old_entry_node_num += old_entry_node_num;
                all_old_exit_node_num += old_exit_node_num;
            }
            if (max_benefit > obj) {
                if (original_compress_condition == false) { 
                    add_spnids_num++;
                } 
                new_inner_edge += temp_old_inner_edge;
                if (max_i == 0) { 
                    new_index_num += temp_old_index_num;
                    all_new_entry_node_num += old_entry_node_num;
                    all_new_exit_node_num += old_exit_node_num;
                } else if (max_i == 1) { 
                    mirror_num += in_mirror_node_num;
                    reduce_edge_num += in_edge_num;
                    new_index_num += temp_entry_index_num;
                    all_new_entry_node_num += new_entry_node_num;
                    all_new_exit_node_num += old_exit_node_num;
                } else if (max_i == 2) { 
                    mirror_num += out_mirror_node_num;
                    reduce_edge_num += out_edge_num;
                    new_index_num += temp_exit_index_num;
                    all_new_entry_node_num += old_entry_node_num;
                    all_new_exit_node_num += new_exit_node_num;
                } else if (max_i == 3) {
                    mirror_num += in_mirror_node_num;
                    mirror_num += out_mirror_node_num;
                    reduce_edge_num += in_edge_num;
                    reduce_edge_num += out_edge_num;
                    new_index_num += temp_new_index_num;
                    all_new_entry_node_num += new_entry_node_num;
                    all_new_exit_node_num += new_exit_node_num;
                } else {
                    LOG(INFO) << "no this type. max_i=" << max_i;
                    exit(0);
                }

                // get new S, in_mirror, out_mirror
                if (max_i == 0) {
                    S = old_entry_node;
                    in_mirror.clear();
                    out_mirror.clear();
                } else if (max_i == 1) {
                    out_mirror.clear();
                } else if (max_i == 2) {
                    S = old_entry_node;
                    in_mirror.clear();
                }

                int ids_id = -1;
                {
                    std::unique_lock<std::mutex> lk(supernode_ids_mux_);
                    supernode_ids.emplace_back(old_P.begin(), old_P.end());
                    ids_id = int(supernode_ids.size()) - 1; // root_id
                    cluster_ids.emplace_back(old_P.begin(), old_P.end());
                    supernode_source.emplace_back(S.begin(), S.end());
                    supernode_in_mirror.emplace_back(in_mirror.begin(), 
                                                     in_mirror.end());
                    supernode_out_mirror.emplace_back(out_mirror.begin(), 
                                                      out_mirror.end());
                    mirror_node_num += in_mirror.size();
                    mirror_node_num += out_mirror.size();
                }
                // CHECK(ids_id >= 0);
                for(auto u : old_P){
                    Fc[u] = -(ids_id+1);
                    id2spids[u] = ids_id;
                }
            }
        }
        
      return mirror_node_num;
    }

    void final_build_supernode(vid_t mirror_node_num) {
        double final_build_supernode = GetCurrentTime();
        size_t mirror_node_cnt = 0;
        size_t inmirror2source_cnt = 0;
        size_t outmirror2source_cnt = 0;
        auto new_node_range = VertexRange<vid_t>(0, 
                                              old_node_num + mirror_node_num);
        Fc_map.Init(new_node_range, ID_default_value);
        for (vid_t i = 0; i < supernode_ids.size(); i++) {
            vid_t ids_id = i;
            auto& old_P = supernode_ids[ids_id];
            auto& S = supernode_source[ids_id];
            auto& in_mirror = supernode_in_mirror[ids_id];
            auto& out_mirror = supernode_out_mirror[ids_id];
            std::vector<vertex_t> del_v;
            for(auto v : supernode_in_mirror[i]) {
                if (Fc[v] == FC_default_value) {
                    mirror_node_cnt++;
                    old_P.emplace_back(v);
                    cluster_ids[ids_id].emplace_back(v);
                    // supernode_in_mirror[ids_id].erase(v);
                    del_v.emplace_back(v);
                    const auto& ies = this->graph_->GetIncomingAdjList(v);
                    for(auto& e : ies){
                        if(id2spids[e.neighbor] != ids_id 
                        //    ){ // new source
                           && in_mirror.find(e.neighbor) == in_mirror.end()){ 
                            S.emplace_back(v);
                            inmirror2source_cnt++;
                            break;
                        }
                    }
                    Fc[v] = -(ids_id+1);
                    id2spids[v] = ids_id;
                }
            }
            for (auto v : del_v) {
                supernode_in_mirror[ids_id].erase(v);
                if (out_mirror.find(v) != out_mirror.end()) { 
                    out_mirror.erase(v);
                }
            }
            del_v.clear();
            for(auto v : supernode_out_mirror[i]) {
                if (Fc[v] == FC_default_value) {
                    mirror_node_cnt++;
                    old_P.emplace_back(v);
                    cluster_ids[ids_id].emplace_back(v);
                    del_v.emplace_back(v);
                    const auto& ies = this->graph_->GetIncomingAdjList(v);
                    for(auto& e : ies){
                        if(id2spids[e.neighbor] != ids_id 
                            // ){ // new source
                           && in_mirror.find(e.neighbor) == in_mirror.end()){
                            S.emplace_back(v);
                            outmirror2source_cnt++;
                            break;
                        }
                    }
                    Fc[v] = -(ids_id+1);
                    id2spids[v] = ids_id;
                }
            }
            for (auto v : del_v) {
                supernode_out_mirror[ids_id].erase(v);
                if (in_mirror.find(v) != in_mirror.end()) {
                    in_mirror.erase(v);
                }
            }
            cluster_in_mirror_ids.resize(cluster_in_mirror_ids.size() + 1);
            cluster_out_mirror_ids.resize(cluster_out_mirror_ids.size() + 1);
            for (auto v : in_mirror) {
                vid2in_mirror_cluster_ids[v.GetValue()].emplace_back(ids_id);
                vid_t mirror_id = all_node_num++; // have a lock
                vertex_t m(mirror_id);
                mirrorid2vid[m] = v;
                cluster_ids[ids_id].emplace_back(m);
                cluster_in_mirror_ids[ids_id].emplace_back(m);
            }
            for (auto v : out_mirror) {
                vid_t mirror_id = all_node_num++; // have a lock
                vertex_t m(mirror_id);
                mirrorid2vid[m] = v;
                cluster_ids[ids_id].emplace_back(m);
                cluster_out_mirror_ids[ids_id].emplace_back(m);
            }
            for(auto src : S){
                Fc[src] = ids_id;
                /* build a supernode */
                vid_t supernode_id = __sync_fetch_and_add(&supernodes_num, 1);
                Fc_map[src] = supernode_id;
                shortcuts[src.GetValue()][ids_id] = supernode_id;
                supernodes[supernode_id].id = src;
                supernodes[supernode_id].ids = ids_id; // root_id
            }
            for (auto mid : cluster_in_mirror_ids[ids_id]) {
                vid_t supernode_id = __sync_fetch_and_add(&supernodes_num, 1);
                // vid2in_mirror_cluster_ids[v.GetValue()].emplace_back(ids_id);
                vid2in_mirror_mids[mirrorid2vid[mid].GetValue()].emplace_back(
                                                                mid.GetValue());
                Fc_map[mid] = supernode_id;
                auto src = mirrorid2vid[mid];
                shortcuts[src.GetValue()][ids_id] = supernode_id;
                // supernodes[supernode_id].id = src;
                supernodes[supernode_id].id = mid; // source
                supernodes[supernode_id].ids = ids_id; // root_id
                // LOG(INFO) << "======2oid=" << this->graph_->GetId(src);
            }
            // get vertex's mirror address
            for (auto mid : cluster_out_mirror_ids[ids_id]) {
                vid2out_mirror_mids[mirrorid2vid[mid].GetValue()].emplace_back(
                                                                mid.GetValue());
            } 
        }

        // Fc_map.Resize(new_node_range);
        VertexArray<vid_t, vid_t> new_id2spids;
        new_id2spids.Init(new_node_range, ID_default_value);
        parallel_for(vid_t i = 0; i < old_node_num; i++) {
            vertex_t v(i);
            new_id2spids[v] = id2spids[v];
        }
        parallel_for(vid_t i = 0; i < this->cluster_ids.size(); i++) {
            for(auto v : this->cluster_in_mirror_ids[i]) {
                new_id2spids[v] = i;
            }
            for(auto v : this->cluster_out_mirror_ids[i]) {
                new_id2spids[v] = i;
            }
        }
        id2spids.Init(new_node_range);
        for (vertex_t v : new_node_range) {
            id2spids[v] = new_id2spids[v];
        }

        for(vid_t j = 0; j < this->cluster_ids.size(); j++){
          for (auto u : this->cluster_out_mirror_ids[j]) {
            this->all_out_mirror.emplace_back(u);
          }
        }

        this->indegree.resize(this->cluster_ids.size()+1);
        parallel_for (vid_t i = 0; i < this->cluster_ids.size(); i++) {
          vid_t sum = 0;
          for (auto v : this->supernode_ids[i]) {
            sum += this->graph_->GetIncomingAdjList(v).Size();
          }
          this->indegree[i] = sum;
        }
        this->indegree[this->GetClusterSize()] = this->graph_->GetEdgeNum() / 2;
    }

    void build_subgraph_mirror(const std::shared_ptr<fragment_t>& new_graph) {
        //subgraph
        double subgraph_time = GetCurrentTime();
        const vid_t spn_ids_num = this->supernode_ids.size();
        vid_t inner_node_num = this->all_node_num;
        subgraph.resize(inner_node_num);
        // std::vector<size_t> ia_oe_degree(inner_node_num+1, 0);
        vid_t ia_oe_num = 0; 
        parallel_for(vid_t i = 0; i < spn_ids_num; i++){
            std::vector<vertex_t> &node_set = this->supernode_ids[i];
            std::vector<vertex_t> &in_mirror_ids 
                                    = this->cluster_in_mirror_ids[i];
            std::vector<vertex_t> &out_mirror_ids 
                                    = this->cluster_out_mirror_ids[i];
            vid_t temp_a = 0;
            // auto ids_id = this->id2spids[*(node_set.begin())];
            // CHECK_EQ(ids_id, i);
            auto ids_id = i;
            for(auto v : node_set){
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                for(auto& oe : oes){
                    if(this->id2spids[oe.neighbor] == ids_id){ // inner edge
                        subgraph[v.GetValue()].emplace_back(oe);
                    }
                }
            }
            for(auto m_id : in_mirror_ids){
                vertex_t v = mirrorid2vid[m_id];
                auto v_superid = this->id2spids[v];
                std::unordered_set<vertex_t>& v_out_mirror = 
                    this->supernode_out_mirror[v_superid];
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                for(auto& oe : oes){
                    if(this->id2spids[oe.neighbor] == ids_id
                         ){ // in-mirror edge
                        // && v_out_mirror.find(oe.neighbor) == v_out_mirror.end()){ // in-mirror edge
                        subgraph[m_id.GetValue()].emplace_back(nbr_t(oe.neighbor,
                                                                     oe.data));
                    }
                }
            }
            for(auto m_id : out_mirror_ids){
                vertex_t v = mirrorid2vid[m_id];
                auto v_superid = this->id2spids[v];
                std::unordered_set<vertex_t>& v_in_mirror = 
                    this->supernode_in_mirror[v_superid]; 
                const auto& ies = new_graph->GetIncomingAdjList(v);
                for(auto& ie : ies){
                    if(this->id2spids[ie.neighbor] == ids_id
                        //  ){ // out-mirror edge
                        && v_in_mirror.find(ie.neighbor) == v_in_mirror.end()){ // out-mirror edge
                        subgraph[ie.neighbor.GetValue()].emplace_back(nbr_t(m_id,
                                                                      ie.data));
                    }
                }
            }
        }

        this->subgraph_old = this->subgraph; // use to update.
    }

    void inc_build_subgraph_mirror(const std::shared_ptr<fragment_t>& new_graph) {
        //subgraph
        const vid_t spn_ids_num = this->update_cluster_ids.size();
        // vid_t inner_node_num = this->all_node_num;
        parallel_for(vid_t i = 0; i < spn_ids_num; i++){
            auto ids_id = this->update_cluster_ids[i];
            std::vector<vertex_t> &node_set = this->supernode_ids[ids_id];
            std::vector<vertex_t> &in_mirror_ids 
                                    = this->cluster_in_mirror_ids[ids_id];
            std::vector<vertex_t> &out_mirror_ids 
                                    = this->cluster_out_mirror_ids[ids_id];
            vid_t temp_a = 0;
            // auto ids_id = this->id2spids[*(node_set.begin())];
            // CHECK_EQ(ids_id, i);
            for(auto v : node_set){
                subgraph[v.GetValue()].clear();
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                for(auto& oe : oes){
                    if(this->id2spids[oe.neighbor] == ids_id){ // inner edge
                        subgraph[v.GetValue()].emplace_back(oe);
                    }
                }
            }
            for(auto m_id : in_mirror_ids){
                subgraph[m_id.GetValue()].clear();
                vertex_t v = mirrorid2vid[m_id];
                auto v_superid = this->id2spids[v];
                std::unordered_set<vertex_t>& v_out_mirror = 
                    this->supernode_out_mirror[v_superid];
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                for(auto& oe : oes){
                    if(this->id2spids[oe.neighbor] == ids_id
                         ){ // in-mirror edge
                        // && v_out_mirror.find(oe.neighbor) == v_out_mirror.end()){ // in-mirror edge
                        subgraph[m_id.GetValue()].emplace_back(nbr_t(oe.neighbor,
                                                                     oe.data));
                    }
                }
            }
            for(auto m_id : out_mirror_ids){
                vertex_t v = mirrorid2vid[m_id];
                auto v_superid = this->id2spids[v];
                std::unordered_set<vertex_t>& v_in_mirror = 
                    this->supernode_in_mirror[v_superid]; 
                const auto& ies = new_graph->GetIncomingAdjList(v);
                for(auto& ie : ies){
                    if(this->id2spids[ie.neighbor] == ids_id
                        //  ){ // out-mirror edge
                        && v_in_mirror.find(ie.neighbor) == v_in_mirror.end()){ // out-mirror edge
                        subgraph[ie.neighbor.GetValue()].emplace_back(nbr_t(m_id,
                                                                      ie.data));
                    }
                }
            }
        }
    }

    void judge_out_bound_node(const std::shared_ptr<fragment_t>& new_graph) {
        const vid_t spn_ids_num = this->supernode_ids.size();
        bool compressor_flags_cilk = true;
        if (compressor_flags_cilk) {
#ifdef INTERNAL_PARALLEL
            LOG(FATAL) << "Layph is not compiled with -DUSE_CILK";
#endif
            parallel_for(vid_t j = 0; j < spn_ids_num; j++){  // parallel compute
                this->judge_out_bound_node_detail(j, new_graph);
            }
        }
        else{
#pragma omp parallel for num_threads(NUM_THREADS)
            for(vid_t j = 0; j < spn_ids_num; j++){  // parallel compute
                this->judge_out_bound_node_detail(j, new_graph);
            }
#pragma omp barrier
        }
        size_t not_bound = 0;
        for (vid_t i = 0; i < supernode_in_mirror.size(); i++) {
            for(auto v : supernode_in_mirror[i]) {
                if (this->supernode_out_bound[v.GetValue()] == false) {
                    not_bound++;
                }
                this->supernode_out_bound[v.GetValue()] = true;
            }
        }
    }

    void inc_judge_out_bound_node(const std::shared_ptr<fragment_t>& new_graph) {
        bool compressor_flags_cilk = true;
        vid_t update_size = this->update_cluster_ids.size();
        if (compressor_flags_cilk) {
#ifdef INTERNAL_PARALLEL
            LOG(FATAL) << "System is not compiled with -DUSE_CILK";
#endif
            parallel_for(vid_t j = 0; j < update_size; j++){  // parallel compute
                this->judge_out_bound_node_detail(this->update_cluster_ids[j], new_graph);
            }
        }
        else{
#pragma omp parallel for num_threads(NUM_THREADS)
            for(vid_t j = 0; j < update_size; j++){  // parallel compute
                this->judge_out_bound_node_detail(this->update_cluster_ids[j], new_graph);
            }
#pragma omp barrier
        }
        size_t not_bound = 0;
        for (vid_t i = 0; i < update_size; i++) {
            for(auto v : supernode_in_mirror[this->update_cluster_ids[i]]) {
                if (this->supernode_out_bound[v.GetValue()] == false) {
                    not_bound++;
                }
                this->supernode_out_bound[v.GetValue()] = true;
            }
        }
    }


    void judge_out_bound_node_detail(const vid_t ids_id, 
                                     const std::shared_ptr<fragment_t>& new_graph){
        std::vector<vertex_t> &node_set = this->supernode_ids[ids_id]; 
        std::unordered_set<vertex_t> &out_mirror = this->supernode_out_mirror[ids_id]; 
        for(auto v : node_set){
            const auto& oes = new_graph->GetOutgoingAdjList(v);
            for(auto& e : oes){
                if(this->id2spids[e.neighbor] != ids_id 
                    && out_mirror.find(e.neighbor) == out_mirror.end()){
                    this->supernode_out_bound[v.GetValue()] = true;
                    break;
                }
            }
        }
    } 

    void print_cluster() {
        LOG(INFO) << "---------------print_cluster--------------------";
        for (vid_t i = 0; i < supernode_ids.size(); i++) {
            LOG(INFO) << "---------------------------";
            LOG(INFO) << "spids_id=" << i;
            LOG(INFO) << " P:";
            for (auto p : supernode_ids[i]) { 
                LOG(INFO) << v2Oid(p);
            } 
            LOG(INFO) << " source:";
            for (auto p : supernode_source[i]) { 
                LOG(INFO) << v2Oid(p);
            } 
            LOG(INFO) << " supernode_in_mirror:";
            for (auto p : supernode_in_mirror[i]) { 
                LOG(INFO) << v2Oid(p);
            } 
            LOG(INFO) << " supernode_out_mirror:";
            for (auto p : supernode_out_mirror[i]) { 
                LOG(INFO) << v2Oid(p);
            } 
        }
        LOG(INFO) << "===============================================";
    }
    
    void print_subgraph() {
        LOG(INFO) << "----------------print_subraph-------------------";
        // print cluster
        for(auto vs : cluster_ids) {
            LOG(INFO) << " ----";
            for (auto v : vs) {
                for (auto e : subgraph[v.GetValue()]) {
                    LOG(INFO) << " " << v2Oid(v) << "->" << v2Oid(e.neighbor);
                }
            }
        }
    }

    vid_t vid2Oid(vid_t vid) {
        if (vid < old_node_num) {
            vertex_t v(vid);
            return graph_->GetId(v);
        } else {
            return vid;
        }
    }

    vid_t v2Oid(vertex_t v) {
        if (v.GetValue() < old_node_num) {
            return graph_->GetId(v);
        } else {
            return v.GetValue();
        }
    }

    /**
     * Provide node type to worker.
    */
    void get_nodetype(vid_t inner_node_num, std::vector<char>& node_type) {
        node_type.clear();
        node_type.resize(inner_node_num, std::numeric_limits<char>::max());
        parallel_for(vid_t i = 0; i < inner_node_num; i++) {
          vertex_t u(i);
          if (this->Fc[u] == this->FC_default_value) {
            node_type[i] = NodeType::SingleNode; // out node
          } else if (this->Fc[u] >= 0) {
            node_type[i] = NodeType::OnlyInNode; // source node
          } else if(!this->supernode_out_bound[i]) {
            node_type[i] = NodeType::InnerNode; // inner node
          }
          if (this->supernode_out_bound[i]) {
            node_type[i] = NodeType::OnlyOutNode; // bound node
            if (this->Fc[u] >= 0) {
              node_type[i] = NodeType::BothOutInNode; // source node + bound node
            }
          }
      }
      for(vid_t i = 0; i < this->supernode_in_mirror.size(); i++) { // can'nt parallel
        for(auto v : this->supernode_in_mirror[i]) {
          // LOG(INFO) << "----vid=" << v.GetValue();
          if(node_type[v.GetValue()] == NodeType::OnlyOutNode 
            //  || node_type[v.GetValue()] == NodeType::SingleNode 
             || node_type[v.GetValue()] == NodeType::BothOutInNode) {
                node_type[v.GetValue()] = NodeType::BothOutInNode;
          } else {
            node_type[v.GetValue()] = NodeType::OnlyInNode; // in-node or inner node
          }
        }
      }
    }

    void get_nodetype_mirror(vid_t inner_node_num, std::vector<char>& node_type) {
      node_type.clear();
      node_type.resize(inner_node_num, std::numeric_limits<char>::max());
      parallel_for(vid_t i = 0; i < inner_node_num; i++) {
        vertex_t u(i);
        if (this->Fc[u] == this->FC_default_value) {
          node_type[i] = NodeType::SingleNode; // out node
        } else if (this->Fc[u] >= 0) {
          node_type[i] = NodeType::OnlyInNode; // source node
        } else if(!this->supernode_out_bound[i]) {
          node_type[i] = NodeType::InnerNode; // inner node
        }
        if (this->supernode_out_bound[i]) {
          node_type[i] = NodeType::OnlyOutNode; // bound node
          if (this->Fc[u] >= 0) {
            node_type[i] = NodeType::BothOutInNode; // source node + bound node
          }
        }
      }
      for(vid_t i = 0; i < this->supernode_in_mirror.size(); i++) { // can'nt parallel
        for(auto v : this->supernode_in_mirror[i]) {
          if (node_type[v.GetValue()] == NodeType::OnlyOutNode) {
            node_type[v.GetValue()] = NodeType::OutMaster;
          } else if (node_type[v.GetValue()] == NodeType::BothOutInNode) {
            node_type[v.GetValue()] = NodeType::BothOutInMaster;
          } else if (node_type[v.GetValue()] == NodeType::OnlyInNode) {
          }
        }
      }
    }

    void sketch2csr_merge(std::vector<char>& node_type){
      double transfer_csr_time = GetCurrentTime();
      double init_time_1 = GetCurrentTime();
      auto inner_vertices = graph_->InnerVertices();
      vid_t inner_node_num = inner_vertices.end().GetValue() 
                             - inner_vertices.begin().GetValue();
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
        const char type = node_type[i];
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){
          for(auto mp : this->shortcuts[i]) {
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            is_e_degree[i+1] += spnode.bound_delta.size();
            // atomic_add(source_e_num, spnode.bound_delta.size());
          }
        }
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = 
                                            this->supernode_out_mirror[ids_id];
          size_t temp_cnt = 0;
          for(vid_t j = 0; j < out_degree; j++){ 
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  temp_cnt += 1;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  temp_cnt += 1;
              }
            }
          }
          ib_e_degree[i+1] += temp_cnt;
          // atomic_add(bound_e_num, temp_cnt);
        } else if (NodeType::SingleNode == type) { // edge
          auto oes = graph_->GetOutgoingAdjList(u);
          size_t temp_cnt = oes.Size();
          ib_e_degree[i+1] = temp_cnt;
        }
      }
      LOG(INFO) << "csr_time_1=" << (GetCurrentTime()-csr_time_1); //1.25987

      /* get index start */
      double index_time = GetCurrentTime();
      for(vid_t i = 1; i <= inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
      }
      bound_e_num = ib_e_degree[inner_node_num];
      source_e_num = is_e_degree[inner_node_num];
      LOG(INFO) << " bound_e_num=" << bound_e_num;
      LOG(INFO) << " source_e_num=" << source_e_num;
      LOG(INFO) << "index_time=" << (GetCurrentTime()-index_time); //0.226317

      LOG(INFO) << "inner_node_num=" << inner_node_num;
      LOG(INFO) << "inner_node_num=" << graph_->GetVerticesNum();

      double init_time_2 = GetCurrentTime();
      is_e_.resize(source_e_num);
      ib_e_.resize(bound_e_num);
      is_e_offset_.resize(inner_node_num+1);
      ib_e_offset_.resize(inner_node_num+1);
      LOG(INFO) << "init_time_2=" << (GetCurrentTime()-init_time_2); //1.13601

      // build index/edge
      double csr_time_2 = GetCurrentTime();
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        /* source node */
        vid_t index_s = is_e_degree[i];
        is_e_offset_[i] = &is_e_[index_s];
        const char type = node_type[i];
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){
          for(auto mp : this->shortcuts[i]) {
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            for(auto& oe : spnode.bound_delta){
                // is_e_[index_s] = nbr_index_t(oe.first.GetValue(), oe.second);
                is_e_[index_s].neighbor = oe.first;
                is_e_[index_s].data = oe.second;
                index_s++;
            }
          }
        }
        /* inner_bound node */
        vid_t index_b = ib_e_degree[i];
        ib_e_offset_[i] = &ib_e_[index_b];
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = 
                                            this->supernode_out_mirror[ids_id];
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            }
          }
        } 
        else if (NodeType::SingleNode == type) { // edge
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            ib_e_[index_b] = e;
            index_b++;
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      LOG(INFO) << "csr_time_2=" << (GetCurrentTime()-csr_time_2); //0.207281

      parallel_for (vid_t i = 0; i < inner_node_num; ++i) {
        std::sort(is_e_offset_[i], is_e_offset_[i + 1],
                [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                });
      }

      LOG(INFO) << " transfer_csr_time=" << (GetCurrentTime()- transfer_csr_time);
    }

    void sketch2csr_divide(std::vector<char>& node_type){
      auto inner_vertices = graph_->InnerVertices();
      vid_t inner_node_num = inner_vertices.end().GetValue() 
                             - inner_vertices.begin().GetValue();
      is_e_.clear();
      is_e_offset_.clear();
      im_e_.clear();
      im_e_offset_.clear();
      om_e_.clear();
      om_e_offset_.clear();
      oim_e_.clear();
      oim_e_offset_.clear();
      ib_e_.clear();
      ib_e_offset_.clear();
      size_t source_e_num = 0;
      size_t bound_e_num = 0;
      size_t out_mirror_e_num = 0;
      size_t out_imirror_e_num = 0;
      size_t in_mirror_e_num = 0;
      std::vector<size_t> is_e_degree(inner_node_num+1, 0);
      std::vector<size_t> im_e_degree(inner_node_num+1, 0);
      std::vector<size_t> om_e_degree(inner_node_num+1, 0);
      std::vector<size_t> oim_e_degree(inner_node_num+1, 0);
      std::vector<size_t> ib_e_degree(inner_node_num+1, 0);

      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        const char type = node_type[i];
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){
          const vid_t ids_id = this->id2spids[u];
          for(auto mp : this->shortcuts[i]) {
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            // is_e_degree[i+1] += spnode.bound_delta.size();
            if (ids_id == mp.first) { // origin index
              size_t origin_size = 0;
              for (auto e : spnode.bound_delta) {
                if (this->id2spids[e.first] == ids_id) {
                    origin_size++;
                }
              }
              is_e_degree[i+1] += origin_size;
              om_e_degree[i+1] += (spnode.bound_delta.size() - origin_size);
            } else { // mirror index
            //   im_e_degree[i+1] += spnode.bound_delta.size();
              size_t in_size = 0;
              for (auto e : spnode.bound_delta) {
                if (this->id2spids[e.first] == mp.first) { // out-mirror
                    in_size++;
                }
              }
              im_e_degree[i+1] += in_size;
              oim_e_degree[i+1] += (spnode.bound_delta.size() - in_size);
            }
          }
        }
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = 
                                            this->supernode_out_mirror[ids_id];
          size_t temp_cnt = 0;
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  temp_cnt += 1;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  temp_cnt += 1;
              }
            }
          }
          ib_e_degree[i+1] += temp_cnt;
          // atomic_add(bound_e_num, temp_cnt);
        } else if (NodeType::SingleNode == type) { // edge
          auto oes = graph_->GetOutgoingAdjList(u);
          size_t temp_cnt = oes.Size();
          ib_e_degree[i+1] = temp_cnt;
        }
      }
      /* get index start */
      for(vid_t i = 1; i <= inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
        im_e_degree[i] += im_e_degree[i-1];
        om_e_degree[i] += om_e_degree[i-1];
        oim_e_degree[i] += oim_e_degree[i-1];
      }
      bound_e_num = ib_e_degree[inner_node_num];
      source_e_num = is_e_degree[inner_node_num];
      in_mirror_e_num = im_e_degree[inner_node_num];
      out_mirror_e_num = om_e_degree[inner_node_num];
      out_imirror_e_num = oim_e_degree[inner_node_num];

      is_e_.resize(source_e_num);
      ib_e_.resize(bound_e_num);
      im_e_.resize(in_mirror_e_num);
      om_e_.resize(out_mirror_e_num);
      oim_e_.resize(out_imirror_e_num);
      is_e_offset_.resize(inner_node_num+1);
      ib_e_offset_.resize(inner_node_num+1);
      im_e_offset_.resize(inner_node_num+1);
      om_e_offset_.resize(inner_node_num+1);
      oim_e_offset_.resize(inner_node_num+1);

      // build index/edge
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        /* source node */
        vid_t index_s = is_e_degree[i];
        is_e_offset_[i] = &is_e_[index_s];
        vid_t index_im = im_e_degree[i];
        im_e_offset_[i] = &im_e_[index_im];
        vid_t index_om = om_e_degree[i];
        om_e_offset_[i] = &om_e_[index_om];
        vid_t index_oim = oim_e_degree[i];
        oim_e_offset_[i] = &oim_e_[index_oim];
        const char type = node_type[i];
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){
          const vid_t ids_id = this->id2spids[u];
          for(auto mp : this->shortcuts[i]) {
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            // is_e_degree[i+1] += spnode.bound_delta.size();
            if (ids_id == mp.first) { // origin index
              for (auto oe : spnode.bound_delta) {
                if (this->id2spids[oe.first] == ids_id) {
                    is_e_[index_s].neighbor = oe.first;
                    is_e_[index_s].data = oe.second;
                    index_s++;
                } else {
                    om_e_[index_om].neighbor = oe.first;
                    om_e_[index_om].data = oe.second;
                    index_om++;
                }
              }
            } else { // mirror index
              for(auto& oe : spnode.bound_delta){
                if (this->id2spids[oe.first] == mp.first) { // out-mirror
                    im_e_[index_im].neighbor = oe.first;
                    im_e_[index_im].data = oe.second;
                    index_im++;
                } else {
                    oim_e_[index_oim].neighbor = oe.first;
                    oim_e_[index_oim].data = oe.second;
                    index_oim++;
                }
              }
            }
          }
        }
        /* inner_bound node */
        vid_t index_b = ib_e_degree[i];
        ib_e_offset_[i] = &ib_e_[index_b];
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = 
                                            this->supernode_out_mirror[ids_id];
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            }
          }
        } 
        else if (NodeType::SingleNode == type) { // edge
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            ib_e_[index_b] = e;
            index_b++;
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      im_e_offset_[inner_node_num] = &im_e_[in_mirror_e_num-1] + 1;
      om_e_offset_[inner_node_num] = &om_e_[out_mirror_e_num-1] + 1;
      oim_e_offset_[inner_node_num] = &oim_e_[out_imirror_e_num-1] + 1;

      parallel_for (vid_t i = 0; i < inner_node_num; ++i) {
        std::sort(is_e_offset_[i], is_e_offset_[i + 1],
                [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                });
      }
    }

    void sketch2csr_mirror(std::vector<char>& node_type){
      auto inner_vertices = graph_->InnerVertices();
      // vid_t inner_node_num = inner_vertices.end().GetValue() 
      //                        - inner_vertices.begin().GetValue();
      vid_t inner_node_num = this->all_node_num;
      is_e_.clear();
      is_e_offset_.clear();
      ib_e_.clear();
      ib_e_offset_.clear();
      sync_e_.clear();
      sync_e_offset_.clear();
      size_t source_e_num = 0;
      size_t bound_e_num = 0;
      size_t sync_e_num = 0;
      std::vector<size_t> is_e_degree(inner_node_num+1, 0);
      std::vector<size_t> ib_e_degree(inner_node_num+1, 0);
      std::vector<size_t> sync_e_degree(inner_node_num+1, 0);

      {
        vid_t cluster_size = this->supernode_in_mirror.size();
        std::vector<std::vector<vertex_t> > syncE;
        syncE.resize(inner_node_num);
        for(vid_t i = 0; i < cluster_size; i++) { // can'nt parallel
          for(auto v : this->cluster_in_mirror_ids[i]) {
            syncE[this->mirrorid2vid[v].GetValue()].emplace_back(v);
            sync_e_degree[this->mirrorid2vid[v].GetValue()+1]++; // master -> in-mirror
          }
          for(auto v : this->cluster_out_mirror_ids[i]) {
            syncE[v.GetValue()].emplace_back(this->mirrorid2vid[v]);
            sync_e_degree[v.GetValue()+1]++;  // out-mirror -> master
          }
        }
        for(vid_t i = 1; i <= inner_node_num; i++) {
          sync_e_degree[i] += sync_e_degree[i-1];
        }
        sync_e_num = sync_e_degree[inner_node_num];
        sync_e_.resize(sync_e_num);
        sync_e_offset_.resize(inner_node_num+1);
        parallel_for(vid_t i = 0; i < inner_node_num; i++) {
          vid_t index = sync_e_degree[i];
          sync_e_offset_[i] = &sync_e_[index];
          for (auto v : syncE[i]) {
            sync_e_[index].neighbor = v;
            index++;
          }
        }
        sync_e_offset_[inner_node_num] = &sync_e_[sync_e_num-1] + 1;
      }

      // in-mirror
      parallel_for(vid_t i = this->old_node_num; i < this->all_node_num; i++) {
        vertex_t u(i);
        vid_t sp_id = Fc_map[u];
        if (sp_id != ID_default_value) {
          // LOG(INFO) << " u.oid=" << this->v2Oid(u) << " sp_id=" << sp_id;
          supernode_t &spnode = this->supernodes[sp_id];
          is_e_degree[i+1] += spnode.bound_delta.size();
        }
      }

      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        const char type = node_type[i];
        // LOG(INFO) << "i=" << i << " oid=" << this->vid2Oid(i)
        //           << " type=" << int(type);
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode
           || type == NodeType::BothOutInMaster){
          // for(auto mp : this->shortcuts[i]) {
            vid_t sp_id = Fc_map[u];
            supernode_t &spnode = this->supernodes[sp_id];
            is_e_degree[i+1] += spnode.bound_delta.size();
            // atomic_add(source_e_num, spnode.bound_delta.size());
          // }
        }
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode
           || type == NodeType::BothOutInMaster || type == NodeType::OutMaster){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = 
                                            this->supernode_out_mirror[ids_id];
          size_t temp_cnt = 0;
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  temp_cnt += 1;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  temp_cnt += 1;
              }
            }
          }
          ib_e_degree[i+1] += temp_cnt;
          // atomic_add(bound_e_num, temp_cnt);
        } else if (NodeType::SingleNode == type) { // edge
          auto oes = graph_->GetOutgoingAdjList(u);
          size_t temp_cnt = oes.Size();
          ib_e_degree[i+1] = temp_cnt;
        }
      }

      /* get index start */
      for(vid_t i = 1; i <= inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
      }
      bound_e_num = ib_e_degree[inner_node_num];
      source_e_num = is_e_degree[inner_node_num];

      is_e_.resize(source_e_num);
      ib_e_.resize(bound_e_num);
      is_e_offset_.resize(inner_node_num+1);
      ib_e_offset_.resize(inner_node_num+1);

      // build index/edge
      // in-mirror
      parallel_for(vid_t i = this->old_node_num; i < this->all_node_num; i++) {
        vertex_t u(i);
        vid_t index_s = is_e_degree[i];
        is_e_offset_[i] = &is_e_[index_s];
        vid_t index_b = ib_e_degree[i];
        ib_e_offset_[i] = &ib_e_[index_b];  // must init
        vid_t sp_id = Fc_map[u];
        if (sp_id != ID_default_value) {
          supernode_t &spnode = this->supernodes[sp_id];
          for(auto& oe : spnode.bound_delta){
            // is_e_[index_s] = nbr_index_t(oe.first.GetValue(), oe.second);
            is_e_[index_s].neighbor = oe.first;
            is_e_[index_s].data = oe.second;
            index_s++;
          }
        }
      }
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        /* source node */
        vid_t index_s = is_e_degree[i];
        is_e_offset_[i] = &is_e_[index_s];
        const char type = node_type[i];
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode
           || type == NodeType::BothOutInMaster){
          // for(auto mp : this->shortcuts[i]) {
            vid_t sp_id = Fc_map[u];
            supernode_t &spnode = this->supernodes[sp_id];
            for(auto& oe : spnode.bound_delta){
                // is_e_[index_s] = nbr_index_t(oe.first.GetValue(), oe.second);
                is_e_[index_s].neighbor = oe.first;
                is_e_[index_s].data = oe.second;
                index_s++;
            }
          // }
        }
        /* inner_bound node */
        vid_t index_b = ib_e_degree[i];
        ib_e_offset_[i] = &ib_e_[index_b];
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode
           || type == NodeType::BothOutInMaster || type == NodeType::OutMaster){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = 
                                            this->supernode_out_mirror[ids_id];
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            }
          }
        } 
        else if (NodeType::SingleNode == type) { // edge
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            ib_e_[index_b] = e;
            index_b++;
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      parallel_for (vid_t i = 0; i < inner_node_num; ++i) {
        std::sort(is_e_offset_[i], is_e_offset_[i + 1],
                [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                });
      }
    }

    void sketch2csr_renumber(vid_t inner_node_num,
                            std::vector<char>& node_type,
                            Array<vid_t, Allocator<vid_t>>& oldId2newId,
                            Array<vid_t, Allocator<vid_t>>& newId2oldId,
                            Array<vid_t, Allocator<vid_t>>& oldGid2newGid,
                            Array<vid_t, Allocator<vid_t>>& newGid2oldGid,
                            std::vector<vid_t>& node_range,
                            std::vector<std::vector<vertex_t>>& all_nodes,
                            Array<nbr_index_t, Allocator<nbr_index_t>>& is_e_,
                            Array<nbr_index_t*, Allocator<nbr_index_t*>>& is_e_offset_,
                            Array<nbr_t, Allocator<nbr_t>>& ib_e_,
                            Array<nbr_t*, Allocator<nbr_t*>>& ib_e_offset_
                            ) {
      all_nodes.clear();
      all_nodes.resize(5);
      for(vid_t i = 0; i < inner_node_num; i++) {
          all_nodes[node_type[i]].emplace_back(vertex_t(i));
      }
      
      /* renumber internal vertices */
      oldId2newId.clear();
      oldId2newId.resize(inner_node_num);
      newId2oldId.clear();
      newId2oldId.resize(inner_node_num);
      oldGid2newGid.clear();
      oldGid2newGid.resize(inner_node_num);
      newGid2oldGid.clear();
      newGid2oldGid.resize(inner_node_num);
      node_range.clear();
      node_range.resize(6);
      vid_t index_id = 0;
      for (vid_t i = 0; i < 5; i++) {
        const std::vector<vertex_t>& nodes = all_nodes[i];
        size_t size = nodes.size();
        node_range[i] = index_id;
        parallel_for (vid_t j = 0; j < size; j++) {
          oldId2newId[nodes[j].GetValue()] = index_id + j;
          newId2oldId[index_id + j] = nodes[j].GetValue();
          vid_t old_gid = this->graph_->Vertex2Gid(nodes[j]);
          vid_t new_gid = this->graph_->Vertex2Gid(vertex_t(index_id + j));
          oldGid2newGid[old_gid] = new_gid;
          newGid2oldGid[new_gid] = old_gid;
        }
        index_id += size;
      }
      node_range[5] = index_id;

      double  transfer_csr_time = GetCurrentTime();

      /* source to in_bound_node */
      is_e_.clear();
      is_e_offset_.clear();
      ib_e_.clear();
      ib_e_offset_.clear();
      size_t source_e_num = 0;
      size_t bound_e_num = 0;
      std::vector<size_t> is_e_degree(inner_node_num+1, 0);
      std::vector<size_t> ib_e_degree(inner_node_num+1, 0);

      parallel_for(vid_t i = 0; i < inner_node_num; i++) {
        vertex_t u(i);
        char type = node_type[i];
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){ // index
          for(auto mp : this->shortcuts[i]) {
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            is_e_degree[oldId2newId[i]+1] += spnode.bound_delta.size(); // Note: Accumulation is used here.
          }
        }
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){ // edge
          auto oes = this->graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = this->supernode_out_mirror[ids_id];
          size_t temp_cnt = 0;
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  // bound_e_num += 1;
                  temp_cnt += 1;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  // bound_e_num += 1;
                  temp_cnt += 1;
              }
            }
          }
          // ib_e_degree[i+1] += temp_cnt;
          ib_e_degree[oldId2newId[i]+1] += temp_cnt;
        } else if (NodeType::SingleNode == type) { // edge
          auto oes = this->graph_->GetOutgoingAdjList(u);
          size_t temp_cnt = oes.Size();
          // ib_e_degree[i+1] += temp_cnt;
          ib_e_degree[oldId2newId[i]+1] = temp_cnt;
        }
      }

      /* get index start */
      double index_time = GetCurrentTime();
      for(vid_t i = 1; i <= inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
      }
      bound_e_num = ib_e_degree[inner_node_num];
      source_e_num = is_e_degree[inner_node_num];

      is_e_.resize(source_e_num);
      ib_e_.resize(bound_e_num);
      is_e_offset_.resize(inner_node_num+1);
      ib_e_offset_.resize(inner_node_num+1);

      /* build index/edge */
      parallel_for(vid_t i = 0; i < inner_node_num; i++) {
        vertex_t u(i);
        /* source node */
        vid_t new_id = oldId2newId[i];
        vid_t index_s = is_e_degree[new_id];
        is_e_offset_[new_id] = &is_e_[index_s];
        char type = node_type[i];
        // LOG(INFO) << "i=" << i << " oid=" << this->vid2Oid(new_id) << " type=" << int(type);
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){ // index
          for(auto mp : this->shortcuts[i]) {
            // LOG(INFO) << " mp.second=" << mp.second;
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            for(auto& oe : spnode.bound_delta){
              // LOG(INFO) << " -sp.oid=" << this->v2Oid(spnode.id) << "->"
              //           << this->v2Oid(oe.first) << " data=" << oe.second;
              if (oe.first.GetValue() < inner_node_num) {
                is_e_[index_s].neighbor = oldId2newId[oe.first.GetValue()];
              } else {
                is_e_[index_s].neighbor = oe.first;
              }
              // The dependent parent id also donot needs to be updated, 
              // because it is gid.
              is_e_[index_s].data = oe.second;
              if (oe.second.parent_gid < inner_node_num) {
                is_e_[index_s].data.parent_gid = oldId2newId[oe.second.parent_gid];
              }
              index_s++;
            }
          }
        }
        /* inner_bound node */
        // vid_t index_b = ib_e_degree[i];
        vid_t index_b = ib_e_degree[new_id];
        ib_e_offset_[new_id] = &ib_e_[index_b];
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){ // edge
          auto oes = this->graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = this->supernode_out_mirror[ids_id];
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id
                && out_mirror.find(e.neighbor) == out_mirror.end()
                && in_mirror.find(u) == in_mirror.end()){
                  ib_e_[index_b] = e;
                  auto nbr = ib_e_[index_b].neighbor;
                  if (nbr.GetValue() < inner_node_num) {
                    ib_e_[index_b].neighbor = oldId2newId[nbr.GetValue()];
                  }
                  index_b++;
              }
            } else {
              if(ids_id != out_ids_id
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  ib_e_[index_b] = e;
                  auto nbr = ib_e_[index_b].neighbor;
                  if (nbr.GetValue() < inner_node_num) {
                    ib_e_[index_b].neighbor = oldId2newId[nbr.GetValue()];
                  }
                  index_b++;
              }
            }
          }
        }
        if (NodeType::SingleNode == type) { // edge
          auto oes = this->graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            ib_e_[index_b] = e;
            auto nbr = ib_e_[index_b].neighbor;
            if (nbr.GetValue() < inner_node_num) {
              ib_e_[index_b].neighbor = oldId2newId[nbr.GetValue()];
            }
            index_b++;
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      {
        parallel_for (vid_t i = 0; i < inner_node_num; ++i) {
          std::sort(is_e_offset_[i], is_e_offset_[i + 1],
                  [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                    return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                  });
        }
      }
      
      // just count mirror for expr and count skeleton
      if (FLAGS_count_skeleton) {
        LOG(INFO) << "\nopen cout skeleton:";
        size_t bound_node_out_edge_num = 0; 
        size_t skeleton_edge_num = 0; // shortcut + all out_edge
        size_t skeleton_node_num = 0; 
        size_t mirror_node_num = 0;
        size_t local_edges_num = graph_->GetEdgeNum() / 2;

        for (vid_t i = node_range[1]; i < node_range[2]; i++) {
          vertex_t u(i);
          adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
          bound_node_out_edge_num += oes.Size();
        }
        for (vid_t i = node_range[3]; i < node_range[4]; i++) {
          vertex_t u(i);
          adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
          bound_node_out_edge_num += oes.Size();
        }
        LOG(INFO) << " for mirror-master expr:";
        LOG(INFO) << "#bound_node_out_edge_num: " << bound_node_out_edge_num;

        skeleton_edge_num += source_e_num;
        skeleton_edge_num += bound_e_num;
        LOG(INFO) << "#skeleton_edge_num: " << skeleton_edge_num;
        LOG(INFO) << "  source_e_num: " << source_e_num;
        LOG(INFO) << "  bound_e_num: " << bound_e_num;

        skeleton_node_num = this->old_node_num - all_nodes[4].size();
        LOG(INFO) << "#skeleton_node_num: " << skeleton_node_num;
        LOG(INFO) << "#local_all_edges_num=" << local_edges_num;
        LOG(INFO) << "#new_cmp_rate=" 
                  << (local_edges_num-skeleton_edge_num)*1.0/local_edges_num;
      }
    }

    void sketch2csr(vid_t inner_node_num,
                            std::vector<char>& node_type,
                            std::vector<std::vector<vertex_t>>& all_nodes,
                            Array<nbr_index_t, Allocator<nbr_index_t>>& is_e_,
                            Array<nbr_index_t*, Allocator<nbr_index_t*>>& is_e_offset_,
                            Array<nbr_t, Allocator<nbr_t>>& ib_e_,
                            Array<nbr_t*, Allocator<nbr_t*>>& ib_e_offset_
                            ) {
      all_nodes.clear();
      all_nodes.resize(5);
      for(vid_t i = 0; i < inner_node_num; i++) {
          all_nodes[node_type[i]].emplace_back(vertex_t(i));
      }
      
      /* source to in_bound_node */
      is_e_.clear();
      is_e_offset_.clear();
      ib_e_.clear();
      ib_e_offset_.clear();
      size_t source_e_num = 0;
      size_t bound_e_num = 0;
      std::vector<size_t> is_e_degree(inner_node_num+1, 0);
      std::vector<size_t> ib_e_degree(inner_node_num+1, 0);

      parallel_for(vid_t i = 0; i < inner_node_num; i++) {
        vertex_t u(i);
        char type = node_type[i];
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){ // index
          for(auto mp : this->shortcuts[i]) {
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            is_e_degree[i+1] += spnode.bound_delta.size(); // Note: Accumulation is used here.
          }
        }
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){ // edge // else if
          auto oes = this->graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = this->supernode_out_mirror[ids_id];
          size_t temp_cnt = 0;
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  // bound_e_num += 1;
                  temp_cnt += 1;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  // bound_e_num += 1;
                  temp_cnt += 1;
              }
            }
          }
          // ib_e_degree[i+1] += temp_cnt;
          ib_e_degree[i+1] += temp_cnt;
        } else if (NodeType::SingleNode == type) { // edge
          auto oes = this->graph_->GetOutgoingAdjList(u);
          size_t temp_cnt = oes.Size();
          // ib_e_degree[i+1] += temp_cnt;
          ib_e_degree[i+1] = temp_cnt;
        }
      }

      /* get index start */
      for(vid_t i = 1; i <= inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
      }
      bound_e_num = ib_e_degree[inner_node_num];
      source_e_num = is_e_degree[inner_node_num];

      is_e_.resize(source_e_num);
      ib_e_.resize(bound_e_num);
      is_e_offset_.resize(inner_node_num+1);
      ib_e_offset_.resize(inner_node_num+1);

      /* build index/edge */
      parallel_for(vid_t i = 0; i < inner_node_num; i++) {
        vertex_t u(i);
        /* source node */
        vid_t new_id = i;
        vid_t index_s = is_e_degree[new_id];
        is_e_offset_[new_id] = &is_e_[index_s];
        char type = node_type[i];
        // LOG(INFO) << "i=" << i << " oid=" << this->vid2Oid(new_id) << " type=" << int(type);
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){ // index
          for(auto mp : this->shortcuts[i]) {
            // LOG(INFO) << " mp.second=" << mp.second;
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            for(auto& oe : spnode.bound_delta){
              is_e_[index_s].neighbor = oe.first;
              is_e_[index_s].data = oe.second;
              index_s++;
            }
          }
        }
        /* inner_bound node */
        // vid_t index_b = ib_e_degree[i];
        vid_t index_b = ib_e_degree[new_id];
        ib_e_offset_[new_id] = &ib_e_[index_b];
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){ // edge
          auto oes = this->graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = this->supernode_out_mirror[ids_id];
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id
                && out_mirror.find(e.neighbor) == out_mirror.end()
                && in_mirror.find(u) == in_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            } else {
              if(ids_id != out_ids_id
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            }
          }
        }
        if (NodeType::SingleNode == type) { // edge
          auto oes = this->graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            ib_e_[index_b] = e;
            index_b++;
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      {
        parallel_for (vid_t i = 0; i < inner_node_num; ++i) {
          std::sort(is_e_offset_[i], is_e_offset_[i + 1],
                  [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                    return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                  });
        }
      }
      // just count mirror for expr and count skeleton
      if (FLAGS_count_skeleton) {
        LOG(INFO) << "==================COUNT SKELETON========================";
        size_t bound_node_out_edge_num = 0;
        size_t skeleton_edge_num = 0; 
        size_t skeleton_node_num = 0;
        size_t mirror_node_num = 0;
        size_t local_edges_num = graph_->GetEdgeNum() / 2;

        for (vid_t i = 0; i < inner_node_num; i++) {
          if (node_type[i] == NodeType::OnlyOutNode 
              || node_type[i] == NodeType::BothOutInNode) {
            vertex_t u(i);
            adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
            bound_node_out_edge_num += oes.Size();
          }
        }
        LOG(INFO) << " for mirror-master expr:";
        LOG(INFO) << "#bound_node_out_edge_num: " << bound_node_out_edge_num;

        skeleton_edge_num += source_e_num;
        skeleton_edge_num += bound_e_num;
        LOG(INFO) << "#skeleton_edge_num: " << skeleton_edge_num;
        LOG(INFO) << "  source_e_num: " << source_e_num;
        LOG(INFO) << "  bound_e_num: " << bound_e_num;

        skeleton_node_num = this->old_node_num - all_nodes[4].size();
        LOG(INFO) << "#skeleton_node_num: " << skeleton_node_num;
        LOG(INFO) << "#local_all_edges_num=" << local_edges_num;
        LOG(INFO) << "#new_cmp_rate=" 
                  << (local_edges_num-skeleton_edge_num)*1.0/local_edges_num;
        LOG(INFO) << "=========================END============================";
      }
    }

    void inc_trav_compress_mirror(
            std::vector<std::pair<vid_t, vid_t>>& deleted_edges, 
            std::vector<std::pair<vid_t, vid_t>>& added_edges,
            const std::shared_ptr<fragment_t>& new_graph){
        size_t old_supernodes_num = this->supernodes_num;
        fid_t fid = this->graph_->fid();
        auto vm_ptr = this->graph_->vm_ptr();
        std::unordered_set<vid_t> temp_update_cluster_ids;

        for(auto& pair : deleted_edges) {
            auto u_gid = pair.first;
            auto v_gid = pair.second;
            fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                  v_fid = vm_ptr->GetFidFromGid(v_gid);
            vertex_t u;
            CHECK(this->graph_->Gid2Vertex(u_gid, u));
            vertex_t v;
            CHECK(this->graph_->Gid2Vertex(v_gid, v));
            if(u_fid == fid && this->Fc[u] != this->FC_default_value){
                // LOG(INFO) << " u_id=" << this->v2Oid(u); 
                temp_update_cluster_ids.insert(this->id2spids[u]);
                for (auto spid : this->vid2in_mirror_cluster_ids[u.GetValue()]) {
                    // LOG(INFO) << " u_id-mirror spid=" << spid; 
                    temp_update_cluster_ids.insert(spid);
                }
                // reset_edges.template emplace_back(u.GetValue(), v.GetValue());
            }
            if(v_fid == fid && this->Fc[v] != this->FC_default_value){
                temp_update_cluster_ids.insert(this->id2spids[v]);
            }
        }
        for(auto& pair : added_edges) {
            auto u_gid = pair.first;
            auto v_gid = pair.second;
            fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                  v_fid = vm_ptr->GetFidFromGid(v_gid);
            // u -> v
            // LOG(INFO) << u_gid << "->" << v_gid;
            vertex_t u;
            CHECK(this->graph_->Gid2Vertex(u_gid, u));
            vertex_t v;
            CHECK(this->graph_->Gid2Vertex(v_gid, v));
            if(u_fid == fid && this->Fc[u] != this->FC_default_value){
                temp_update_cluster_ids.insert(this->id2spids[u]);
                for (auto spid : this->vid2in_mirror_cluster_ids[u.GetValue()]) {
                    temp_update_cluster_ids.insert(spid);
                }
            }
            if(v_fid == fid && this->Fc[v] != this->FC_default_value){
                temp_update_cluster_ids.insert(this->id2spids[v]);
            }
        }

        this->update_cluster_ids.assign(temp_update_cluster_ids.begin(), 
                                       temp_update_cluster_ids.end());

        for (auto ids_id : this->update_cluster_ids) {
            supernode_t& spnode = this->supernodes[ids_id];
            std::vector<vertex_t> &node_set = this->supernode_ids[ids_id];
            std::vector<vertex_t> &old_S = this->supernode_source[ids_id];
            std::unordered_set<vertex_t> &in_mirror = 
                                            this->supernode_in_mirror[ids_id];
            std::set<vertex_t> S;
            for(auto v : node_set){
                const auto& oes = new_graph->GetIncomingAdjList(v); // get new adj
                for(auto& e : oes){
                    if(this->id2spids[e.neighbor] != ids_id 
                        && in_mirror.find(e.neighbor) == in_mirror.end()){
                        S.insert(v);
                        break;
                    }
                }
            }
            std::vector<vid_t> delete_spid;
            for (auto s : old_S) {
                CHECK(this->Fc[s] >= 0);
                if (S.find(s) == S.end()) {
                    vid_t spid = this->Fc_map[s];
                    this->supernodes[spid].clear();
                    delete_spid.emplace_back(spid);
                    this->Fc[s] = -(ids_id+1);
                    this->Fc_map[s] = this->ID_default_value;
                    // this->shortcuts[s.GetValue()].erase(spid);
                    this->shortcuts[s.GetValue()].erase(ids_id);
                }
            }
            int delete_id = delete_spid.size() - 1;
            for(auto src : S){
                if (this->Fc_map[src] != this->ID_default_value) {
                    // LOG(INFO) << " inc: spid=" << this->Fc_map[src]
                    //           << " source=" << this->v2Oid(src);
                    continue;
                }
                vid_t supernode_id;
                if (delete_id >= 0) {
                    supernode_id = delete_spid[delete_id];
                    delete_id--;
                } else {
                    supernode_id = __sync_fetch_and_add(&this->supernodes_num, 1);
                }
                this->Fc[src] = ids_id;
                // LOG(INFO) << " build a new sp=" << supernode_id
                //           << " source=" << this->v2Oid(src);
                /* build a supernode */
                this->Fc_map[src] = supernode_id;
                {
                    std::unique_lock<std::mutex> lk(this->supernode_ids_mux_);
                    this->shortcuts[src.GetValue()][ids_id] = supernode_id;
                }
                this->supernodes[supernode_id].status = false;
                this->supernodes[supernode_id].id = src;
                this->supernodes[supernode_id].ids = ids_id; // root_id
            }
            for (int i = 0; i <= delete_id; i++) {
              this->delete_supernode(delete_spid[i]);
            }
            old_S.clear();
            old_S.insert(old_S.begin(), S.begin(), S.end());
        }
    }

    void inc_compress_mirror(std::vector<std::pair<vid_t, vid_t>>& deleted_edges, 
            std::vector<std::pair<vid_t, vid_t>>& added_edges,
            const std::shared_ptr<fragment_t>& new_graph){
      fid_t fid = graph_->fid();
      auto vm_ptr = graph_->vm_ptr();
      update_cluster_ids.clear();
      update_source_id.clear();
      std::unordered_set<vid_t> temp_update_cluster_ids;
      std::unordered_set<vid_t> temp_update_source_id;
      size_t old_supernodes_num = this->supernodes_num;
      vid_t add_num = 0; // just count num
      vid_t del_num = 0; // just count num

      double del_time_1 = 0;
      double del_time_2 = 0;

      std::vector<vid_t> delete_spid;

      for(auto& pair : deleted_edges) {
        auto u_gid = pair.first;
        auto v_gid = pair.second;
        fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
              v_fid = vm_ptr->GetFidFromGid(v_gid);
        // u -> v
        // LOG(INFO) << u_gid << "->" << v_gid;
        vertex_t u;
        CHECK(graph_->Gid2Vertex(u_gid, u));
        vertex_t v;
        CHECK(graph_->Gid2Vertex(v_gid, v));
        if(u_fid == fid && Fc[u] != FC_default_value){
          vid_t src_id = this->id2spids[u];
          temp_update_cluster_ids.insert(src_id);
          // all source in src_id, include in-mirror
          for(auto source : this->supernode_source[src_id]){
            temp_update_source_id.insert(source.GetValue());
          }
          // cluster u's in-mirror
          for(auto mid : this->cluster_in_mirror_ids[src_id]){
            temp_update_source_id.insert(mid.GetValue());
          }
          // node u's in-mirror
          for (auto mid : this->vid2in_mirror_mids[u.GetValue()]) {
            temp_update_source_id.insert(mid);
            temp_update_cluster_ids.insert(this->id2spids[vertex_t(mid)]);
          }
        }
        if(v_fid == fid && Fc[v] != FC_default_value && Fc[v] >= 0){ // FC_default_value > 0
          // vid_t del_id = Fc_map[Fc[v][0]];
          vid_t ids_id = this->id2spids[v];
          temp_update_cluster_ids.insert(ids_id);
          // if(ids_id != this->id2spids[u] && src.size() > 1){
          if(ids_id != this->id2spids[u]){
            CHECK(Fc[v] >= 0);
            const auto& ies = new_graph->GetIncomingAdjList(v);
            bool hava_out_inadj = false;
            std::unordered_set<vertex_t> &in_mirror = 
                                            this->supernode_in_mirror[ids_id];
            for (auto& e : ies) {
              auto& nb = e.neighbor;
              if(nb != u && ids_id != this->id2spids[nb]
                  && in_mirror.find(e.neighbor) == in_mirror.end()){
                hava_out_inadj = true;
                break;
              }
            }
           
            if(hava_out_inadj == false){
              {
                CHECK(remove_array(supernode_source[ids_id], v));
              }
              vid_t del_spid = this->Fc_map[v];
              this->supernodes[del_spid].clear();
              delete_spid.emplace_back(del_spid);
              this->Fc[v] = -(ids_id+1);
              this->Fc_map[v] = this->ID_default_value;
              this->shortcuts[v.GetValue()].erase(ids_id);
              del_num++;
            }
          }
        }
      }
      int del_index = delete_spid.size() - 1;
      for(auto& pair : added_edges){
        auto u_gid = pair.first;
        auto v_gid = pair.second;
        fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
              v_fid = vm_ptr->GetFidFromGid(v_gid);
        // u -> v
        // LOG(INFO) << u_gid << "->" << v_gid;
        vertex_t u;
        CHECK(graph_->Gid2Vertex(u_gid, u));
        vertex_t v;
        CHECK(graph_->Gid2Vertex(v_gid, v));
        if(u_fid == fid && Fc[u] != FC_default_value){
          vid_t src_id = this->id2spids[u];
          temp_update_cluster_ids.insert(src_id);
          // all source in src_id, include in-mirror
          for(auto source : this->supernode_source[src_id]){
            temp_update_source_id.insert(source.GetValue());
          }
          // cluster u's in-mirror
          for(auto mid : this->cluster_in_mirror_ids[src_id]){
            temp_update_source_id.insert(mid.GetValue());
          }
          // node u's in-mirror
          for (auto mid : this->vid2in_mirror_mids[u.GetValue()]) {
            temp_update_source_id.insert(mid);
            temp_update_cluster_ids.insert(this->id2spids[vertex_t(mid)]);
          }
        }
        if(v_fid == fid && Fc[v] != FC_default_value){
          vid_t ids_id = this->id2spids[v];
          temp_update_cluster_ids.insert(ids_id);
          std::unordered_set<vertex_t> &in_mirror = 
                                            this->supernode_in_mirror[ids_id];
          if(Fc[v] < 0 && ids_id != this->id2spids[u] 
              && in_mirror.find(u) == in_mirror.end()){
            Fc[v] = ids_id;
            this->supernode_source[ids_id].emplace_back(v);
            // build a new spnode idnex
            vid_t supernode_id = 0;
            if (del_index >= 0) {
              supernode_id = delete_spid[del_index];
              del_index--;
            } else {
              supernode_id = supernodes_num;
              supernodes_num++;
            }
            this->Fc_map[v] = supernode_id;
            {
              std::unique_lock<std::mutex> lk(this->supernode_ids_mux_);
              this->shortcuts[v.GetValue()][ids_id] = supernode_id;
            }
            this->supernodes[supernode_id].id = v;
            this->supernodes[supernode_id].ids = ids_id;
            this->supernodes[supernode_id].status = false;

            temp_update_source_id.insert(v.GetValue());
            add_num++;
          }
        }
      }

      for (int i = 0; i <= del_index; i++) {
        this->delete_supernode(delete_spid[i]);
      }
      this->update_cluster_ids.assign(temp_update_cluster_ids.begin(), 
                                       temp_update_cluster_ids.end());
      this->update_source_id.assign(temp_update_source_id.begin(), 
                                     temp_update_source_id.end());
    }

    void get_reverse_shortcuts() {
      this->reverse_shortcuts.clear();
      this->reverse_shortcuts.resize(this->all_node_num);
      parallel_for(vid_t i = 0; i < this->GetClusterSize(); i++) {
        auto& entry_node_set = this->supernode_source[i];
        for (auto v : entry_node_set) {
          vid_t spid = this->Fc_map[v];
          supernode_t &spnode = this->supernodes[spid];
          for (auto e : spnode.inner_delta) {
            // this->reverse_shortcuts[e.first.GetValue()][v.GetValue()] = e;
            this->reverse_shortcuts[e.first.GetValue()][v.GetValue()] = e.second;
          }
          for (auto e : spnode.bound_delta) {
            this->reverse_shortcuts[e.first.GetValue()][v.GetValue()] = e.second;
          }
        }
        auto& entry_mirror_node_set = this->cluster_in_mirror_ids[i];
        for (auto v : entry_mirror_node_set) {
          vid_t spid = this->Fc_map[v];
          supernode_t &spnode = this->supernodes[spid];
          vid_t master_id = this->mirrorid2vid[v].GetValue();
          for (auto e : spnode.inner_delta) {
            this->reverse_shortcuts[e.first.GetValue()][master_id] = e.second;
          }
          for (auto e : spnode.bound_delta) {
            this->reverse_shortcuts[e.first.GetValue()][master_id] = e.second;
          }
        }
      }
    }
    
    
    vid_t GetClusterSize() {
      return this->cluster_ids.size();
    }

    void clean_no_used(VertexArray<value_t, vid_t>& values_, value_t default_value) {
      vid_t delete_cluster_num = 0;
      for (auto cid : this->update_cluster_ids) {
        value_t diff = 0;
        for (auto v : this->cluster_ids[cid]) {
          diff += fabs(values_[v] - default_value);
        }
        if (diff == 0) {
          this->delete_cluster(cid);
          delete_cluster_num++;
        }
      }
    }

    ~CompressorBase(){
        delete[] supernodes;
    }

public:
    std::shared_ptr<APP_T>& app_;
    std::shared_ptr<fragment_t>& graph_;
    Communicator communicator_;
    CommSpec comm_spec_;
    vid_t supernodes_num=0;
    vid_t MAX_NODE_NUM=FLAGS_max_node_num;
    vid_t MIN_NODE_NUM=FLAGS_min_node_num;
    VertexArray<fc_t, vid_t> Fc; // fc[v]= index of cluster_ids, Fc[v] = ids_id if v is a source node. V doest not include mirror node.
    VertexArray<vid_t, vid_t> Fc_map; // fc[v]= index of supernodes and v is a source node, inclue mirror node, Fc_map[v] = supernode_id;
    supernode_t *supernodes; // max_len = nodes_num
    const vid_t FC_default_value = std::numeric_limits<fc_t>::max(); 
    const vid_t ID_default_value = std::numeric_limits<vid_t>::max(); // max id
    // std::vector<vid_t> supernode_ids;
    std::vector<std::vector<vertex_t>> supernode_ids;  // the set of vertices contained in each supernode
    std::vector<std::vector<vertex_t>> cluster_ids;  // the set of vertices contained in each supernode include mirrorid
    std::vector<std::vector<vertex_t>> supernode_source;  // the set of source vertices of each supernode
    std::vector<std::vector<vertex_t>> cluster_in_mirror_ids;  // the set of mirrorid of in_mirror contained in each supernode include mirror
    std::vector<std::vector<vertex_t>> cluster_out_mirror_ids;  // the set of mirrorid of out_mirror contained in each supernode include mirror
    std::vector<std::unordered_set<vertex_t>> supernode_in_mirror;  // the set of vid of in_mirror vertices of each supernode
    std::vector<std::unordered_set<vertex_t>> supernode_out_mirror;  // the set of vid of out_mirror vertices of each supernode
    std::vector<std::vector<vid_t>> vid2in_mirror_cluster_ids;  // the set of cluster id of each in-mirror vertex
    std::vector<std::vector<vid_t>> vid2in_mirror_mids;  // the set of spid of each in-mirror vertex
    std::vector<std::vector<vid_t>> vid2out_mirror_mids;  // the set of spid of each out-mirror vertex
    // std::vector<std::vector<vid_t>> out_mirror2spids;  // the set of spids of each mirror vertex
    // std::vector<std::vector<vertex_t>> supernode_bound_ids;  // the set of bound vertices of each supernode
    std::vector<short int> supernode_out_bound;  // if is out_bound_node
    VertexArray<vid_t, vid_t> id2spids;                // record the cluster id of each node(include mirror node), note that it is not an index structure id
    std::unordered_set<vid_t> recalculate_spnode_ids;  // the set of recalculated super vertices
    std::unordered_set<vid_t> inccalculate_spnode_ids; // the set of inc-recalculated super vertices
    std::mutex supernode_ids_mux_;
    std::mutex shortcuts_mux_; // for inc_compress
    // std::vector<idx_t> graph_part;  // metis result
    std::vector<std::unordered_map<vid_t, vid_t>> shortcuts; // record shortcuts for each entry vertice (including Mirror vertices)
    std::vector<std::unordered_map<vid_t, delta_t>> reverse_shortcuts; // record re-shortcuts for each entry vertice (including Mirror vertices)
    std::unordered_map<vertex_t, vertex_t> mirrorid2vid; // record the mapping between mirror id and vertex id
    // std::unordered_map<vertex_t, vertex_t> vid2mirrorid; // record the mapping between mirror id and vertex id
    vid_t old_node_num;
    vid_t all_node_num;
    std::vector<std::vector<nbr_t>> subgraph;
    std::vector<std::vector<nbr_t>> subgraph_old;
    std::vector<vid_t> update_cluster_ids; // the set of ids_id of updated cluster
    std::vector<vid_t> update_source_id; // the set of spid of updated supernode
    /* source to in_bound_node */
    Array<nbr_index_t, Allocator<nbr_index_t>> is_e_;
    Array<nbr_index_t*, Allocator<nbr_index_t*>> is_e_offset_;
    /* master_source to mirror cluster */
    Array<nbr_index_t, Allocator<nbr_index_t>> im_e_; // in-mirror
    Array<nbr_index_t*, Allocator<nbr_index_t*>> im_e_offset_;
    Array<nbr_index_t, Allocator<nbr_index_t>> om_e_; // out-mirror
    Array<nbr_index_t*, Allocator<nbr_index_t*>> om_e_offset_;
    Array<nbr_index_t, Allocator<nbr_index_t>> oim_e_; // out-mirror
    Array<nbr_index_t*, Allocator<nbr_index_t*>> oim_e_offset_;
    /* in_bound_node to out_bound_node */
    Array<nbr_t, Allocator<nbr_t>> ib_e_;
    Array<nbr_t*, Allocator<nbr_t*>> ib_e_offset_;
    Array<nbr_t, Allocator<nbr_t>> sync_e_; // Synchronized edges between master-mirror without weights.
    Array<nbr_t*, Allocator<nbr_t*>> sync_e_offset_;
    std::vector<vertex_t> all_out_mirror; // all out mirror
    std::vector<vid_t> indegree; // degree of each cluster
};

}  // namespace grape
#endif  // GRAPE_FRAGMENT_COMPRESSOR_BASE_H_