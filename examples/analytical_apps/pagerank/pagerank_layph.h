
#ifndef ANALYTICAL_APPS_PAGERANK_PAGERANK_LAYPH_H_
#define ANALYTICAL_APPS_PAGERANK_PAGERANK_LAYPH_H_

#include "grape/app/layph_app_base.h"
#include "grape/fragment/immutable_edgecut_fragment.h"
#include "grape/parallel/parallel.h"
#include <cilk/reducer_opadd.h>

namespace grape {

template <typename FRAG_T, typename VALUE_T>
class PageRankLayph : public IterateKernel<FRAG_T, VALUE_T> {
 public:
  using fragment_t = FRAG_T;
  using vid_t = typename fragment_t::vid_t;
  using vertex_t = typename fragment_t::vertex_t;
  using value_t = VALUE_T;
  using adj_list_t = typename fragment_t::adj_list_t;
  using edata_t = typename fragment_t::edata_t;
  using adj_list_index_t = AdjList<vid_t, value_t>;
  const value_t d = 0.85f;
  long long f_cnt = 0;
  long long u_cnt = 0;

  void init_c(vertex_t v, value_t& delta, const FRAG_T& frag) override {
    delta = (1-d) / frag.GetTotalVerticesNum();
    // delta = 1-d;
  }

  void init_v(const vertex_t v, value_t& value) override { value = 0.0f; }

  bool accumulate_atomic(value_t& a, value_t b) override {
    atomic_add(a, b);
    return true;
  }

  bool accumulate(value_t& a, value_t b) override {
    a += b;
    return true;
  }

  void priority(value_t& pri, const value_t& value,
                const value_t& delta) override {
    pri = delta;
  }

  inline void g_function(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, const value_t& delta,
                         const adj_list_t& oes) override {
    if (delta != default_v()) {
      auto out_degree = oes.Size();
      value_t outv;

      if (out_degree > 0) {
        outv = delta * d / out_degree;
        #ifdef COUNT_ACTIVE_EDGE_NUM
          atomic_add(this->f_send_delta_num, (long long)out_degree);
        #endif
        auto it = oes.begin();

        granular_for(j, 0, out_degree, (out_degree > 1024), {
          auto& e = *(it + j);
          this->accumulate_to_delta(const_cast<Vertex<vid_t>&>(e.neighbor), outv);
        })
      } 
    }
  }

  /* in_bound node send message */
  inline void g_function(const FRAG_T& frag, const vertex_t v,
                          const value_t& value, const value_t& delta,
                          const adj_list_t& old_oes, const adj_list_t& to_send_oes) override {
    if (delta != default_v()) {
      auto old_out_degree = old_oes.Size(); // real out degree
      auto out_degree = to_send_oes.Size();
      value_t outv;

      if (out_degree > 0) {
        outv = delta * d / old_out_degree;
        #ifdef COUNT_ACTIVE_EDGE_NUM
          atomic_add(this->f_send_delta_num, (long long)out_degree);
        #endif
        auto it = to_send_oes.begin();

        granular_for(j, 0, out_degree, (out_degree > 1024), {
          auto& e = *(it + j);
          this->accumulate_to_delta(const_cast<Vertex<vid_t>&>(e.neighbor), outv);
        })
      } 
    }
  }

  /* source send message */
  inline void g_index_function(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, const value_t& delta,
                         const adj_list_index_t& oes, VertexArray<value_t, vid_t>& bound_node_values) override{
    if (delta != default_v()) {
      auto out_degree = oes.Size();
      value_t outv;

      if (out_degree > 0) {
        auto it = oes.begin();
        #ifdef COUNT_ACTIVE_EDGE_NUM
          atomic_add(this->f_send_delta_num, (long long)out_degree);
        #endif
        granular_for(j, 0, out_degree, (out_degree > 1024), {
          auto& e = *(it + j);
          this->accumulate_atomic(bound_node_values[e.neighbor], e.data * delta);
        })
      } 
    }
  }

  /* 
    F operation：
      Returns the message that a single edge needs to send.
      Before calling this function, you need to judge the validity of the call.
  */
  inline void g_function(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, const value_t& delta,
                         const adj_list_t& oes, const Nbr<vid_t, edata_t>& oe, value_t& outv) {
    outv = delta * d / oes.Size();
  }

  void init_c(vertex_t v, value_t& delta, const FRAG_T& frag, const vertex_t source) override {
    if(v == source){
      delta = 1;
    } 
    else{
      delta = 0;
    }
  }

  void g_revfunction(value_t& value, value_t& rt_value){
      rt_value = value;
  }

  inline void g_index_func_delta(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, const value_t& delta,
                         const std::vector<std::pair<vertex_t, value_t>>& oes) override {
    if (delta != default_v()) {
      auto out_degree = oes.size();

      if (out_degree > 0) {
        #ifdef COUNT_ACTIVE_EDGE_NUM
          atomic_add(this->f_send_delta_num, (long long)out_degree);
        #endif
        granular_for(j, 0, out_degree, (out_degree > 1024), {
          auto& e = oes[j];
          this->accumulate_to_delta(const_cast<Vertex<vid_t>&>(e.first), delta * e.second);
        })
      }  
    }
  }

  inline void g_index_func_value(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, const value_t& delta,
                         const std::vector<std::pair<vertex_t, value_t>>& oes) override {
    if (delta != default_v()) {
      auto out_degree = oes.size();

      if (out_degree > 0) {
        #ifdef COUNT_ACTIVE_EDGE_NUM
          atomic_add(this->f_send_delta_num, (long long)out_degree);
        #endif
        granular_for(j, 0, out_degree, (out_degree > 1024), {
          auto& e = oes[j];
          this->accumulate_to_value(const_cast<Vertex<vid_t>&>(e.first), delta * e.second);
        })
      }  
    }
  }
  
  //===============================PULL====================================
  void get_last_delta(const FRAG_T& frag, const std::vector<std::vector<vertex_t>>& all_nodes) override {
    for(int i = 0; i < 4; i++){
      const std::vector<vertex_t>& nodes = all_nodes[i];
      vid_t node_size = nodes.size();
      parallel_for(vid_t j = 0; j < node_size; j++){
        // LOG(INFO) << "i=" << i << "j=" << j;
        vertex_t u = nodes[j];
        auto& delta = this->deltas_[u];
        auto& index_value = this->index_values_[u];
        this->last_deltas_[u] = delta * d / frag.GetLocalOutDegree(u);
        this->last_index_values_[u] = index_value;
        delta = this->default_v();
        index_value = this->default_v();
        // LOG(INFO) << "----";
      }
    }
  }

  void get_last_delta(const FRAG_T& frag) override {
    auto inner_vertices = frag.InnerVertices();
    auto size = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();
    parallel_for(vid_t j = 0; j < size; j++){
      vertex_t v(j);
      // auto oes = frag.GetOutgoingAdjList(v);
      // this->last_deltas_[v] = this->deltas_[v] * d  / oes.Size();
      auto& delta = this->deltas_[v];
      this->last_deltas_[v] = delta * d  / frag.GetLocalOutDegree(v);
      delta = this->default_v();
    }
  }
 
  inline void g_function_pull(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, value_t& delta,
                         const adj_list_t& ies) override {
    auto in_degree = ies.Size();
    if (in_degree > 0) {
        for(auto ie : ies){
          accumulate(delta, this->last_deltas_[ie.neighbor]);
        }
    }
  }

  inline void g_function_pull_by_index(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, value_t& delta,
                         const adj_list_index_t& iindexes) override {
    auto in_degree = iindexes.Size();
    if (in_degree > 0) {
        for(auto ie : iindexes){
          accumulate(delta, this->last_index_values_[ie.neighbor]*ie.data);
        }
    }
  }

  inline void g_function_pull_spnode_datas_by_index(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, value_t& delta,
                         const adj_list_index_t& iindexes, VertexArray<value_t, vid_t>& spnode_datas) override {
    auto in_degree = iindexes.Size();
    if (in_degree > 0) {
        for(auto ie : iindexes){
          accumulate(delta, spnode_datas[ie.neighbor]*ie.data);
        }
    }
  }
  //========================================================================

  value_t default_v() override { return 0; }

  value_t min_delta() override { return 0; }
};

}  // namespace grape

#endif  // ANALYTICAL_APPS_PAGERANK_PAGERANK_LAYPH_H_
