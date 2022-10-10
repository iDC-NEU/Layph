/*
    Just for batch sssp.
*/
#ifndef ANALYTICAL_APPS_SSSPBATCH_SSSPBATCH_LAYPH_H_
#define ANALYTICAL_APPS_SSSPBATCH_SSSPBATCH_LAYPH_H_

#include "grape/app/layph_app_base.h"
#include "grape/fragment/immutable_edgecut_fragment.h"
#include "grape/parallel/parallel.h"
#include <cilk/reducer_opadd.h>
#include "flags.h"

namespace grape {

template <typename FRAG_T, typename VALUE_T>
class SSSPBatchLayph : public IterateKernel<FRAG_T, VALUE_T> {
 public:
  using fragment_t = FRAG_T;
  using vid_t = typename fragment_t::vid_t;
  using vertex_t = typename fragment_t::vertex_t;
  using value_t = VALUE_T;
  using adj_list_t = typename fragment_t::adj_list_t;
  using edata_t = typename fragment_t::edata_t;
  using adj_list_index_t = AdjList<vid_t, value_t>;
  const value_t d = 0.85f;

  void init_c(vertex_t v, value_t& delta, const FRAG_T& frag) override {
    vertex_t source;
    bool native_source =
        frag.GetInnerVertex(FLAGS_sssp_source, source);
    delta = default_v();

    if (native_source && source == v) {
      delta = 0;
    }
  }

  void init_v(const vertex_t v, value_t& value) override { value = default_v(); }

  bool accumulate_atomic(value_t& a, value_t b) override {
    return atomic_min(a, b);
  }

  bool accumulate(value_t& a, value_t b) override {
    if (a > b) {
        a = b;
        return true;
    }
    return false;
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

      if (out_degree > 0) {
        auto it = oes.begin();

        granular_for(j, 0, out_degree, (out_degree > 1024), {
          auto& e = *(it + j);
          value_t outv = delta + e.data;
          this->accumulate_to_delta(const_cast<Vertex<vid_t>&>(e.neighbor), outv);
        })
      } 
    }
  }

  inline void g_function(const FRAG_T& frag, const vertex_t& v,
                         const value_t& value, const value_t& delta,
                        //  const adj_list_t& oes,
               DenseVertexSet<vid_t>& modified) {
    auto oes = frag.GetOutgoingAdjList(v);
    auto out_degree = oes.Size();

      auto it = oes.begin();

      granular_for(j, 0, out_degree, (out_degree > 1024), {
        auto& e = *(it + j);
        auto v = e.neighbor;
        auto outv = delta + e.data;
        if (outv < this->deltas_[v]) {
            this->accumulate_to_delta(v, outv);
            modified.Insert(v);
        }
      })
  }

  /* in_bound node send message */
  inline void g_function(const FRAG_T& frag, const vertex_t v,
                          const value_t& value, const value_t& delta,
                          const adj_list_t& old_oes, const adj_list_t& to_send_oes) override {
    if (delta != default_v()) {
    //   auto old_out_degree = old_oes.Size(); // real out degree
      auto out_degree = to_send_oes.Size();

      if (out_degree > 0) {
        auto it = to_send_oes.begin();

        granular_for(j, 0, out_degree, (out_degree > 1024), {
          auto& e = *(it + j);
          value_t outv = delta + e.data;
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

      if (out_degree > 0) {
        auto it = oes.begin();
        granular_for(j, 0, out_degree, (out_degree > 1024), {
        // for(int j = 0; j < out_degree; j++){
          auto& e = *(it + j);
          this->accumulate_atomic(bound_node_values[e.neighbor], e.data + delta);
        // }
        })
      } 
    }
  }

  inline void g_function(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, const value_t& delta,
                         const adj_list_t& oes, const Nbr<vid_t, edata_t>& oe, value_t& outv) override {
    if (delta != default_v()) {
      auto out_degree = oes.Size();
      if (out_degree > 0) {
        outv = delta + oe.data;
      }
      else {
        outv = default_v();
      }
    }
    else{
      outv = default_v();
    }
  }

  void init_c(vertex_t v, value_t& delta, const FRAG_T& frag, const vertex_t source) override {
    if(v == source){
      delta = 0;
    } 
    else{
      delta = default_v();
    }
  }

  void g_revfunction(value_t& value, value_t& rt_value){
      rt_value = value;
  }

  // Used for the interior of the supernode in the later stage of convergence
  inline void g_index_func_delta(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, const value_t& delta,
                         const std::vector<std::pair<vertex_t, value_t>>& oes) override {
    if (delta != default_v()) {
      auto out_degree = oes.size();

      if (out_degree > 0) {
        granular_for(j, 0, out_degree, (out_degree > 1024), {
          // auto& e = *(it + j);
          auto& e = oes[j];
          this->accumulate_to_delta(const_cast<Vertex<vid_t>&>(e.first), delta + e.second);
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
        granular_for(j, 0, out_degree, (out_degree > 1024), {
          // auto& e = *(it + j);
          auto& e = oes[j];
          this->accumulate_to_value(const_cast<Vertex<vid_t>&>(e.first), delta + e.second);
        })
      }  
    }
  }
  
  //===============================PULL====================================
  void get_last_delta(const FRAG_T& frag, const std::vector<std::vector<vertex_t>>& all_nodes) override {
  }

  void get_last_delta(const FRAG_T& frag) override {
  }
 
  inline void g_function_pull(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, value_t& delta,
                         const adj_list_t& ies) override {
  }

  inline void g_function_pull_by_index(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, value_t& delta,
                         const adj_list_index_t& iindexes) override {

  }

  inline void g_function_pull_spnode_datas_by_index(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, value_t& delta,
                         const adj_list_index_t& iindexes, VertexArray<value_t, vid_t>& spnode_datas) override {

  }
  //========================================================================

  value_t default_v() override { return std::numeric_limits<value_t>::max(); }

  value_t min_delta() override { return 0; }
};

}  // namespace grape

#endif  // ANALYTICAL_APPS_SSSPBATCH_SSSPBATCH_LAYPH_H_
