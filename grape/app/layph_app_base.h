
#ifndef LIBGRAPE_LITE_GRAPE_APP_LAYPH_APP_BASE_H_
#define LIBGRAPE_LITE_GRAPE_APP_LAYPH_APP_BASE_H_

#include "grape/types.h"
#include "grape/utils/vertex_array.h"

namespace grape {
template <typename FRAG_T, typename VALUE_T>
class IterateKernel {
 public:
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using value_t = VALUE_T;
  using delta_t = VALUE_T;
  using vertex_t = typename fragment_t::vertex_t;
  using adj_list_t = typename fragment_t::adj_list_t;
  using edata_t = typename fragment_t::edata_t;
  using adj_list_index_t = AdjList<vid_t, value_t>;

  static constexpr bool need_split_edges = false;
  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kSyncOnOuterVertex;

  explicit IterateKernel() = default;

  virtual ~IterateKernel() = default;

  virtual void init_c(const vertex_t v, value_t& delta, const FRAG_T& frag) {}

  virtual void init_c(const FRAG_T& frag, const vertex_t v, value_t& delta,
                      DenseVertexSet<vid_t>& modified) {}

  virtual void init_v(const vertex_t v, value_t& value) = 0;

  // virtual void iterate_begin(const FRAG_T& frag) {}
  /* php */
  virtual void iterate_begin(FRAG_T& frag) {}
  virtual void rebuild_graph(FRAG_T& frag) {}

  virtual bool accumulate_atomic(value_t& a, value_t b) = 0;

  inline bool accumulate_to_delta(vertex_t& v, value_t val) {
    return accumulate_atomic(deltas_[v], val);
  }

  virtual void priority(value_t& pri, const value_t& value,
                        const value_t& delta) = 0;

  /* Normal edge sending */
  virtual void g_function(const FRAG_T& frag, const vertex_t v,
                          const value_t& value, const value_t& delta,
                          const adj_list_t& oes) {}

  virtual void g_function(const FRAG_T& frag, const vertex_t v,
                          const value_t& value, const value_t& delta,
                          const adj_list_t& oes,
                          DenseVertexSet<vid_t>& modified) {}

  /* Used for creating index within the supernode */
  // virtual void g_function(const FRAG_T& frag, const vertex_t v,
  void g_function(const FRAG_T& frag, const vertex_t v,
                          const value_t& value, const value_t& delta,
                          const adj_list_t& oes, const Nbr<vid_t, edata_t>& oe, value_t& outv) {}
  
  /* in_bound node sending */
  virtual void g_function(const FRAG_T& frag, const vertex_t v,
                          const value_t& value, const value_t& delta,
                          const adj_list_t& old_oes, const adj_list_t& to_send_oes) {}

  /* source sending */
  virtual inline void g_index_function(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, const value_t& delta,
                         const adj_list_index_t& oes, VertexArray<value_t, vid_t>& bound_node_values) {}

  virtual void init_c(const vertex_t v, value_t& delta, const FRAG_T& frag, const vertex_t source) {}

  virtual void g_revfunction(value_t& value, value_t& rt_value){}

  /* Used for the interior of the supernode in the later stage of convergence */
  virtual void g_index_func_delta(const FRAG_T& frag, const vertex_t v,
                          const value_t& value, const value_t& delta,
                          const std::vector<std::pair<vertex_t, value_t>>& oes) {}

  virtual void g_index_func_value(const FRAG_T& frag, const vertex_t v,
                          const value_t& value, const value_t& delta,
                          const std::vector<std::pair<vertex_t, value_t>>& oes) {}

  //============================PULL=======================================

  virtual void get_last_delta(const FRAG_T& frag, const std::vector<std::vector<vertex_t>>& all_nodes) {}

  virtual void get_last_delta(const FRAG_T& frag) {}

  virtual bool accumulate(value_t& a, value_t b) = 0;

  virtual void g_function_pull(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, value_t& delta,
                         const adj_list_t& ies){}

  virtual void g_function_pull_by_index(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, value_t& delta,
                         const adj_list_index_t& iindexes){}

  virtual void g_function_pull_spnode_datas_by_index(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, value_t& delta,
                         const adj_list_index_t& iindexes, VertexArray<value_t, vid_t>& spnode_datas){};

  //=======================================================================

  inline bool accumulate_to_value(vertex_t& v, value_t val) {
    return accumulate_atomic(values_[v], val);
  }

  virtual value_t default_v() = 0;

  virtual value_t min_delta() = 0;

  // if u sends msg to v, then v depends on u
  inline void add_delta_dependency(vid_t u_gid, vertex_t v) {
    delta_dep_[v] = u_gid;
  }

  inline void mark_value_dependency(vertex_t u) { val_dep_[u] = delta_dep_[u]; }

  inline vid_t delta_parent_gid(vertex_t v) { return delta_dep_[v]; }

  inline vid_t value_parent_gid(vertex_t v) { return val_dep_[v]; }

  std::vector<value_t> DumpResult() {
    std::vector<value_t> result;

    for (auto v : values_.GetVertexRange()) {
      result.push_back(values_[v]);
    }
    return result;
  }

 protected:
  void Init(const CommSpec& comm_spec, const FRAG_T& frag, bool dependent,
            bool data_driven = false) {
    auto vertices = frag.Vertices();
    auto inner_vertices = frag.InnerVertices();

    values_.Init(inner_vertices);
    deltas_.Init(vertices, default_v());
    degree.Init(vertices, default_v());
    last_deltas_.Init(vertices, default_v());
    last_index_values_.Init(vertices, default_v());
    index_values_.Init(vertices, default_v());

    curr_modified_.Init(vertices);
    next_modified_.Init(vertices);
    curr_modified_.Clear();
    next_modified_.Clear();
    if (data_driven) {

      for (auto v : inner_vertices) {
        value_t value;
        init_v(v, value);
        values_[v] = value;

        value_t delta;
        init_c(frag, v, delta, curr_modified_);
        deltas_[v] = delta;

        if (deltas_[v] != default_v()) {
          curr_modified_.Insert(v);
        }

        int EdgeNum = frag.GetOutgoingAdjList(v).Size();
        degree[v] = EdgeNum;
      }
    } else {
      for (auto v : inner_vertices) {
        value_t value;
        init_v(v, value);
        values_[v] = value;

        value_t delta;
        init_c(v, delta, frag);
        deltas_[v] = delta;

        if (deltas_[v] != default_v()) {
          curr_modified_.Insert(v);
        }

        int EdgeNum = frag.GetOutgoingAdjList(v).Size();
        degree[v] = EdgeNum;
      }
    }

    if (dependent) {
      val_dep_.Init(inner_vertices);
      delta_dep_.Init(vertices);

      for (auto v : inner_vertices) {
        val_dep_[v] = frag.Vertex2Gid(v);
      }

      for (auto v : vertices) {
        delta_dep_[v] = frag.Vertex2Gid(v);
      }
    }

    uint64_t memory = 0, global_mem;
    if (!FLAGS_efile_update.empty()) {
      memory += sizeof(vid_t) * val_dep_.size();
      memory += sizeof(vid_t) * delta_dep_.size();
    }
    memory += curr_modified_.Range().size() / 64;
    memory += next_modified_.Range().size() / 64;
    memory += sizeof(value_t) * values_.size();
    memory += sizeof(value_t) * deltas_.size();

    Communicator communicator;
    communicator.InitCommunicator(comm_spec.comm());
    communicator.template Sum(memory, global_mem);

    if (batch_stage_) {
      batch_stage_ = false;
      if (comm_spec.worker_id() == grape::kCoordinatorRank) {
        LOG(INFO) << "Mem: " << global_mem / 1024 / 1024 << " MB";
      }
    }
  }

  void reInit(size_t all_node_num, const FRAG_T& frag) {
    auto vertices = frag.Vertices();
    auto inner_vertices = frag.InnerVertices();
    VertexRange<vid_t> allrange(vertices.begin().GetValue(), all_node_num);
    values_.Init(allrange, default_v());
    deltas_.Init(allrange, default_v());
    for (auto v : inner_vertices) {
      value_t value;
      init_v(v, value);
      values_[v] = value;
      value_t delta;
      init_c(v, delta, frag);
      deltas_[v] = delta;
    }
  }

  long long f_send_num=0;
  long long f_send_value_num=0;
  long long f_send_delta_num=0;
  long long f_index_count_num = 0;
  long long g_num=0;
  long long node_update_num=0;
  std::set<vid_t> touch_nodes;
  VertexArray<vid_t, vid_t> val_dep_;
  VertexArray<vid_t, vid_t> delta_dep_;
  DenseVertexSet<vid_t> curr_modified_, next_modified_;
  VertexArray<value_t, vid_t> values_{};
  VertexArray<value_t, vid_t> deltas_{};
  VertexArray<value_t, vid_t> last_deltas_{};
  VertexArray<value_t, vid_t> last_index_values_{};
  VertexArray<value_t, vid_t> index_values_{};
  VertexArray<value_t, vid_t> degree{};
  bool batch_stage_{true};
  template <typename APP_T, typename SUPERNODE_T>
  friend class IterCompressor;
  template <typename APP_T>
  friend class SumSyncIterWorker;
  template <typename APP_T>
  friend class SumSyncIterWorkerPull;
  template <typename APP_T>
  friend class SumBatchWorker;
};

}  // namespace grape
#endif  // LIBGRAPE_LITE_GRAPE_APP_LAYPH_APP_BASE_H_
