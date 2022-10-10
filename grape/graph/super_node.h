#ifndef GRAPE_GRAPH_SUPER_NODE_H_
#define GRAPE_GRAPH_SUPER_NODE_H_

#include <vector>

namespace grape {

template<class vertex_t, class value_t, class vid_t>
class SuperNodeForIter{

    using elist = std::vector<std::pair<vertex_t, value_t>>;

public:
    vertex_t id; // source node
    // value_t data=0; // sum of all delta
    bool status = false; //Mark whether to create a shortcut
    // std::vector<vertex_t> ids;
    vid_t ids;
    elist inner_value;
    elist inner_delta;
    elist bound_delta;
    void swap(SuperNodeForIter & x){
        std::swap(id, x.id);
        std::swap(status, x.status);
        std::swap(ids, x.ids);
        inner_value.swap(x.inner_value);
        inner_delta.swap(x.inner_delta);
        bound_delta.swap(x.bound_delta);
    }
    void clear(){
        status = false;
        bound_delta.clear();
        inner_value.clear();
        inner_delta.clear();
    }
};

// min/max
template<class vertex_t, class value_t, class delta_t, class vid_t>
class SuperNodeForTrav{
    using elist = std::vector<std::pair<vertex_t, delta_t>>;

public:
    vertex_t id;
    // value_t data=0; // min/max of all delta
    bool status = false; // Mark whether to create a shortcut
    // std::vector<vertex_t> ids;
    vid_t ids;
    elist inner_delta;
    elist bound_delta;
    void swap(SuperNodeForTrav & x){
        std::swap(id, x.id);
        std::swap(status, x.status);
        std::swap(ids, x.ids);
        inner_delta.swap(x.inner_delta);
        bound_delta.swap(x.bound_delta);
    }
    void clear(){
        status = false;
        bound_delta.clear();
        inner_delta.clear();
    }
};

}  // namespace grape
#endif  // GRAPE_GRAPH_SUPER_NODE_H_

