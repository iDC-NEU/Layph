/*
    vector simulation queue, support deletion.
*/
#ifndef GRAPE_UTILS_Queue_H_
#define GRAPE_UTILS_Queue_H_
#include <vector>
#include <random>
template<class Data_t, class vid_t>
class Queue{
public:
    std::vector<Data_t> data;
    vid_t head;
    vid_t tail;

    Queue(vid_t size=10){
        data.resize(size);
        head=0;
        tail=0;
    }

    void push(const Data_t& e){
        data[tail] = e;
        tail++;
        if(tail == data.size()){
            data.resize(data.size()*2);
        }
    }

    Data_t front(){
        return data[head];
    }

    Data_t getById(vid_t id){
        return data[id];
    }

    Data_t pop(){
        return data[head++];
    }

    void erase(vid_t id){
        if(id != tail-1){
            std::swap(data[id], data[tail-1]);
        }
        tail--;
    }

    int size(){
        return tail-head;
    }

    bool empty(){
        return size() == 0;
    }

    void print(){
        std::cout << "data: ";
        for(int i = head; i < tail; i++){
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }

    /* return index of data */
    void sample(vid_t sample_size, std::vector<vid_t>& sample){
        if(sample_size >= size()){
            for(vid_t i = head; i < tail; i++){
                sample.emplace_back(i);
            }
        }
        else{
            std::unordered_set<vid_t> id_set;
            // random number generator
            std::random_device rd;  //Get random number seed
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(head, tail - 1);
            // sample random pos, the sample reflect the whole data set more or less 
            vid_t i;
            for (i = 0; i < sample_size; i++) {
                vid_t rand_pos = dis(gen);
                while (id_set.find(rand_pos) != id_set.end()) {
                    rand_pos = dis(gen);
                }
                id_set.insert(rand_pos);
                sample.emplace_back(rand_pos);
            }
        }
    }

};
#endif  // GRAPE_UTILS_Queue_H_