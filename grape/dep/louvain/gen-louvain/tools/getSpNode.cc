#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <ctime>
#include <string>
#include <stdlib.h>
#include <cmath>
using namespace std;

void readData(string edge_new_path, int pass_step)
{
    std::cout << "\nread file... " << edge_new_path << std::endl;
    ifstream inFile(edge_new_path);
    if(!inFile){
        std::cout << "open file failed. " << edge_new_path << std::endl;
        exit(0);
    }
    int u, v;
    std::string pass;
    for(int i = 0; i < pass_step; i++){
        getline(inFile, pass);
        std::cout << pass << std::endl;
    }

    std::unordered_map<int, int> cluster; 
    int node_num = 0;
    while(inFile >> u >> v){
        // std::cout << u << " " << v << std::endl;
        if(cluster.find(v) == cluster.end()){
            cluster[v] = 1;
        }
        else{
            cluster[v]++;
        }
        node_num++;
    }
    inFile.close();

    std::cout << "read finish..." << std::endl;
    std::cout << "node_num=" << node_num << std::endl;
    std::cout << "cluster.size=" << cluster.size() << std::endl;
    int cmp_node_num = 0;
    int MIN_NODE = 8;
    int MAX_NODE = 100;
    int min_node = 0x3f3f3f3f;
    int max_node = 0;
    int super_node_num = 0;
    for(auto cs : cluster){
        // std::cout << cs.first << ": " << cs.second << std::endl;
        if(cs.second > min_node){
            cmp_node_num += cs.second;
            super_node_num++;
        }
        min_node = std::min(min_node, cs.second);
        max_node = std::max(max_node, cs.second);
    }
    std::cout << "cmp_rate=" << (cmp_node_num*1.0) / node_num << std::endl;
    std::cout << "super_node_num=" << super_node_num << std::endl;
    std::cout << "min_node=" << min_node << std::endl;
    std::cout << "max_node=" << max_node << std::endl;
}

void read_data(std::string &line, std::set<int>& P) {
    std::string linestr(line);
    int pos = linestr.find_first_of(" "); //找到第一个空格的位置
    if (pos == -1) return; 

    int size = atoi(linestr.substr(0, pos).c_str()); //字符串转换成数值  源点
    std::string links = linestr.substr(pos + 1);
    cout<<"links:"<<links<<endl;
    if (*(links.end() - 1) != ' ') {
        links = links + " ";
    }
    int spacepos = 0;
    while ((spacepos = links.find_first_of(" ")) != links.npos) {
        int to;
        if (spacepos > 0) {
            to = atoi(links.substr(0, spacepos).c_str());
        }
        links = links.substr(spacepos + 1);
        P.insert(to);
    }
    if(P.size() != size){
        std::cout << "error!" << std::endl;
    }
}

void read_supernode1(const string path){
    std::cout << "read file... " << path << std::endl;
    std::ifstream inFile;
    inFile.open(path);
    if (!inFile) {
       std::cout << "open file failed. " << path;
        exit(0);
    }
    char linechr[2024000];
    int u, v;
    int w = 1;
    while (inFile.getline(linechr, 2024000)) { //read a line of the input file, ensure the buffer is large enough
        std::string line(linechr);
        if(line.length() < 1 || line[0] == '#'){
            continue;
        }
        std::set<int> P;
        read_data(line, P); //invoke api, get the value of key field and data field
        std::cout << "w=" << w << " p.size=" << P.size() << std::endl;
        for(auto p : P){
            std::cout << p << " ";
        }
        std::cout << std::endl;
        w++;
        if(w > 4){
            return ;
        }
    }
}
void read_supernode(const string path){
    ifstream inFile(path);
    if(!inFile){
        std::cout << "open file failed. " << path << std::endl;
        exit(0);
    }
    size_t size;
    int v_id;
    int w = 0;
    while(inFile >> size){
        std::set<int> P;
        for(int i = 0; i < size; i++){
            inFile >> v_id;
            P.insert(v_id);
        }
        std::cout << "w=" << w << " p.size=" << P.size() << std::endl;
        for(auto p : P){
            std::cout << p << " ";
        }
        std::cout << std::endl;
        w++;
        if(w > 4){
            return ;
        }
    }
}


/*
    r_w_Data():
    从聚类结果中统计超点信息，并生成超点文件。
    聚类文件：
        epsilon: 0.4
        mu: 5
        Runtime: 0.035864 sec 
        node	cluster
        0	0
        1	0
        2	0
        3	1
        4	1
        5	2
        前几行为统计信息，后面每行表示每个点所属类别id
    超点文件：
        3 0 1 2 
        2 3 4
        1 5
        每行第一列表示包含点的个数， 后面为该聚类所包含点的id
*/ 
void r_w_Data(string edge_new_path, int pass_step, string sp_path) {
    std::cout << "read file... " << edge_new_path << std::endl;
    ifstream inFile(edge_new_path);
    if(!inFile){
        std::cout << "open file failed. " << edge_new_path << std::endl;
        exit(0);
    }
    int u, v;
    std::string pass;
    for(int i = 0; i < pass_step; i++){
        getline(inFile, pass);
        std::cout << "pass:" << pass << std::endl;
    }

    std::unordered_map<int, std::unordered_set<int>> cluster; 
    int node_num = 0;
    while(inFile >> u >> v){
        // std::cout << u << " " << v << std::endl;
        // if(cluster.find(v) == cluster.end()){
        //     cluster[v] = 1;
        // }
        // else{
        //     cluster[v]++;
        // }
        cluster[v].insert(u);
        node_num++;
    }
    inFile.close();

    std::cout << "read finish..." << std::endl;
    std::cout << "node_num=" << node_num << std::endl;
    std::cout << "cluster.size=" << cluster.size() << std::endl;
    int cmp_node_num = 0;
    int MIN_NODE = 5;
    int MAX_NODE = 100;
    int min_node = 0x3f3f3f3f;
    int max_node = 0;
    int super_node_num = 0;
    for(auto cs : cluster){
        // std::cout << cs.first << ": " << cs.second << std::endl;
        if(cs.second.size() > min_node){
            cmp_node_num += cs.second.size();
            super_node_num++;
        }
        min_node = std::min(min_node, int(cs.second.size()));
        max_node = std::max(max_node, int(cs.second.size()));
    }
    std::cout << "cmp_rate=" << (cmp_node_num*1.0) / node_num << std::endl;
    std::cout << "super_node_num=" << super_node_num << std::endl;
    std::cout << "min_node=" << min_node << std::endl;
    std::cout << "max_node=" << max_node << std::endl;

    // 写入文件
    // string sp_path = edge_new_path+".c";
    ofstream fout(sp_path);
    for(auto cs : cluster){
        // std::cout << cs.first << ": " << cs.second << std::endl;
        if(cs.second.size() > min_node){
            fout << cs.second.size();
            for(auto u : cs.second){
                fout << " " << u;
            }
            fout << "\n";
        }
    }
    fout.close();
    std::cout << "finish write... " << sp_path << std::endl;
}

int main(int argc, char **argv){
    std::cout << "- start getspNode..." << std::endl;
    // cmd: g++ getSpNode.cc -o getSpNode && ./getSpNode uk-2005 0.0010
    // std::string filename = "arabic-2005";
    // std::string filename = argv[1];
    // std::string percentage = argv[2];
    // std::cout << filename << ", " << percentage << std::endl;
    // inc
    // std::string cluster_path = "/mnt2/neu/yusong/dataset/louvain_bin/"+filename+"_"+percentage+"_node2comm_level";
    // std::string path = "/mnt2/neu/yusong/dataset/large/"+filename+"/"+percentage+"/"+filename+".base";
    // no-inc
    // std::string cluster_path = "/mnt2/neu/yusong/dataset/louvain_bin/"+filename+"_node2comm_level";
    // std::string path = "/mnt/data/nfs/yusong/dataset/large/"+filename+"/"+filename+".e";

    std::cout << "--------" << std::endl;
    std::string cluster_path = argv[1];
    std::cout << "--------" << std::endl;
    std::string path = argv[2];
    std::cout << "--------" << std::endl;
    std::string max_comm_size = argv[3];
    std::string sp_path = path + ".c_" + max_comm_size; // 超点文件
    std::cout << "cluster_path=" << cluster_path << std::endl;
    std::cout << "write path=" << path << std::endl;
    
    // readData(path, 4);
    // std::cout << "filename=" << filename << std::endl;
    int passLine = 0;
    r_w_Data(cluster_path, passLine, sp_path); // 读聚类的数据,并写入超点数据

    // std::string sp_path = path +".c";
    // read_supernode(sp_path); // 读超点数据t  -

    std::cout << "- finish!!!" << std::endl;
    return 0;
}