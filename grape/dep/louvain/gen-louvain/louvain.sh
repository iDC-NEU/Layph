
echo -e "\n Executing louvain.sh...\n"

################################################################################
root_path=../../
ipath=graph.txt
################################################################################


name=test
# name=europe_osm
# name=web-uk-2005
percentage=0.0000 # 0.3000 0.4000
max_comm_size=5000
max_level=2
beta=0.80

opath=${ipath}

echo ipath=${ipath}
echo opath=${opath}

louvain_path=${root_path}/dep/louvain/gen-louvain

${louvain_path}/convert -i ${ipath} -o ${opath}.bin
echo "-------------------------------------"

cmd="${louvain_path}/louvain ${opath}.bin -l -1 -v -q 0 -e 0.001 -a ${max_level} -m ${max_comm_size} > ${opath}.tree"  # q=0: modularity, q=10: suminc
echo $cmd
eval $cmd
echo "-------------------------------------"

${louvain_path}/hierarchy ${opath}.tree
level=`expr ${max_level} - 1`
echo level=$level
${louvain_path}/hierarchy ${opath}.tree -l ${level} > ${opath}_node2comm_level
echo "-------------------------------------"

echo ${louvain_path}/getSpNode.cc
g++ ${louvain_path}/tools/getSpNode.cc -o ${louvain_path}/tools/getSpNode
cmd="${louvain_path}/tools/getSpNode ${opath}_node2comm_level ${ipath}"
echo $cmd
eval $cmd
echo "-------------------------------------"