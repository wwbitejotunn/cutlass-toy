# bash compile.sh 2>&1 |tee compile.log
# echo "==== compile finish ===="
CUDA_VISIBLE_DEVICES=2 ./fpAintB_test 16000 12288 6144