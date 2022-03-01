#export LD_LIBRARY_PATH="/home/ubuntu/":$LD_LIBRARY_PATH
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export OMP_NUM_THREADS=4
export KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1

#taskset -c 0-20 python3 /home/ubuntu/opennmt/OpenNMT-py-1.2.0/translate.py \
numactl --physcpubind=0-3 --membind=0 python  /home/xiaobing/Downloads/openNMT/OpenNMT-py-1.2.0_new/translate.py \
    --model ./checkpoint_step_250000.pt \
    --src ./test_data/test.en \
    --output ./translated_xianf \
    --shard_size 1000000 \
    --beam_size 4 \
    --max_length 200 \
    --length_penalty "wu" \
    --alpha 0.6 \
    --coverage_penalty "wu" \
    --beta 0.6 \
    --batch_size 1 \
    --report_time \
    --fp32
