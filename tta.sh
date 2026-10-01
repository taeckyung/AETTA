#!/bin/bash

SRC_PREFIX="reproduce_src"
LOG_PREFIX="eval_results"

BASE_DATASETS=("cifar10outdist") # cifar10outdist cifar100outdist imagenetoutdist
METHODS=("TENT" "EATA" "SAR" "SoTTA" "RoTTA" "CoTTA") # "TENT" "EATA" "SAR" "SoTTA" "RoTTA" "CoTTA"
SEEDS=(0 1 2)
PREFIXES=("--acc_est_method aetta softmax_score gde src_validation adv_perturb")
POSTFIXES=("")  # change to "--reset_function aetta" to use our reset
DISTS=(1)

# For continual TTA, set TGTS to cont
# TGTS="cont"

# For fully TTA, set TGTS to each corruptions
TGTS="gaussian_noise-5
    shot_noise-5
    impulse_noise-5
    defocus_blur-5
    glass_blur-5
    motion_blur-5
    zoom_blur-5
    snow-5
    frost-5
    fog-5
    brightness-5
    contrast-5
    elastic_transform-5
    pixelate-5
    jpeg_compression-5"

echo BASE_DATASETS: "${BASE_DATASETS[@]}"
echo METHODS: "${METHODS[@]}"
echo SEEDS: "${SEEDS[@]}"
echo PREFIXES: "${PREFIXES[@]}"

GPUS=(0 1 2 3 4 5 6 7) #available gpus
NUM_GPUS=${#GPUS[@]}

sleep 1 # prevent mistake
mkdir raw_logs # save console outputs here


#### Useful functions
wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=8  #num concurrent jobs
  local num_max_jobs=${1:-$default_num_jobs}
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}


test_time_adaptation() {
  ###############################################################
  ###### Run Baselines & Ours; Evaluation: Target domains  ######
  ###############################################################

  i=0

  for PREFIX in "${PREFIXES[@]}"; do
    for DATASET in "${BASE_DATASETS[@]}"; do
      for METHOD in "${METHODS[@]}"; do
        for POSTFIX in "${POSTFIXES[@]}"; do
          update_every_x="64"
          memory_size="64"
          SEED="0"
          lr="0.001" #other baselines
          weight_decay="0"

          if [ "${DATASET}" = "cifar10" ] || [ "${DATASET}" = "cifar10outdist" ]; then
            MODEL="resnet18"
            CP_base="log/cifar10/Src/tgt_test/"${SRC_PREFIX}
          elif [ "${DATASET}" = "cifar100" ] || [ "${DATASET}" = "cifar100outdist" ]; then
              MODEL="resnet18"
              CP_base="log/cifar100/Src/tgt_test/"${SRC_PREFIX}
          elif [ "${DATASET}" = "imagenet" ] || [ "${DATASET}" = "imagenetoutdist" ]; then
            MODEL="resnet18_pretrained"
            CP_base="log/imagenet/Src/tgt_test/"${SRC_PREFIX}
          fi

          for SEED in "${SEEDS[@]}"; do #multiple seeds

            if [ "${DATASET}" = "cifar10outdist" ] || [ "${DATASET}" = "cifar100outdist" ]; then
              CP="--load_checkpoint_path ${CP_base}_${SEED}/cp/cp_last.pth.tar"
            else
              CP="" # use torchvision for imagenet
            fi

            if [ "${METHOD}" = "Src" ]; then
              EPOCH=0
              #### Train with BN
              for TGT in $TGTS; do
                python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --update_every_x ${update_every_x} --seed $SEED \
                  --log_prefix ${LOG_PREFIX}_${SEED} \
                  ${PREFIX} \
                  ${POSTFIX}  \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            elif [ "${METHOD}" = "SoTTA" ]; then

              lr="0.001"
              EPOCH=1
              loss_scaler=0
              bn_momentum=0.2

              if [ "${DATASET}" = "cifar10" ] || [ "${DATASET}" = "cifar10outdist" ]; then
                high_threshold=0.99
              elif [ "${DATASET}" = "cifar100" ] || [ "${DATASET}" = "cifar100outdist" ]; then
                high_threshold=0.66
              elif [ "${DATASET}" = "imagenet" ] || [ "${DATASET}" = "imagenetoutdist" ]; then
                high_threshold=0.33
              fi
              #### Train with BN

              for dist in "${DISTS[@]}"; do
                for memory_type in "HUS"; do
                  for TGT in $TGTS; do
                    python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method SoTTA --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                      --remove_cp --online --use_learned_stats --lr ${lr} --weight_decay ${weight_decay} --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} --memory_type ${memory_type} --bn_momentum ${bn_momentum} \
                      --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist}_mt${bn_momentum}_${memory_type}_ht${high_threshold}_lr${lr} \
                      --loss_scaler ${loss_scaler} \
                      ${PREFIX} --sam \
                      ${POSTFIX} \
                      --high_threshold ${high_threshold} \
                      2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                    i=$((i + 1))
                    wait_n
                  done
                done
              done
            elif [ "${METHOD}" = "RoTTA" ]; then
              EPOCH=1
              loss_scaler=0
              lr="0.001"
              bn_momentum=0.05
              #### Train with BN

              for dist in "${DISTS[@]}"; do

                for memory_type in "CSTU"; do
                  for TGT in $TGTS; do
                    python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method "RoTTA" --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                      --remove_cp --online --use_learned_stats --lr ${lr} --weight_decay ${weight_decay} --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} --memory_type ${memory_type} --bn_momentum "0.05" \
                      --log_prefix "${LOG_PREFIX}_${SEED}_dist${dist}_mt0.05_${memory_type}" \
                      --loss_scaler ${loss_scaler} \
                      ${PREFIX} \
                      ${POSTFIX} \
                      2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                    i=$((i + 1))
                    wait_n
                  done
                done
              done
            elif [ "${METHOD}" = "TENT" ]; then
              EPOCH=1
              if [ "${DATASET}" = "imagenet" ] || [ "${DATASET}" = "imagenetoutdist" ]; then
                lr=0.00025 #referred to the paper
              else
                lr=0.001
              fi
              #### Train with BN
              for dist in "${DISTS[@]}"; do
                for TGT in $TGTS; do

                  python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                    --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                    --lr ${lr} --weight_decay ${weight_decay} \
                    --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                    ${PREFIX} \
                    ${POSTFIX} \
                    2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                  i=$((i + 1))
                  wait_n
                done
              done
            elif [ "${METHOD}" = "CoTTA" ]; then
              lr=0.001
              EPOCH=1

              if [ "${DATASET}" = "cifar10" ] || [ "${DATASET}" = "cifar10outdist" ]; then
                aug_threshold=0.92 #value reported from the official code
              elif [ "${DATASET}" = "cifar100" ] || [ "${DATASET}" = "cifar100outdist" ]; then
                aug_threshold=0.72 #value reported from the official code
              elif [ "${DATASET}" = "imagenet" ] || [ "${DATASET}" = "imagenetoutdist" ]; then
                aug_threshold=0.1 #value reported from the official code
              fi

              for dist in "${DISTS[@]}"; do
                for TGT in $TGTS; do

                  python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                    --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                    --lr ${lr} --weight_decay ${weight_decay} \
                    --aug_threshold ${aug_threshold} \
                    --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                    ${PREFIX} \
                    ${POSTFIX} \
                    2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                  i=$((i + 1))
                  wait_n
                done
              done
            elif [ "${METHOD}" = "SAR" ]; then
              EPOCH=1

              BATCH_SIZE=64
              lr=0.00025 # From SAR paper: args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025

              #### Train with BN
              for dist in "${DISTS[@]}"; do
                for TGT in $TGTS; do
                  python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL ${CP} --epoch $EPOCH --seed $SEED \
                    --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                    --lr ${lr} --weight_decay ${weight_decay} \
                    --log_prefix "${LOG_PREFIX}_${SEED}_dist${dist}" \
                    ${PREFIX} \
                    ${POSTFIX} \
                    2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                  i=$((i + 1))
                  wait_n
                done
              done
            elif [ "${METHOD}" = "EATA" ] || [ "${METHOD}" = "ETA" ]; then
              EPOCH=1

              if [ "${DATASET}" = "cifar10" ] || [ "${DATASET}" = "cifar10outdist" ]; then
                lr=0.005
                e_margin=0.92103 # 0.4*ln(10)
                d_margin=0.4
                fisher_alpha=1
              elif [ "${DATASET}" = "cifar100" ] || [ "${DATASET}" = "cifar100outdist" ]; then
                lr=0.005
                e_margin=1.84207 # 0.4*ln(100)
                d_margin=0.4
                fisher_alpha=1
              elif [ "${DATASET}" = "imagenet" ] || [ "${DATASET}" = "imagenetoutdist" ]; then
                lr=0.00025
                e_margin=2.76310 # 0.4*ln(1000)
                d_margin=0.05
                fisher_alpha=2000
              fi

              #### Train with BN
              for dist in "${DISTS[@]}"; do
                for TGT in $TGTS; do
                  python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL ${CP} --epoch $EPOCH --seed $SEED \
                    --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                    --lr ${lr} --weight_decay ${weight_decay} \
                    --log_prefix "${LOG_PREFIX}_${SEED}_dist${dist}" \
                    --e_margin ${e_margin} --d_margin ${d_margin} --fisher_alpha ${fisher_alpha} \
                    ${PREFIX} \
                    ${POSTFIX} \
                    2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                  i=$((i + 1))
                  wait_n
                done
              done
            fi

          done
        done
      done
    done
  done

  wait
}

test_time_adaptation
