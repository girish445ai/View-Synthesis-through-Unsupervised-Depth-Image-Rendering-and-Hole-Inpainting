MODEL='PCOnvNet' 
BASE_PATH='/data2/pconv_files_new'
BATCH_SIZE=1

NUM_EPOCHS=500
DEVICE='cuda'
# NC=2

DATA_ROOT_DIR='/data2/AdaMPI-Inpaint/warpback/data'
DEPTH_ROOT_DIR='/data2/AdaMPI-Inpaint/warpback/data_depth'

LR_SCHEDULER='cosS'  # noS, expS, cosS, StepLR
LR=0.01
OPTIMIZER='AdamW'

EXP_DIR=${BASE_PATH}'/inpaint_experiments/'${MODEL}'/using_monocular_depths'


echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --data-root-dir ${DATA_ROOT_DIR} --depth-root-dir ${DEPTH_ROOT_DIR} --lr_sched ${LR_SCHEDULER} --lr $LR --optimizer ${OPTIMIZER} --preds-output-dir ${EXP_DIR}'/preds' 
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --data-root-dir ${DATA_ROOT_DIR} --depth-root-dir ${DEPTH_ROOT_DIR} --lr_sched ${LR_SCHEDULER} --lr $LR --optimizer ${OPTIMIZER}  --preds-output-dir ${EXP_DIR}'/preds' 

