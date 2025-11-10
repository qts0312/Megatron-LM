PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:25.03-py3"

HOST_MEGATRON_LM_DIR="${HOME}/Megatron-LM"
HOST_CHECKPOINT_PATH="${HOME}/checkpoints/llama3_8b_bf16"
HOST_TENSORBOARD_LOGS_PATH="${HOME}/tensorboard_logs/llama3_8b_bf16"

docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "${HOST_MEGATRON_LM_DIR}:/workspace/megatron-lm" \
  -v "${HOST_CHECKPOINT_PATH}:/workspace/checkpoints" \
  -v "${HOST_TENSORBOARD_LOGS_PATH}:/workspace/tensorboard_logs" \
  --workdir /workspace/megatron-lm \
  $PYTORCH_IMAGE \
  bash
  # bash examples/llama/train_llama3_8b_h100_fp8.sh \
  #   /workspace/checkpoints \
  #   /workspace/tensorboard_logs \
  # 2>&1 | tee "${HOST_TENSORBOARD_LOGS_PATH}/training_mock_$(date +'%y-%m-%d_%H-%M-%S').log"