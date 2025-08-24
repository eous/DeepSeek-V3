#!/bin/bash
# DeepSeek-V3 DDP Training Script with --num-gpu support
# Usage: ./train_ddp.sh [config_name] [--num-gpu N]
# Example: ./train_ddp.sh config_8B_v2 --num-gpu 2

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Parse arguments
CONFIG_NAME="${1:-config_8B_v2}"  # Default to 8B v2 config
NUM_GPUS=1  # Default to 1 GPU
PARALLEL_MODE="auto"  # Default to auto mode

# Parse command line arguments
shift  # Skip the config name
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-gpu)
            NUM_GPUS="$2"
            shift 2
            ;;
        --parallel-mode)
            PARALLEL_MODE="$2"
            shift 2
            ;;
        *)
            print_error "Unknown argument: $1"
            echo "Usage: $0 [config_name] [--num-gpu N] [--parallel-mode {auto|ddp|tp|none}]"
            exit 1
            ;;
    esac
done

# Paths
CONFIG_PATH="configs/training/${CONFIG_NAME}.json"
echo "Using config: $CONFIG_PATH"
DATA_DIR="${DATA_DIR:-data/pile_parquet/man_info_pages/}"  # Can be overridden
CHECKPOINT_DIR="checkpoints/ddp-${CONFIG_NAME}-$(date +%Y%m%d_%H%M%S)"

# Training parameters
BATCH_SIZE=${BATCH_SIZE:-8}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
LOG_INTERVAL=${LOG_INTERVAL:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
GRAD_CLIP=${GRAD_CLIP:-10.0}
SEQ_LEN=${SEQ_LEN:-256}
SEED=${SEED:-77232917}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-1}
NUM_EPOCHS=${NUM_EPOCHS:-1}

# Auto-calculate max steps if not provided
if [ -z "$MAX_STEPS" ]; then
    print_info "Auto-calculating max steps based on dataset size..."
    STEP_INFO=$(python calculate_training_steps.py \
        --data-dir "$DATA_DIR" \
        --seq-len $SEQ_LEN \
        --batch-size $BATCH_SIZE \
        --accumulation-steps $GRADIENT_ACCUMULATION \
        --num-epochs $NUM_EPOCHS \
        --output-json)
    
    MAX_STEPS=$(echo "$STEP_INFO" | python -c "import sys, json; print(json.load(sys.stdin)['total_steps'])")
    WARMUP_STEPS=$(echo "$STEP_INFO" | python -c "import sys, json; print(json.load(sys.stdin)['warmup_steps'])")
    
    print_info "Calculated max steps: $MAX_STEPS"
    print_info "Calculated warmup steps: $WARMUP_STEPS"
else
    MAX_STEPS=${MAX_STEPS:-10000}
    # Calculate warmup as percentage of max steps if not explicitly set
    if [ -z "$WARMUP_STEPS" ]; then
        WARMUP_STEPS=$((MAX_STEPS / 20))  # 5% warmup
        print_info "Calculated warmup steps: $WARMUP_STEPS (5% of $MAX_STEPS)"
    fi
fi

# If WARMUP_STEPS was explicitly set, use it
WARMUP_STEPS=${WARMUP_STEPS:-1000}

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    print_error "Config file not found: $CONFIG_PATH"
    echo "Available configs:"
    ls configs/inference/config_*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//'
    exit 1
fi

# Activate virtual environment
if [ -d "venv" ]; then
    print_info "Activating virtual environment..."
    source venv/bin/activate
else
    print_warning "Virtual environment not found. Running with system Python."
fi

# Print configuration
echo "====================================="
print_info "DDP Training Configuration:"
echo "====================================="
echo "Config: $CONFIG_NAME"
echo "Number of GPUs: $NUM_GPUS"
echo "Parallel mode: $PARALLEL_MODE"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Data dir: $DATA_DIR"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total batch size: $((BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION))"
echo "Learning rate: $LEARNING_RATE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Sequence length: $SEQ_LEN"
echo "Max steps: $MAX_STEPS"
echo "====================================="

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Memory optimization
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
export CUDA_MODULE_LOADING=LAZY

# Force color output for rich console
export FORCE_COLOR=1
export PYTHONUNBUFFERED=1

# DataLoader optimization - reduce workers if system suggests it
# You can override by setting FORCE_SINGLE_WORKER=1 for stability
# export FORCE_SINGLE_WORKER=1

# Build the training command
if [ "$NUM_GPUS" -gt 1 ]; then
    print_info "Using distributed training with $NUM_GPUS GPUs"
    
    # Set distributed environment variables
    export OMP_NUM_THREADS=4
    export GOMP_CPU_AFFINITY=
    
    # Use torchrun for distributed training
    CMD="torchrun \
        --nproc_per_node=$NUM_GPUS \
        --standalone \
        --nnodes=1 \
        training.py"
else
    print_info "Using single GPU training"
    CMD="python training.py"
fi

# Add training arguments
CMD="$CMD \
    --config $CONFIG_PATH \
    --data-dirs $DATA_DIR \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --warmup-steps $WARMUP_STEPS \
    --max-steps $MAX_STEPS \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --save-dir $CHECKPOINT_DIR \
    --use-amp \
    --grad-clip $GRAD_CLIP \
    --seq-len $SEQ_LEN \
    --seed $SEED \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
    --gradient-checkpointing \
    --use-gradient-monitor \
    --parallel-mode $PARALLEL_MODE \
    --num-workers 2"

# Add DDP flag if using multiple GPUs with auto mode
if [ "$NUM_GPUS" -gt 1 ] && [ "$PARALLEL_MODE" = "auto" ]; then
    CMD="$CMD --use-ddp"
fi

# Log the full command
print_debug "Full command: $CMD"

# Create a log file
LOG_FILE="$CHECKPOINT_DIR/training.log"
print_info "Logging to: $LOG_FILE"

# Run training
echo ""
print_info "Starting training..."
echo ""

# Run with tee to both display and log output
$CMD 2>&1 | tee "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_info "Training completed successfully!"
    print_info "Checkpoints saved to: $CHECKPOINT_DIR"
else
    print_error "Training failed! Check the log file: $LOG_FILE"
    exit 1
fi
