#!/usr/bin/env bash
# F1-B (Partial FT, audio=24 / text=6) watchdog.
# Same structure as bin/watchdog_c5.sh — matches by unique model_path substring.
#
# Start:  nohup bash bin/watchdog_f1b.sh > /dev/null 2>&1 & disown
# Stop:   pkill -f watchdog_f1b.sh
#
# Morning check:
#   cat experiments/strategyA_f1b_watchdog_status.txt
#   tail -50 experiments/strategyA_f1b_watchdog.log
#   ls experiments/F1b_*.marker 2>/dev/null

set -u

CRAB=/home/brant/Project/SAILER_test/Crab
LOG_DIR=$CRAB/experiments
WATCH_LOG=$LOG_DIR/strategyA_f1b_watchdog.log
STATUS=$LOG_DIR/strategyA_f1b_watchdog_status.txt
TRAIN_LOG=$(ls -t $LOG_DIR/strategyA_f1b_launch_*.log 2>/dev/null | head -1)

INTERVAL=300

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$WATCH_LOG"; }

write_status() {
  local state="$1"; shift
  cat > "$STATUS" <<EOF
=== F1-B (Partial FT audio=24 text=6) watchdog status ===
last_check : $(date '+%Y-%m-%d %H:%M:%S')
state      : $state
detail     : $*
train_log  : $TRAIN_LOG
watch_log  : $WATCH_LOG
EOF
}

log "watchdog started (interval=${INTERVAL}s, train_log=$TRAIN_LOG)"
write_status starting "first check pending"

CYCLE=0
while true; do
  CYCLE=$((CYCLE + 1))
  PID=$(pgrep -of "train_crab_lora.py.*strategyA_f1b_partialft_asym" 2>/dev/null)

  if [ -z "${PID:-}" ]; then
    if [ -n "$TRAIN_LOG" ] && grep -aq "INFO - Done\." "$TRAIN_LOG"; then
      TEST_LINE=$(grep -a "TEST:" "$TRAIN_LOG" | tail -1)
      log "TRAINING COMPLETE - $TEST_LINE"
      write_status complete "TEST: $TEST_LINE"
      touch "$LOG_DIR/F1b_COMPLETE_$(date +%Y%m%d_%H%M%S).marker"
      break
    fi
    if [ -n "$TRAIN_LOG" ] && grep -aiqE "out of memory|cuda error|runtimeerror|traceback" "$TRAIN_LOG"; then
      OOM_LINE=$(grep -aiE "out of memory|cuda error|runtimeerror|traceback" "$TRAIN_LOG" | tail -1)
      log "CRASH DETECTED - $OOM_LINE"
      write_status crashed "$OOM_LINE"
      touch "$LOG_DIR/F1b_CRASHED_$(date +%Y%m%d_%H%M%S).marker"
      break
    fi
    log "PROCESS GONE - no error marker"
    write_status disappeared "no PID, no error. inspect $TRAIN_LOG"
    touch "$LOG_DIR/F1b_PROCESS_GONE_$(date +%Y%m%d_%H%M%S).marker"
    break
  fi

  GPU_LINE=$(nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu,utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
  GPU_USED=$(echo "$GPU_LINE" | awk -F, '{gsub(/ /,""); print $1}')
  GPU_FREE=$(echo "$GPU_LINE" | awk -F, '{gsub(/ /,""); print $2}')
  GPU_TEMP=$(echo "$GPU_LINE" | awk -F, '{gsub(/ /,""); print $3}')
  GPU_UTIL=$(echo "$GPU_LINE" | awk -F, '{gsub(/ /,""); print $4}')
  GPU_PWR=$(echo "$GPU_LINE"  | awk -F, '{gsub(/ /,""); print $5}')
  RAM_AVAIL=$(free -h | awk '/^Mem:/ {print $7}')
  SWAP_USED=$(free -h | awk '/^Swap:/ {print $3}')

  STEP=$(tail -c 800 "$TRAIN_LOG" 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/(10436|1306)' | tail -1)
  STEP=${STEP:-N/A}
  LAST_DEV=$(grep -a "Dev: loss" "$TRAIN_LOG" 2>/dev/null | tail -1 | sed 's/.*Dev:/Dev:/')
  LAST_DEV=${LAST_DEV:-"(no dev eval yet)"}
  LAST_BEST=$(grep -a "New best macro-F1" "$TRAIN_LOG" 2>/dev/null | tail -1 | sed 's/.*INFO - //')
  LAST_BEST=${LAST_BEST:-"(no best yet)"}

  WARN=""
  if [ "${GPU_USED:-0}" -gt 24000 ] 2>/dev/null; then WARN="${WARN}[VRAM>24GB]"; fi
  if [ "${GPU_TEMP:-0}" -gt 85 ] 2>/dev/null; then WARN="${WARN}[GPU>85C]"; fi

  log "cycle=$CYCLE PID=$PID step=$STEP GPU=${GPU_USED}/24576 MiB free=${GPU_FREE} temp=${GPU_TEMP}C util=${GPU_UTIL}% pwr=${GPU_PWR}W ram=${RAM_AVAIL} swap=${SWAP_USED} ${WARN}"
  write_status alive "PID=$PID step=$STEP | GPU ${GPU_USED}/24576 MiB free=${GPU_FREE} temp=${GPU_TEMP}C util=${GPU_UTIL}% | RAM avail=${RAM_AVAIL} | $LAST_BEST | $LAST_DEV ${WARN}"

  sleep $INTERVAL
done

log "watchdog exiting (cycle=$CYCLE)"
