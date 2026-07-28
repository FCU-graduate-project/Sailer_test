#!/usr/bin/env bash
# C14 Full FT watchdog
# ====================
# Independent monitor that survives terminal/Claude/SSH close.
#
# Runs every 5 minutes:
#   • check training PID alive
#   • check GPU VRAM / temp
#   • check RAM availability
#   • read latest tqdm step (so you know which epoch we're on)
#   • write summary to status file with timestamp
#   • on crash: leave a marker file + log loudly + exit
#
# NOT auto-restarting:
#   train_crab_lora.py's --resume only loads ser_model state, not full FT
#   text+ssl state. Safer to leave manual restart for the morning so we
#   don't loop on a broken config.
#
# Start (nohup so it survives terminal close):
#   nohup bash bin/watchdog_c14.sh > /dev/null 2>&1 &
#
# Stop:
#   pkill -f watchdog_c14.sh
#
# In the morning:
#   cat experiments/strategyA_fullft_watchdog_status.txt   # latest snapshot
#   tail -50 experiments/strategyA_fullft_watchdog.log     # full history
#   ls experiments/*.marker 2>/dev/null                    # crash markers if any

set -u

CRAB=/home/brant/Project/SAILER_test/Crab
LOG_DIR=$CRAB/experiments
WATCH_LOG=$LOG_DIR/strategyA_fullft_watchdog.log
STATUS=$LOG_DIR/strategyA_fullft_watchdog_status.txt
TRAIN_LOG=$(ls -t $LOG_DIR/strategyA_fullft_relaunch_*.log 2>/dev/null | head -1)

INTERVAL=300   # 5 minutes between checks

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$WATCH_LOG"
}

write_status() {
  local state="$1"; shift
  cat > "$STATUS" <<EOF
=== C14 Full FT watchdog status ===
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

  # ── locate training PID ──
  # `pgrep -o -f` returns the OLDEST matching process = the parent (DataLoader
  # workers fork later and inherit cmdline). Was `... | head -1` which assumes
  # lowest PID = parent — fragile if Linux ever recycles PIDs.
  PID=$(pgrep -of "train_crab_lora.py.*ft_mode full_ft" 2>/dev/null)

  # ── handle process gone ──
  if [ -z "${PID:-}" ]; then
    # case A: training finished cleanly
    if [ -n "$TRAIN_LOG" ] && grep -aq "INFO - Done\." "$TRAIN_LOG"; then
      TEST_LINE=$(grep -a "TEST:" "$TRAIN_LOG" | tail -1)
      log "TRAINING COMPLETE — Done. marker present"
      log "  $TEST_LINE"
      write_status complete "training finished cleanly. TEST: $TEST_LINE"
      touch "$LOG_DIR/COMPLETE_$(date +%Y%m%d_%H%M%S).marker"
      break
    fi

    # case B: crashed (look for typical CUDA / Python errors)
    if [ -n "$TRAIN_LOG" ] && grep -aiqE "out of memory|cuda error|runtimeerror|traceback" "$TRAIN_LOG"; then
      OOM_LINE=$(grep -aiE "out of memory|cuda error|runtimeerror|traceback" "$TRAIN_LOG" | tail -1)
      log "CRASH DETECTED — error string in log: $OOM_LINE"
      write_status crashed "$OOM_LINE"
      touch "$LOG_DIR/CRASHED_$(date +%Y%m%d_%H%M%S).marker"
      log "marker file written. NOT auto-restarting (resume not supported for full_ft)"
      log "to restart manually in the morning: bash bin/run_strategyA_fullft.sh"
      break
    fi

    # case C: process disappeared but no obvious error — unknown
    log "PROCESS GONE but no crash marker in log — unknown cause"
    write_status disappeared "no PID, no error string. Inspect $TRAIN_LOG tail."
    touch "$LOG_DIR/PROCESS_GONE_$(date +%Y%m%d_%H%M%S).marker"
    break
  fi

  # ── process alive, snapshot system ──
  GPU_LINE=$(nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu,utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
  GPU_USED=$(echo "$GPU_LINE" | awk -F, '{gsub(/ /,""); print $1}')
  GPU_FREE=$(echo "$GPU_LINE" | awk -F, '{gsub(/ /,""); print $2}')
  GPU_TEMP=$(echo "$GPU_LINE" | awk -F, '{gsub(/ /,""); print $3}')
  GPU_UTIL=$(echo "$GPU_LINE" | awk -F, '{gsub(/ /,""); print $4}')
  GPU_PWR=$(echo "$GPU_LINE"  | awk -F, '{gsub(/ /,""); print $5}')
  RAM_AVAIL=$(free -h | awk '/^Mem:/ {print $7}')
  SWAP_USED=$(free -h | awk '/^Swap:/ {print $3}')

  # latest tqdm step
  STEP=$(tail -c 800 "$TRAIN_LOG" 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/(10436|1306)' | tail -1)
  STEP=${STEP:-N/A}

  # latest dev eval result (if any new)
  LAST_DEV=$(grep -a "Dev: loss" "$TRAIN_LOG" 2>/dev/null | tail -1 | sed 's/.*Dev:/Dev:/')
  LAST_DEV=${LAST_DEV:-"(no dev eval yet)"}

  # latest best
  LAST_BEST=$(grep -a "New best macro-F1" "$TRAIN_LOG" 2>/dev/null | tail -1 | sed 's/.*INFO - //')
  LAST_BEST=${LAST_BEST:-"(no best saved yet)"}

  STATE="alive"
  WARN=""
  # warnings
  if [ "${GPU_USED:-0}" -gt 24000 ] 2>/dev/null; then WARN="${WARN}[VRAM>24GB]"; fi
  if [ "${GPU_TEMP:-0}" -gt 85 ] 2>/dev/null; then WARN="${WARN}[GPU>85C]"; fi

  log "cycle=$CYCLE PID=$PID step=$STEP GPU=${GPU_USED}/24576 MiB free=${GPU_FREE} temp=${GPU_TEMP}C util=${GPU_UTIL}% pwr=${GPU_PWR}W ram=${RAM_AVAIL} swap=${SWAP_USED} ${WARN}"

  write_status "$STATE" "PID=$PID step=$STEP | GPU ${GPU_USED}/24576 MiB free=${GPU_FREE} temp=${GPU_TEMP}C util=${GPU_UTIL}% | RAM avail=${RAM_AVAIL} swap=${SWAP_USED} | $LAST_BEST | $LAST_DEV ${WARN}"

  sleep $INTERVAL
done

log "watchdog exiting (cycle=$CYCLE)"
