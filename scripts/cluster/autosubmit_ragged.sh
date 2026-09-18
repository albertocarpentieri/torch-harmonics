#!/bin/bash
#
# Submit the ragged validation job as soon as the Slurm controller answers.
#
# The controller has been intermittent for two days: it answered long enough for
# job 3787269 to be queued, then went back to "Unable to contact slurm controller"
# within the hour. Sitting on a shell waiting for it is a poor use of anyone's
# attention, and a submission that is one minute late costs nothing, so this polls.
#
# Writes to logs/autosubmit_ragged.log rather than only to a terminal, because the
# last attempt at this lived in a shell whose output file was cleaned up with the
# session and left no record of whether it had ever succeeded.
#
# Usage:
#   bash sbatch_scripts/autosubmit_ragged.sh &            # default: 12h, every 2 min
#   INTERVAL=60 MAX_HOURS=24 bash sbatch_scripts/autosubmit_ragged.sh &
#
# It exits as soon as one job is accepted, and does not resubmit after that, so it
# cannot quietly queue the same 2.5-hour job ten times.

set -uo pipefail

PROJECT="/home/acarpentieri/healda_project"
SCRIPT="${SCRIPT:-${PROJECT}/sbatch_scripts/rebuild_and_validate_ragged.sh}"
LOG="${LOG:-${PROJECT}/logs/autosubmit_ragged.log}"
INTERVAL="${INTERVAL:-120}"
MAX_HOURS="${MAX_HOURS:-12}"

mkdir -p "$(dirname "${LOG}")"
attempts=$(( MAX_HOURS * 3600 / INTERVAL ))

{
  echo "=== $(date -Is) autosubmit started: ${SCRIPT}"
  echo "    polling every ${INTERVAL}s for up to ${MAX_HOURS}h (${attempts} attempts)"
} >> "${LOG}"

for i in $(seq 1 "${attempts}"); do
  # A generous timeout: when the controller is degraded rather than absent, sbatch
  # can take tens of seconds to answer, and killing it early looks like a failure.
  out=$(timeout 90 sbatch "${SCRIPT}" 2>&1 || true)

  if echo "${out}" | grep -q "Submitted batch job"; then
    jobid=$(echo "${out}" | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+")
    {
      echo "=== $(date -Is) SUBMITTED job ${jobid} on attempt ${i}"
      echo "    logs: ${PROJECT}/logs/ragged_rebuild_validate_${jobid}.out"
      echo "    and:  ${PROJECT}/logs/ragged_rebuild_validate_${jobid}.err"
    } >> "${LOG}"
    exit 0
  fi

  # One line per attempt, last line of sbatch's complaint only; the lua requeue
  # banner it prints first is noise.
  echo "$(date -Is) attempt ${i}/${attempts}: $(echo "${out}" | grep -vE '^sbatch: lua:' | tail -1)" >> "${LOG}"
  sleep "${INTERVAL}"
done

echo "=== $(date -Is) gave up after ${MAX_HOURS}h without reaching the controller" >> "${LOG}"
exit 1
