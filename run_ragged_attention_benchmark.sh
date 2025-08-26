#!/bin/bash

# Script to run ragged_attention benchmark multiple times
# Author: Generated for yhao
# Date: $(date)

# Configuration
LOG_DIR="/tmp/yhao"
LOG_FILE="${LOG_DIR}/iter_0-27-tritonbench.log"
CSV_FILE="${LOG_DIR}/iter_0-27-results.csv"
COMMAND="python run.py --op ragged_attention --metrics tflops --bwd --only hstu"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Initialize log file
echo "=== Tritonbench Ragged Attention Benchmark Log ===" > "${LOG_FILE}"
echo "Started at: $(date)" >> "${LOG_FILE}"
echo "Command: ${COMMAND}" >> "${LOG_FILE}"
echo "=========================================" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

# Initialize CSV file
echo "iter,command,exit_code,status,timestamp" > "${CSV_FILE}"

# Main loop
for CURRENT_ITER in {0..27}; do
    # Export CURRENT_ITER as environment variable
    export CURRENT_ITER

    echo "=== Starting iteration ${CURRENT_ITER} ===" | tee -a "${LOG_FILE}"
    echo "Timestamp: $(date)" | tee -a "${LOG_FILE}"
    echo "Command: ${COMMAND}" | tee -a "${LOG_FILE}"
    echo "Environment: CURRENT_ITER=${CURRENT_ITER}" | tee -a "${LOG_FILE}"
    echo "---" | tee -a "${LOG_FILE}"

    # Execute the command and capture exit code
    if ${COMMAND} >> "${LOG_FILE}" 2>&1; then
        EXIT_CODE=0
        STATUS="SUCCESS"
        echo "✓ Iteration ${CURRENT_ITER} completed successfully" | tee -a "${LOG_FILE}"
    else
        EXIT_CODE=$?
        STATUS="FAILED"
        echo "✗ Iteration ${CURRENT_ITER} failed with exit code ${EXIT_CODE}" | tee -a "${LOG_FILE}"
    fi

    # Record to CSV
    echo "${CURRENT_ITER},\"${COMMAND}\",${EXIT_CODE},${STATUS},\"$(date)\"" >> "${CSV_FILE}"

    echo "=== End of iteration ${CURRENT_ITER} ===" | tee -a "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"

    # Optional: Add a small delay between iterations
    sleep 1
done

# Final summary
echo "=== BENCHMARK COMPLETED ===" | tee -a "${LOG_FILE}"
echo "Finished at: $(date)" | tee -a "${LOG_FILE}"
echo "Total iterations: 28 (0-27)" | tee -a "${LOG_FILE}"

# Count successes and failures
SUCCESS_COUNT=$(grep -c "SUCCESS" "${CSV_FILE}")
FAILURE_COUNT=$(grep -c "FAILED" "${CSV_FILE}")

echo "Successful runs: ${SUCCESS_COUNT}" | tee -a "${LOG_FILE}"
echo "Failed runs: ${FAILURE_COUNT}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Results CSV: ${CSV_FILE}" | tee -a "${LOG_FILE}"

echo ""
echo "Benchmark completed!"
echo "Check logs at: ${LOG_FILE}"
echo "Check results at: ${CSV_FILE}"
