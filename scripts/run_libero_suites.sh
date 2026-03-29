#!/bin/bash

# Optimize CUDA memory allocation to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==============================================================================
# Script to run VLS experiments with different LIBERO suites sequentially.
# 
# Usage:
#   ./scripts/run_libero_suites.sh suite_name1 suite_name2 ... [extra_hydra_args]
# 
# Example:
#   ./scripts/run_libero_suites.sh libero_goal libero_spatial main.episode_num=50
# ==============================================================================

# Default suites to run if no positional arguments are provided
# (Uncomment or modify this list if you want a default set)
# DEFAULT_SUITES=("libero_goal" "libero_spatial" "libero_object" "libero_10")

# Separate suite names from extra Hydra arguments
SUITES=()
EXTRA_ARGS=()

# Handle help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [suite_name1 suite_name2 ...] [extra_hydra_args...]"
    echo ""
    echo "If no suite names provided, runs all default suites."
    echo ""
    echo "Examples:"
    echo "  $0                                    # run all default suites"
    echo "  $0 libero_goal libero_spatial"
    echo "  $0 libero_object_task main.episode_num=50"
    echo ""
    echo "Available LIBERO suites (examples):"
    echo "  - libero_goal, libero_spatial, libero_object, libero_10"
    echo "  - libero_goal_lan, libero_spatial_lan, libero_object_lan, libero_10_lan"
    echo "  - libero_goal_task, libero_spatial_task, libero_object_task, libero_10_task"
    echo "  - libero_goal_object, libero_spatial_object, libero_object_object, libero_10_object"
    echo "  - libero_goal_swap, libero_spatial_swap, libero_object_swap, libero_10_swap"
    exit 0
fi

for arg in "$@"; do
    if [[ $arg == *=* ]]; then
        EXTRA_ARGS+=("$arg")
    elif [[ $arg == -* ]]; then
        # Forward flags starting with - to extra args (like --help if passed later)
        EXTRA_ARGS+=("$arg")
    else
        SUITES+=("$arg")
    fi
done

# If no suites provided after parsing
if [ ${#SUITES[@]} -eq 0 ]; then
     SUITES=("libero_goal" "libero_goal_object" "libero_goal_lan" "libero_spatial" "libero_spatial_object" "libero_spatial_lan" "libero_10" "libero_10_object" "libero_10_lan" "libero_object" "libero_object_object" "libero_object_lan") 
    # echo "Error: No suite names provided."
    # exit 1
fi

echo "Running experiments for suites: ${SUITES[*]}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "With extra arguments: ${EXTRA_ARGS[*]}"
fi

for SUITE in "${SUITES[@]}"; do
    echo ""
    echo "================================================================================"
    echo "STARTING EXPERIMENT: $SUITE"
    echo "DATE: $(date)"
    echo "================================================================================"
    
    # Execute main.py
    # We force backend=libero and override the suite_name
    python main.py \
        backend=libero \
        backend.libero.suite_name="$SUITE" \
        "${EXTRA_ARGS[@]}"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Successfully finished experiment for: $SUITE"
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "EXPERIMENT FAILED for suite: $SUITE (Exit code: $EXIT_CODE)"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # Optional: Ask if user wants to continue or stop
        # read -p "Do you want to continue to the next suite? (y/n) " -n 1 -r
        # echo
        # if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        #     exit 1
        # fi
    fi
done

echo ""
echo "================================================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "================================================================================"
