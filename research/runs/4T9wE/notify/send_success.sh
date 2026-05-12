#!/bin/bash
# Triggered when a strategy passes BOTH gates: CAGR >= 50% AND Sharpe >= 2.0
# Args: $1=strategy_name $2=cagr $3=sharpe $4=max_dd
echo "=== SUCCESS NOTIFICATION ==="
echo "Strategy: $1"
echo "CAGR: $2"
echo "Sharpe: $3"
echo "MaxDD: $4"
echo "Timestamp: $(date -u)"
echo "==========================="
