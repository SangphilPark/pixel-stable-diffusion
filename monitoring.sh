#!/bin/bash

# Interval in seconds
INTERVAL=1

# Loop indefinitely
while true
do
  # Execute nvidia-smi command
  nvidia-smi
  # Wait for the specified interval
  sleep $INTERVAL
done
