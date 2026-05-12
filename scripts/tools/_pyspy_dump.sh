#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Dump several PIDs in one go. Used by debug-hang skill.
for pid in "$@"; do
  echo ""
  echo "===== PID $pid ====="
  py-spy dump --pid "$pid" 2>&1 | head -120
done
