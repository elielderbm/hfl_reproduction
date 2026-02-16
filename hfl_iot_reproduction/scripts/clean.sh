#!/usr/bin/env bash
docker compose down -v --remove-orphans || true
rm -rf logs/*.jsonl outputs/*.png outputs/*.csv data/har/* data/clients/* 2>/dev/null || true
