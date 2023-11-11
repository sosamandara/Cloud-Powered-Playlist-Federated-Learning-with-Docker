#!/bin/bash
export NUM_SLAVES=10
python3 create_partitions.py
cd Slave
docker build -t slave_image .
cd ..
docker-compose up -d --scale slave=$NUM_SLAVES
