#!/bin/bash

rm -f jn_output.log
rm -f jn_server.log
qsub -q I_40T_64G_NVIDIA_TeslaT4_16G.q -l hostname=m041 jn.sh

# Wait until jn_server.log exists
while [ ! -f "jn_server.log" ]; do
    sleep 1  # Check every second
done

# Output the contents of jn_server.log to the screen
cat jn_server.log

