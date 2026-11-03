#!/bin/bash
# Print an AWS worker's Name tag, instance id and private hostname, from its (changing) public IP.  Needs "Allow tags
# in instance metadata" on the instance (enabled 2026-11-03).  Usage: experiments/awsname.sh <ip>
ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new ubuntu@$1 \
  'T=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60");
   H="X-aws-ec2-metadata-token: $T";
   echo "$(curl -s -H "$H" http://169.254.169.254/latest/meta-data/tags/instance/Name)	$(curl -s -H "$H" http://169.254.169.254/latest/meta-data/instance-id)	$(hostname)"' 2>/dev/null
