docker rm -f foundation_stereo
DIR=$(pwd)/../
xhost +  && docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name foundation_stereo  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /home/bowen:/home/bowen -v $DIR:$DIR -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE foundation_stereo:latest bash
