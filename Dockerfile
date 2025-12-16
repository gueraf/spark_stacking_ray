# Dockerfile for Ray Cluster Node
# Based on rayproject/ray:nightly-py312-cu129-aarch64

FROM rayproject/ray:nightly-py312-cu129-aarch64

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Build NCCL (with Blackwell support)
WORKDIR /home/ray
RUN git clone -b v2.28.9-1 https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121" && \
    rm -rf /home/ray/nccl/.git # Cleanup

# Copy Startup Script
COPY run_node.sh /usr/local/bin/run_node.sh
RUN chmod +x /usr/local/bin/run_node.sh

# Set Environment for NCCL
ENV NCCL_HOME=/home/ray/nccl/build
ENV LD_LIBRARY_PATH=$NCCL_HOME/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

ENTRYPOINT ["/usr/local/bin/run_node.sh"]

