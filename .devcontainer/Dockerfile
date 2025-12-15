FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

# Install additional dependencies
RUN pip install --upgrade pip && \
    pip install numpy scipy

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
