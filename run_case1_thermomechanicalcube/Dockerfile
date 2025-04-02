FROM idaholab/moose-dev:latest

ENV MOOSE_DIR=/opt/moose
ENV PYTHONPATH=${MOOSE_DIR}/python
ARG METHOD=opt
ARG WORKDIR=/opt

ARG NUM_CPUS=4
ARG MOOSE_JOBS=4
ENV PETSC_DIR=/opt/petsc
ENV LIBMESH_DIR=/opt/libmesh
ENV PATH=/opt/miniforge3/bin:$PATH 
ENV WASP_DIR=/opt/wasp 
ENV PATH=${WASP_DIR}/bin:$PATH 

# set env variables and source MOOSE deps
# Build MOOSE
RUN source /opt/rh/gcc-toolset-12/enable && \
    source /opt/mpi/use-openmpi && \
    cd ${WORKDIR} && \
    git clone https://github.com/idaholab/moose.git && \
    cd moose/framework && \
    make -j "$NUM_CPUS" && \
    cd ../modules/combined && \
    make -j "$NUM_CPUS"
    
# install uq-toolkit
RUN cd ${WORKDIR} && \
    git clone https://github.com/farscape-project/uq-toolkit.git && \
    pip install -r uq-toolkit/requirements.txt

