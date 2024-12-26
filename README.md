# FiberDev
## Motivation
**FiberDev** is a volumetric analysis tool for micro-focus X-ray CT images of fiber-reinforced polymer-based composites. This software can compute fiber orientation by [Structural Tensor Analysis: STA](https://en.wikipedia.org/wiki/Structure_tensor).
## Install
### Dependencies
This software dependences following external libraries:
* numpy
* cupy
* numba
* numba_progress
* pandas
* cucim
* matplotlib
* scipy
* opencv-python

Using pip:
```bash
python -m pip install git+https://github.com/Naruki-Ichihara/FiberDev.git@main
```

### Docker
We recomennd to use our docker image. First you should install docker.io or relavant systems. Our docker image was stored in the [dockerhub](https://hub.docker.com/repository/docker/ichiharanaruki/fiberdev/general).

If you use the docker.io and docker-compose, the following command pull and run the above image.
```bash
docker-compose up
```

## Usage
Example files are stored in example directory.
