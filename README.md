# FiberDev
Estimation of axial compressive strength of unidirectional carbon fiber-reinforced plastic considering **Fiber** **Dev**iation.
## Motivation
**FiberDev** is a volumetric analysis tool for micro-focus X-ray CT images of fiber-reinforced polymer-based composites. This software can compute fiber orientation by [Structural Tensor Analysis: STA](https://en.wikipedia.org/wiki/Structure_tensor).
This software also contains an estimation tool of the axial compressive strength in fibrous composite structure considering standard deviation of fiber waviness. See [this paper](https://www.sciencedirect.com/science/article/pii/S1359835X23003974).
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

## Cite
To cite this repository:

```
@software{fiberdev2025github,
  author = {Naruki Ichihara},
  title = {{FiberDev}: Estimation of axial compressive strength of unidirectional carbon fiber-reinforced plastic considering Fiber Deviation},
  url = {https://github.com/Naruki-Ichihara/FiberDev},
  version = {0.0.1},
  year = {2025},
}
```
