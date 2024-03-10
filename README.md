# CSG raycasting with CUDA
[CSG](https://en.wikipedia.org/wiki/Constructive_solid_geometry) (Constructive Solid Geometry) allows a modeler to create 
a complex surface or object by using Boolean operators (union, intersection and difference) to combine simpler objects (wikipedia).

This application allows displaying real-time CSG Scene loaded from `.json` file ((nlohmann/json)[https://github.com/nlohmann/json]). 
The json file format is a little bit akward for now, since it is array representation of a binary tree. 

![image](https://github.com/migoox/cuda-csg-raycasting/assets/56317134/6346f133-ab86-4522-bb9d-640ebd189c9d)


The CSG raycasting algorithm implemented using CUDA was based on the [Spatially Efficient Tree Layout for GPU Ray-tracing of
Constructive Solid Geometry Scenes](https://ceur-ws.org/Vol-1576/090.pdf) by D.Y. Ulyanov, D.K. Bogolepov and V.E. Turlapov.

|  |  |
|---|---|
|![image](https://github.com/migoox/cuda-csg-raycasting/assets/56317134/0f00c2c0-3978-4ff4-b8e8-bfb795598afa)|![image](https://github.com/migoox/cuda-csg-raycasting/assets/56317134/24866f77-bda1-4bce-901d-cdee3f160966)|


## Usage




## Requirements
You need [NVIDIA CUDA GPU](https://developer.nvidia.com/cuda-gpus) to run the application. This application has been tested on the following GPU's: 
| GPU | Memory | Compute Capability |
|---|---|---|
| NVIDIA GTX 1650Ti  | 4GB | 7.5 |
| NVIDIA GTX 1060  | 3GB | 6.1 |
| NVIDIA GTX 1070  | 8GB | 6.1 |

## Compilation Guide
In order to compile the project you need to download and install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [CMake](https://cmake.org/) first.

### Linux
1. Clone the repository to the desired location `git clone https://github.com/migoox/boids-simulation`,
2. Navigate to the cloned directory and run `cmake .`,
3. Run `make`.
### Windows
1. Clone the repository to the desired location `git clone https://github.com/migoox/boids-simulation`,
2. Navigate to the cloned directory and run `cmake .`,
3. Open `boids_simulation.sln` using Visual Studio,
4. Set `boids_simulation` as the Startup Project (right click on `boids_simulation` and choose `Set as Startup Project`),
5. Compile using Debug/Release mode.
