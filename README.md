# ddrf - Dresden Reconstruction Framework

## Build instructions

### Build dependencies

* CMake >= 3.2
* Boost.System >= 1.54.0
* Boost.Filesystem >= 1.54.0
* Boost.Log >= 1.54.0

### Building

```
cd /path/to/ddrf/build/dir
cmake /path/to/ddrf/source/dir -DCMAKE_BUILD_TYPE=RELEASE # or DEBUG
make
```

The built library will reside in /path/to/ddrf/build/dir/Release (Debug)

## Usage

### Additional required dependencies for your project

* Boost.Date_Time >= 1.54.0 when using the SinkStage of the pipeline
* CUDA 7.5 and a compatible C++ compiler with C++11 support (not newer than gcc 4.9) when using the CUDA parts of the framework
* libtiff >= 4.0.0 when using the TIFF image saver