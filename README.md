# AMD HIP image filters

Fast GPU-accelerated image filters (HIP) with a CPU fallback path for portability.

## Prerequisites
- AMD ROCm


### Tested on:

hipcc --version\
HIP version: 6.4.43484-123eb5128\
AMD clang version 19.0.0git (/srcdest/rocm-llvm d366fa84f3fdcbd4b10847ebd5db572ae12a34fb)



## Compilation
```bash
meson setup build --native-file native/hip.ini --reconfigure
ninja -C build
```

## Usage

### Help
```bash
./build/src/app/hip-img-fx --help
Running HIP image fx...

Usage: hip-img-fx [options]
Options:
  --input <input_file|input_dir>     Specifies the input file or directory path.
  --output <output_file|output_dir>  Specifies the output file or directory path.
  --filter <filter_type>             Specifies the type of filter to apply (e.g., "grayscale", "negative", "gaussian-blur").
  --use-cpu                          Use CPU for processing instead of GPU.
  --help                             Displays this help information.

Notes:
  - For batch processing, specify both --input and --output as directories.
  - For single image processing, specify both as files.
  - Supported filters: grayscale, negative, gaussian-blur
```

### Examples
```bash
# Launch (single image)
./build/src/app/hip-img-fx \
    --input examples/example_01.jpg \
    --filter grayscale \
    --output examples/example_01_output_grayscale.jpg

# Launch batch
./build/src/app/hip-img-fx \
    --input examples \
    --filter grayscale \
    --output examples/output
```

### Supported image filters
- Grayscale
- Gaussian Blur
- Negative


### Output example
```bash
~ ./build/src/app/hip-img-fx \
    --input examples \
    --filter grayscale \
    --output examples/output

Running HIP image fx...
Input: examples
Output: examples/output
Filter Type: GRAYSCALE
Using GPU for processing.
    HIP Device Count: 1
    Device 0: AMD Radeon RX 6900 XT
        Compute Capability: ------------ = 10.3
        Total Global Memory: ----------- = 17163091968
        Shared Memory per Block: ------- = 65536
        Registers per Block: ----------- = 65536
        Warp Size: --------------------- = 32
        Max Threads per Block: --------- = 1024
        Max Threads Dimension: --------- = (1024, 1024, 1024)
        Max Grid Size: ----------------- = (2147483647, 65536, 65536)
        Clock Rate: -------------------- = 2660000
        Total Constant Memory: --------- = 2147483647
        Multiprocessor Count: ---------- = 40
        L2 Cache Size: ----------------- = 4194304
        Max Threads per Multiprocessor:  = 2048
        Unified Addressing: ------------ = 0
        Memory Clock Rate: ------------- = 1000000
        Memory Bus Width: -------------- = 256
        Peak Memory Bandwidth: --------- = 64.000000


Processing: examples/example_01.jpg -> examples/output/example_01.jpg
Successfully saved output image: examples/output/example_01.jpg
------------------------------------------------------------

Processing: examples/example_02.jpg -> examples/output/example_02.jpg
Successfully saved output image: examples/output/example_02.jpg
------------------------------------------------------------

Processing: examples/example_03.jpg -> examples/output/example_03.jpg
Successfully saved output image: examples/output/example_03.jpg
------------------------------------------------------------

Processing: examples/example_04.jpg -> examples/output/example_04.jpg
Successfully saved output image: examples/output/example_04.jpg
------------------------------------------------------------

Processing: examples/example_05.jpg -> examples/output/example_05.jpg
Successfully saved output image: examples/output/example_05.jpg
------------------------------------------------------------

Batch processing complete. Success: 5, Failed: 0
Total processing time: 00m 00s 897ms
```

## Benchmark
Tested on a batch of 6499 butterfly images from [Kaggle](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification):


| Device  |                 Total Time | Images / sec | Speedup vs CPU |
| ------- | -------------------------: | -----------: | -------------: |
| **CPU** | 3 min 16.547 s (196.547 s) |    **33.07** |             1× |
| **GPU** |                   10.670 s |   **609.09** |     **18.42×** |

GPU is ≈ **1742%** faster than CPU with a Gaussian blur filter (blurAmount = 11) with a batch of 6499 images.

Both CPU and GPU processed all 6499 images successfully (0 failed).



## Credits:

Images (unsplash.com):
- [example_01](https://unsplash.com/photos/pagoda-surrounded-by-trees-E_eWwM29wfU)
- [example_02](https://unsplash.com/photos/blue-and-brown-bird-on-brown-tree-trunk-DPXytK8Z59Y)
- [example_03](https://unsplash.com/photos/selective-focus-photo-of-giraffe-D6TqIa-tWRY)
- [example_04](https://unsplash.com/photos/black-white-and-yellow-bird-on-brown-tree-branch-during-daytime-vjFC9OjrOtA)
- [example_05](https://unsplash.com/photos/two-white-ferrets-zQTw2g6JY6U)


## License
MIT License. See [LICENSE](LICENSE) for details.

