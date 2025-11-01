# AMD HIP image filters

Apply filters to images with the usage of a GPU.


## Usage
```bash
# Setup
meson setup build --native-file native/hip.ini --reconfigure

# Build
ninja -C build

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
Total processing time: 00m 00s 889ms
```

## Credits:

Images (unsplash.com):
- [example_01](https://unsplash.com/photos/pagoda-surrounded-by-trees-E_eWwM29wfU)
- [example_02](https://unsplash.com/photos/blue-and-brown-bird-on-brown-tree-trunk-DPXytK8Z59Y)
- [example_03](https://unsplash.com/photos/selective-focus-photo-of-giraffe-D6TqIa-tWRY)
- [example_04](https://unsplash.com/photos/black-white-and-yellow-bird-on-brown-tree-branch-during-daytime-vjFC9OjrOtA)
- [example_05](https://unsplash.com/photos/two-white-ferrets-zQTw2g6JY6U)
