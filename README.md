# AMD HIP image filters

Apply filters to images with the usage of a GPU.


## Usage
```bash
# Setup
meson setup build --native-file native/hip.ini --reconfigure

# Build
ninja -C build

# Launch
./build/src/app/hip-img-fx \
    --input examples/example_01.jpg \
    --filter grayscale \
    --output examples/example_01_output_grayscale.jpg
```

### Supported image filters
- Grayscale
- Gaussian Blur
- Negative

## Credits:

Images:
- example_01: https://unsplash.com/photos/pagoda-surrounded-by-trees-E_eWwM29wfU
