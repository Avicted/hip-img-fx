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

## Credits:

Images (unsplash.com):
- [example_01](https://unsplash.com/photos/pagoda-surrounded-by-trees-E_eWwM29wfU)
- [example_02](https://unsplash.com/photos/blue-and-brown-bird-on-brown-tree-trunk-DPXytK8Z59Y)
- [example_03](https://unsplash.com/photos/selective-focus-photo-of-giraffe-D6TqIa-tWRY)
- [example_04](https://unsplash.com/photos/black-white-and-yellow-bird-on-brown-tree-branch-during-daytime-vjFC9OjrOtA)
- [example_05](https://unsplash.com/photos/two-white-ferrets-zQTw2g6JY6U)
