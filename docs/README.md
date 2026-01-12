# HIP Image FX Documentation

Complete documentation for the HIP Image FX GPU-accelerated image processing framework with automatic kernel autotuning.

---

## Getting Started

1. **[Main README](../README.md)** - Project overview, installation, and quick start
2. **[Autotuning Guide](AUTOTUNING_GUIDE.md)** - Complete framework documentation
3. **[Changelog](CHANGELOG.md)** - Version history and release notes
4. **[Cache Files Guide](AUTOTUNING_CACHE_FILES.md)** - Understanding autotuning cache system

### Key Features

- Zero-configuration autotuning: Optimal GPU configurations automatically
- Three-tier caching: Thread-local, persistent, and tuning-based  
- Compile-time safety: C++20 concepts enforce correct usage
- Production-ready: Stable API, pkg-config support, installable headers

---

## Documentation

### [Autotuning Guide](AUTOTUNING_GUIDE.md)

Complete guide covering:
- Quick start (5 minutes)
- API reference
- Advanced usage patterns
- Performance tuning
- Troubleshooting

### Performance

Key performance findings:
- 12-18% speedup from optimal block configurations
- ~100-200ms first-run tuning overhead
- Zero overhead for cached configurations
- Architecture-specific tuning (gfx1030, gfx90a, etc.)

---

## For Developers

### Implementing Custom Kernels

See [Autotuning Guide - Using the Framework](AUTOTUNING_GUIDE.md#using-the-framework) for:
- Defining kernel traits
- Using TuningOrchestrator
- Context-aware tuning
- Embedded cache initialization

### API Reference

See [Autotuning Guide - API Reference](AUTOTUNING_GUIDE.md#api-reference) for complete API documentation.

---

## Installation

Headers install to `${PREFIX}/include/hip-img-fx/autotune/`

```bash
meson setup build
meson install -C build
```

Use in your project:
```bash
pkg-config --cflags hip-img-fx
```

---

## License

See [LICENSE](../LICENSE) for project licensing information.


## References

- [AMD ROCm Documentation](https://rocmdocs.amd.com/)
- [HIP Programming Guide](https://github.com/ROCm-Developer-Tools/HIP)
- [GPU Performance Optimization Techniques](https://developer.amd.com/resources/rocm-learning-center/)
- [Separable Gaussian Blur](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-40-incremental-computation-gaussian)


## Credits

Images (unsplash.com):
- [example_01](https://unsplash.com/photos/pagoda-surrounded-by-trees-E_eWwM29wfU)
- [example_02](https://unsplash.com/photos/blue-and-brown-bird-on-brown-tree-trunk-DPXytK8Z59Y)
- [example_03](https://unsplash.com/photos/selective-focus-photo-of-giraffe-D6TqIa-tWRY)
- [example_04](https://unsplash.com/photos/black-white-and-yellow-bird-on-brown-tree-branch-during-daytime-vjFC9OjrOtA)
- [example_05](https://unsplash.com/photos/two-white-ferrets-zQTw2g6JY6U)



