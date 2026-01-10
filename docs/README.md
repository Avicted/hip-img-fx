# HIP Image FX Documentation

Welcome to the HIP Image FX documentation. This directory contains comprehensive guides and references for the GPU-accelerated image processing framework and its autotuning system.

---

## 🚀 New Users Start Here

### Getting Started

1. **[Main README](../README.md)** - Project overview, installation, and quick start
2. **[Autotuning Guide](AUTOTUNING_GUIDE.md)** - Complete framework documentation
3. **[Examples](../examples/)** - Sample filter implementations

### Key Features

- **Zero-configuration autotuning**: Optimal GPU configurations automatically
- **Three-tier caching**: Thread-local, persistent, and tuning-based
- **Compile-time safety**: C++20 concepts enforce correct usage
- **Production-ready**: Stable API, pkg-config support, installable headers

---

## 📚 Main Documentation

### Core Framework

- **[Autotuning Guide](AUTOTUNING_GUIDE.md)** ⭐
  - Quick start (5 minutes)
  - Complete API reference
  - Advanced usage patterns
  - Performance tuning tips
  - Troubleshooting

### Quick References

- **[Quick Reference Card](COMPILE_TIME_SAFETY_QUICK_REFERENCE.md)**
  - API cheat sheet
  - Common patterns
  - Validation rules

- **[Framework Summary](AUTOTUNING_SUMMARY.md)**
  - High-level architecture
  - Key components
  - System overview

---

## 📊 Performance & Benchmarks

### Empirical Data

- **[Benchmark Results](BENCHMARK_RESULTS.md)**
  - Performance measurements
  - Configuration comparisons
  - Hardware-specific data

- **[Benchmark Methodology](BENCHMARK_FIX_REPORT.md)**
  - Measurement approach
  - Timing accuracy
  - Validation process

### Key Findings

- **12-18% speedup** from optimal block configurations (vs default 256 threads)
- **~100-200ms** first-run tuning overhead
- **Zero overhead** for cached configurations
- **Architecture-specific** tuning (gfx1030, gfx90a, etc.)

---

## 🔧 For Developers

### Implementing Custom Kernels

See [Autotuning Guide - Using the Framework](AUTOTUNING_GUIDE.md#using-the-framework) for:
- Defining kernel traits
- Using TuningOrchestrator
- Context-aware tuning
- Embedded cache initialization

### API Reference

See [Autotuning Guide - API Reference](AUTOTUNING_GUIDE.md#api-reference) for:
- `TuningOrchestrator<KernelTraits>` interface
- `TuningConfig` factory methods
- `TuningOptions` presets
- Kernel traits requirements

### Advanced Topics

See [Autotuning Guide - Advanced Topics](AUTOTUNING_GUIDE.md#advanced-topics) for:
- Custom tuning strategies
- Multi-GPU considerations
- Cache management
- Production deployment

---

## 📦 Installation & Integration

### Installing Headers

```bash
cd hip-img-fx
meson setup build
meson install -C build
```

Headers install to: `${PREFIX}/include/hip-img-fx/autotune/`

### Using in Your Project

#### With pkg-config

```bash
# Get include flags
pkg-config --cflags hip-img-fx

# In your build system
g++ $(pkg-config --cflags hip-img-fx) mykernel.cpp -o myapp
```

#### Manually

```cpp
#include <hip-img-fx/autotune/orchestrator.h>

// Use TuningOrchestrator in your kernel
```

---

## 🗂️ File Organization

```
docs/
├── README.md                              ← You are here
├── AUTOTUNING_GUIDE.md                    ← Main documentation
├── AUTOTUNING_SUMMARY.md                  ← High-level overview
├── COMPILE_TIME_SAFETY_QUICK_REFERENCE.md ← API cheat sheet
├── BENCHMARK_RESULTS.md                   ← Performance data
├── BENCHMARK_FIX_REPORT.md                ← Methodology notes
├── archive/                               ← Historical documents
│   ├── AUTOTUNING_BEFORE_AFTER.md
│   ├── AUTOTUNING_REFACTOR*.md
│   ├── PHASE*.md
│   ├── TIER1_*.md
│   └── ...
└── examples/
    └── *.h (example kernel implementations)
```

---

## 🔍 Finding What You Need

### "I want to..."

- **...get started quickly** → [Main README Quick Start](../README.md#quick-start)
- **...understand how autotuning works** → [Autotuning Guide - Core Concepts](AUTOTUNING_GUIDE.md#core-concepts)
- **...implement a custom kernel** → [Autotuning Guide - Using the Framework](AUTOTUNING_GUIDE.md#using-the-framework)
- **...see API details** → [Autotuning Guide - API Reference](AUTOTUNING_GUIDE.md#api-reference)
- **...optimize performance** → [Autotuning Guide - Performance Tuning](AUTOTUNING_GUIDE.md#performance-tuning)
- **...fix a problem** → [Autotuning Guide - Troubleshooting](AUTOTUNING_GUIDE.md#troubleshooting)
- **...see benchmark data** → [Benchmark Results](BENCHMARK_RESULTS.md)
- **...look up API syntax** → [Quick Reference Card](COMPILE_TIME_SAFETY_QUICK_REFERENCE.md)

### "I need to know about..."

- **Block configurations** → [Autotuning Guide - Configuration Format](AUTOTUNING_GUIDE.md#configuration-format)
- **Caching system** → [Autotuning Guide - Three-Tier Caching](AUTOTUNING_GUIDE.md#three-tier-caching)
- **Kernel traits** → [Autotuning Guide - Kernel Traits Requirements](AUTOTUNING_GUIDE.md#kernel-traits-requirements)
- **TuningOptions** → [Autotuning Guide - API Reference](AUTOTUNING_GUIDE.md#tuningoptions)
- **Multi-GPU** → [Autotuning Guide - Multi-GPU Considerations](AUTOTUNING_GUIDE.md#multi-gpu-considerations)
- **Production deployment** → [Autotuning Guide - Embedded Cache](AUTOTUNING_GUIDE.md#embedded-cache-initialization)

---

## 📖 Historical Context

The `archive/` directory contains development documents from the framework's evolution:

- **Migration Reports**: Transition from old to new autotuning system
- **Phase Reports**: Implementation progress tracking
- **Implementation Notes**: Design decisions and refactoring details
- **Bug Reports**: Issues discovered and resolved during development

These are preserved for reference but not needed for using the framework.

---

## 🤝 Contributing

When updating documentation:

1. **Main changes go in AUTOTUNING_GUIDE.md** - Keep it as the single comprehensive source
2. **Keep AUTOTUNING_SUMMARY.md brief** - High-level overview only
3. **Update Quick Reference Card** - For API changes
4. **Add benchmark data** - To BENCHMARK_RESULTS.md
5. **Update this README** - If organization changes

---

## 📝 License

See [LICENSE](../LICENSE) for project licensing information.

---

**Questions?** See the [Troubleshooting](AUTOTUNING_GUIDE.md#troubleshooting) section or check the [examples](../examples/).
