# Autotuning Framework - Quick Reference

## Adding a New Kernel to Autotuning

### Template Code

```cpp
// In filters.h

struct MyKernelTraits {
    // 1. Kernel name (unique identifier)
    static constexpr const char* name() { return "my_kernel"; }
    
    // 2. Kernel arguments (type-safe)
    struct Args {
        // Define your kernel parameters here
        const unsigned char* input;
        unsigned char* output;
        // ... other args ...
    };
    
    // 3. Cache context (for different workload categories)
    struct Context {
        // Define context fields (e.g., size, parameters)
        size_t data_size;
        
        std::string cache_key() const {
            // Return category string for caching
            if (data_size < SMALL_THRESHOLD) return "small";
            if (data_size < LARGE_THRESHOLD) return "medium";
            return "large";
        }
    };
    
    // 4. Generate candidate configurations
    static std::vector<imgfx::core::autotune::TuningConfig> generate_candidates() {
        using imgfx::core::autotune::TuningConfig;
        std::vector<TuningConfig> configs;
        
        // Example: Test different block sizes
        for (int bx : {64, 128, 256}) {
            TuningConfig cfg;
            cfg.set("block_x", bx);
            cfg.set("block_y", 1);
            configs.push_back(cfg);
        }
        
        // Add more candidates as needed
        return configs;
    }
    
    // 5. Validate configuration
    static bool is_valid_config(
        const imgfx::core::autotune::TuningConfig& cfg,
        const Args& args)
    {
        int threads = cfg.block_x() * cfg.block_y();
        
        // Example constraints
        if (threads % 64 != 0) return false;  // Wavefront alignment
        if (threads < 64 || threads > 1024) return false;  // Reasonable limits
        
        return true;
    }
    
    // 6. Launch kernel
    static void launch(
        const imgfx::core::autotune::TuningConfig& cfg,
        const Args& args,
        hipStream_t stream)
    {
        // Calculate grid dimensions
        int threads_per_block = cfg.block_x() * cfg.block_y();
        int blocks = (args.work_size + threads_per_block - 1) / threads_per_block;
        
        dim3 block_dim(cfg.block_x(), cfg.block_y(), 1);
        dim3 grid_dim(blocks, 1, 1);
        
        // Launch your kernel
        hipLaunchKernelGGL(
            my_kernel,
            grid_dim, block_dim,
            0,  // shared memory
            stream,
            args.input, args.output /* ... */);
    }
};

// In my_kernel_autotune.hip.cpp

void apply_my_kernel_autotuned(
    /* your parameters */,
    hipStream_t stream)
{
    using namespace imgfx::core::autotune;
    
    // Static orchestrator (initialized once per kernel)
    static TuningOrchestrator<MyKernelTraits> orchestrator;
    
    // Prepare arguments
    MyKernelTraits::Args args{ /* ... */ };
    MyKernelTraits::Context ctx{ /* ... */ };
    
    // Execute with autotuning
    orchestrator.execute(args, ctx, stream);
}
```

---

## Common Patterns

### Pattern 1: Size-Based Context

```cpp
struct Context {
    size_t bytes;
    
    std::string cache_key() const {
        if (bytes < 1024 * 1024) return "small";      // < 1MB
        if (bytes < 10 * 1024 * 1024) return "medium"; // 1-10MB
        return "large";                                // > 10MB
    }
};
```

### Pattern 2: Parameter-Based Context

```cpp
struct Context {
    int blur_radius;
    
    std::string cache_key() const {
        // Different configs for different blur amounts
        return "blur_" + std::to_string(blur_radius);
    }
};
```

### Pattern 3: Combined Context

```cpp
struct Context {
    size_t bytes;
    int param;
    
    std::string cache_key() const {
        std::string size = (bytes < 1024*1024) ? "small" : "large";
        return size + "_param" + std::to_string(param);
    }
};
```

### Pattern 4: No Context (Same Config for All)

```cpp
struct Context {
    std::string cache_key() const {
        return "default";  // Single config for all cases
    }
};
```

---

## Adding Tunable Parameters

### Example: Vectorization

```cpp
static std::vector<TuningConfig> generate_candidates() {
    std::vector<TuningConfig> configs;
    
    for (int bx : {64, 128, 256}) {
        for (int vec : {1, 2, 4}) {  // NEW: vectorization factor
            TuningConfig cfg;
            cfg.set("block_x", bx);
            cfg.set("block_y", 1);
            cfg.set("vec_width", vec);  // NEW parameter
            configs.push_back(cfg);
        }
    }
    
    return configs;
}

static void launch(const TuningConfig& cfg, const Args& args, hipStream_t stream) {
    int vec = cfg.get_or("vec_width", 1);
    
    // Dispatch to specialized kernel
    if (vec == 4) {
        hipLaunchKernelGGL(my_kernel_vec4, ...);
    } else if (vec == 2) {
        hipLaunchKernelGGL(my_kernel_vec2, ...);
    } else {
        hipLaunchKernelGGL(my_kernel, ...);
    }
}
```

### Example: Shared Memory Tuning

```cpp
static std::vector<TuningConfig> generate_candidates() {
    std::vector<TuningConfig> configs;
    
    for (int bx : {128, 256}) {
        for (int smem_tiles : {0, 1, 2}) {  // NEW: shared memory tiles
            TuningConfig cfg;
            cfg.set("block_x", bx);
            cfg.set("block_y", 1);
            cfg.set("smem_tiles", smem_tiles);  // NEW parameter
            configs.push_back(cfg);
        }
    }
    
    return configs;
}

static void launch(const TuningConfig& cfg, const Args& args, hipStream_t stream) {
    int smem_tiles = cfg.get_or("smem_tiles", 0);
    size_t shared_bytes = smem_tiles * cfg.block_x() * sizeof(float);
    
    hipLaunchKernelGGL(
        my_kernel,
        grid, block,
        shared_bytes,  // Dynamic shared memory
        stream,
        args.input, args.output, smem_tiles);
}
```

### Example: Loop Unrolling

```cpp
static std::vector<TuningConfig> generate_candidates() {
    std::vector<TuningConfig> configs;
    
    for (int bx : {128, 256}) {
        for (int unroll : {1, 2, 4, 8}) {  // NEW: unroll factor
            TuningConfig cfg;
            cfg.set("block_x", bx);
            cfg.set("unroll_factor", unroll);  // NEW parameter
            configs.push_back(cfg);
        }
    }
    
    return configs;
}

// Note: Unrolling typically requires compile-time templating
// Launch would dispatch to pre-compiled template specializations
```

---

## Validation Patterns

### Pattern 1: Wavefront Alignment (AMD)

```cpp
static bool is_valid_config(const TuningConfig& cfg, const Args& args) {
    int threads = cfg.total_threads();
    return threads % 64 == 0;  // Must be multiple of wavefront size
}
```

### Pattern 2: Thread Count Limits

```cpp
static bool is_valid_config(const TuningConfig& cfg, const Args& args) {
    int threads = cfg.total_threads();
    return threads >= 64 && threads <= 1024;
}
```

### Pattern 3: Shared Memory Limits

```cpp
static bool is_valid_config(const TuningConfig& cfg, const Args& args) {
    size_t smem = cfg.get_or("smem_tiles", 0) * cfg.block_x() * sizeof(float);
    return smem <= 65536;  // 64KB shared memory limit
}
```

### Pattern 4: Work-Size Dependent

```cpp
static bool is_valid_config(const TuningConfig& cfg, const Args& args) {
    int threads = cfg.total_threads();
    // Don't waste threads on small workloads
    return threads <= args.work_size;
}
```

### Pattern 5: Parameter Dependencies

```cpp
static bool is_valid_config(const TuningConfig& cfg, const Args& args) {
    int threads = cfg.total_threads();
    int unroll = cfg.get_or("unroll_factor", 1);
    
    // High unroll requires more registers, limit threads
    if (unroll >= 4 && threads > 256) return false;
    
    return true;
}
```

---

## Grid Calculation Patterns

### Pattern 1: 1D Linear (Simple)

```cpp
static void launch(const TuningConfig& cfg, const Args& args, hipStream_t stream) {
    int threads_per_block = cfg.block_x();
    int blocks = (args.num_elements + threads_per_block - 1) / threads_per_block;
    
    dim3 block_dim(threads_per_block, 1, 1);
    dim3 grid_dim(blocks, 1, 1);
    
    hipLaunchKernelGGL(my_kernel, grid_dim, block_dim, 0, stream, ...);
}
```

### Pattern 2: 2D Grid (Image Processing)

```cpp
static void launch(const TuningConfig& cfg, const Args& args, hipStream_t stream) {
    int blocks_x = (args.width + cfg.block_x() - 1) / cfg.block_x();
    int blocks_y = (args.height + cfg.block_y() - 1) / cfg.block_y();
    
    dim3 block_dim(cfg.block_x(), cfg.block_y(), 1);
    dim3 grid_dim(blocks_x, blocks_y, 1);
    
    hipLaunchKernelGGL(my_kernel, grid_dim, block_dim, 0, stream, ...);
}
```

### Pattern 3: Batched Processing

```cpp
static void launch(const TuningConfig& cfg, const Args& args, hipStream_t stream) {
    int threads_per_block = cfg.total_threads();
    int blocks_per_batch = (args.batch_size + threads_per_block - 1) / threads_per_block;
    
    dim3 block_dim(cfg.block_x(), cfg.block_y(), 1);
    dim3 grid_dim(blocks_per_batch, args.num_batches, 1);
    
    hipLaunchKernelGGL(my_kernel, grid_dim, block_dim, 0, stream, ...);
}
```

### Pattern 4: Work-Per-Thread Adjustment

```cpp
static void launch(const TuningConfig& cfg, const Args& args, hipStream_t stream) {
    int work_per_thread = cfg.get_or("work_per_thread", 1);
    int threads_per_block = cfg.total_threads();
    
    // Each thread processes multiple elements
    int total_work_items = (args.num_elements + work_per_thread - 1) / work_per_thread;
    int blocks = (total_work_items + threads_per_block - 1) / threads_per_block;
    
    dim3 block_dim(cfg.block_x(), cfg.block_y(), 1);
    dim3 grid_dim(blocks, 1, 1);
    
    hipLaunchKernelGGL(my_kernel, grid_dim, block_dim, 0, stream, 
                       args.data, work_per_thread);
}
```

---

## Debugging Tips

### Print Configuration During Tuning

```cpp
static void launch(const TuningConfig& cfg, const Args& args, hipStream_t stream) {
    #ifdef DEBUG_AUTOTUNING
    printf("Launching with config: %s\n", cfg.to_string().c_str());
    #endif
    
    // ... launch code ...
}
```

### Force Re-tuning

```cpp
TuningOptions opts = TuningOptions::defaults();
opts.force_retune = true;  // Ignore cache
orchestrator.execute(args, ctx, stream, opts);
```

### Check Cache Hit

```cpp
if (orchestrator.has_cached_config(ctx)) {
    printf("Using cached config\n");
} else {
    printf("Will perform autotuning...\n");
}
```

### Manual Config for Testing

```cpp
TuningConfig cfg;
cfg.set("block_x", 128);
cfg.set("block_y", 1);

// Test launch directly
MyKernelTraits::launch(cfg, args, stream);
```

---

## Performance Tips

### Tip 1: Reduce Candidate Count

```cpp
// Too many candidates = slow tuning
// Before: 3 × 3 × 4 = 36 candidates
for (int bx : {64, 128, 256}) {          // 3 options
    for (int by : {1, 2, 4}) {           // 3 options
        for (int vec : {1, 2, 4, 8}) {   // 4 options
            // 36 configs to test!
        }
    }
}

// Better: Prune unlikely combinations
for (int bx : {128, 256}) {              // 2 options
    for (int vec : {1, 4}) {             // 2 options
        // Only 4 configs to test
    }
}
```

### Tip 2: Use Coarse Context

```cpp
// Too fine-grained = cache misses
std::string cache_key() const {
    return std::to_string(bytes);  // ❌ Unique key per size
}

// Better: Categorize into buckets
std::string cache_key() const {
    if (bytes < 1024*1024) return "small";       // ✅ 3 categories
    if (bytes < 10*1024*1024) return "medium";
    return "large";
}
```

### Tip 3: Static Orchestrator

```cpp
// ❌ Don't create new orchestrator each call
void apply_kernel(...) {
    TuningOrchestrator<MyKernelTraits> orchestrator;  // Slow!
    orchestrator.execute(...);
}

// ✅ Use static storage
void apply_kernel(...) {
    static TuningOrchestrator<MyKernelTraits> orchestrator;  // Fast!
    orchestrator.execute(...);
}
```

### Tip 4: Warm-up Before Production

```cpp
// Pre-tune all expected workload sizes at startup
void warmup_autotuning() {
    for (size_t size : {512*1024, 5*1024*1024, 50*1024*1024}) {
        MyKernelTraits::Context ctx{size};
        if (!orchestrator.has_cached_config(ctx)) {
            // Dummy args for tuning
            MyKernelTraits::Args dummy_args = create_dummy(size);
            orchestrator.get_or_tune(dummy_args, ctx);
        }
    }
}
```

---

## Common Mistakes

### ❌ Mistake 1: Forgetting to Return Config

```cpp
static std::vector<TuningConfig> generate_candidates() {
    std::vector<TuningConfig> configs;
    // ... add configs ...
    // FORGOT: return configs;
}
```

### ❌ Mistake 2: Using `get()` Without Checking

```cpp
int vec = cfg.get<int>("vec_width");  // Throws if not set!
```

**Fix:** Use `get_or()` with default:
```cpp
int vec = cfg.get_or("vec_width", 1);  // ✅ Safe
```

### ❌ Mistake 3: Non-deterministic Cache Key

```cpp
std::string cache_key() const {
    return std::to_string(rand());  // ❌ Never caches!
}
```

### ❌ Mistake 4: Ignoring Validation

```cpp
static bool is_valid_config(const TuningConfig& cfg, const Args& args) {
    return true;  // ❌ Accepts invalid configs!
}
```

### ❌ Mistake 5: Wrong Grid Calculation

```cpp
int blocks = args.num_elements / threads_per_block;  // ❌ Loses remainder!
```

**Fix:**
```cpp
int blocks = (args.num_elements + threads_per_block - 1) / threads_per_block;  // ✅
```

---

## Testing Checklist

- [ ] Kernel produces correct output with default config
- [ ] All candidate configs pass validation
- [ ] Kernel produces identical output for all valid configs
- [ ] Cache file is created and persists across runs
- [ ] Second run uses cached config (no re-tuning)
- [ ] Different contexts produce different cache entries
- [ ] Performance improves with tuned config vs default

---

## File Organization

```
src/
├── core/
│   └── autotune/              # Framework (reusable)
│       ├── types.h
│       ├── tuning_config.h/.cpp
│       ├── cache_store.h/.cpp
│       ├── benchmarker.h
│       └── orchestrator.h
└── filters/
    ├── filters.h              # Kernel traits definitions
    ├── my_kernel.hip.cpp      # Kernel implementation
    └── my_kernel_autotune.hip.cpp  # apply_*_autotuned() wrapper
```

---

## Further Reading

- [AUTOTUNING_REFACTOR.md](AUTOTUNING_REFACTOR.md) - Full architectural design
- [AUTOTUNING_BEFORE_AFTER.md](AUTOTUNING_BEFORE_AFTER.md) - Detailed code comparisons
- [examples/](examples/) - Complete example implementations

