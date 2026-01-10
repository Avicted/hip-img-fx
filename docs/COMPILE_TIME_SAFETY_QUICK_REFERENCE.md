# Compile-Time Safety Quick Reference

**Last Updated**: January 10, 2026

---

## 🚀 Quick Start

### Validating Your KernelTraits

Add this line after your trait definition:

```cpp
struct MyKernelTraits {
    // ... implementation
};

VALIDATE_KERNEL_TRAITS(MyKernelTraits);  // ← Add this line
```

**That's it!** Compile errors will now catch trait violations at build time.

---

## ✅ Required Trait Methods

```cpp
struct MyKernelTraits {
    // 1. Unique kernel identifier (NEVER change after deployment)
    static constexpr const char* name() { return "my_kernel_v1"; }
    
    // 2. Type-safe kernel arguments
    struct Args {
        const void* input;
        void* output;
        size_t size;
    };
    
    // 3. Cache context for workload-specific tuning
    struct Context {
        size_t workload_size;
        
        // Must be const and deterministic!
        std::string cache_key() const {
            if (workload_size < 1024) return "small";
            return "large";
        }
    };
    
    // 4. Generate candidate configurations (must return ≥1 config)
    static std::vector<TuningConfig> generate_candidates() {
        TuningConfig cfg;
        cfg.set("block_x", 256);
        cfg.set("block_y", 1);
        return {cfg};  // Non-empty!
    }
    
    // 5. Validate configuration
    static bool is_valid_config(const TuningConfig& cfg, const Args& args) {
        int threads = cfg.block_x() * cfg.block_y();
        return threads >= 64 && threads <= 1024;
    }
    
    // 6. Launch kernel
    static void launch(const TuningConfig& cfg, const Args& args, hipStream_t stream) {
        // ... kernel launch code
    }
};

VALIDATE_KERNEL_TRAITS(MyKernelTraits);
```

---

## 🛑 Common Mistakes

### ❌ Mistake 1: Non-static data member

```cpp
struct BrokenTraits {
    int state = 0;  // ❌ ERROR: Not stateless!
    // ...
};
```

**Error**: "KernelTraits must be stateless"  
**Fix**: Remove all data members, make all methods `static`

---

### ❌ Mistake 2: Missing cache_key()

```cpp
struct Context {
    size_t size;
    // ❌ ERROR: Missing cache_key() method
};
```

**Error**: "Context must have cache_key() const method"  
**Fix**: Add `std::string cache_key() const { return "..."; }`

---

### ❌ Mistake 3: Empty candidate list

```cpp
static std::vector<TuningConfig> generate_candidates() {
    return {};  // ❌ ERROR: Empty list (runtime error in debug builds)
}
```

**Error**: "Empty candidate list - invariant violation (INV-2)"  
**Fix**: Return at least one configuration

---

### ❌ Mistake 4: All candidates invalid

```cpp
static std::vector<TuningConfig> generate_candidates() {
    TuningConfig cfg;
    cfg.set("block_x", 999);  // Invalid!
    return {cfg};
}

static bool is_valid_config(const TuningConfig& cfg, const Args&) {
    return cfg.block_x() % 64 == 0;  // 999 % 64 != 0
}
```

**Error**: "All candidates failed validation"  
**Fix**: Ensure generated configs pass `is_valid_config()`

---

## ⚡ Skip Autotuning (For Simple Kernels)

### Option 1: Explicit flag

```cpp
struct SimpleKernelTraits {
    static constexpr bool autotune_needed = false;  // ← Skip autotuning
    
    static TuningConfig default_config() {  // ← Provide default
        TuningConfig cfg;
        cfg.set("block_x", 256);
        cfg.set("block_y", 1);
        return cfg;
    }
    
    // ... rest of implementation (still required)
};
```

**When to use**:
- ✅ Trivial operations (memcpy, fill, simple math)
- ✅ Known optimal config (from theory/analysis)
- ✅ Runtime < 50µs

---

### Option 2: Runtime heuristic (automatic)

If you don't set `autotune_needed`, the framework automatically skips tuning for:
- Workloads < 64 KB
- Estimated runtime < 50µs

**No code changes needed!**

---

## 📊 Performance Impact

| Feature | Debug Build | Release Build |
|---------|-------------|---------------|
| Compile-time checks | 0.1s compile time | Zero runtime cost |
| Runtime validation | ~1µs per call | **Zero** (elided) |
| Candidate pruning | 10-50ms saved | 10-50ms saved |

---

## 🔍 Debugging Compile Errors

### Error: "no type named 'Args'"

```
error: no type named 'Args' in 'MyKernelTraits'
```

**Fix**: Add `struct Args { ... };` to your traits

---

### Error: "no matching function for call"

```
error: no matching function for call to 'MyKernelTraits::launch(...)'
```

**Fix**: Ensure signature is exactly:
```cpp
static void launch(const TuningConfig& cfg, const Args& args, hipStream_t stream);
```

---

### Error: "static assertion failed: KernelTraits must be stateless"

**Fix**: Remove all non-static data members:

```cpp
// ❌ Before
struct MyTraits {
    int member = 0;  // Remove this
    // ...
};

// ✅ After
struct MyTraits {
    // No data members
    // All methods static
};
```

---

## 📝 Checklist for New Kernels

- [ ] Define `static constexpr const char* name()`
- [ ] Define `struct Args { ... }`
- [ ] Define `struct Context { std::string cache_key() const; }`
- [ ] Implement `generate_candidates()` returning ≥1 config
- [ ] Implement `is_valid_config()` returning bool
- [ ] Implement `launch()` method
- [ ] Add `VALIDATE_KERNEL_TRAITS(MyKernelTraits);`
- [ ] Compile and check for errors
- [ ] (Optional) Set `autotune_needed = false` for simple kernels
- [ ] (Optional) Provide `default_config()` if skipping tuning

---

## 🔗 Full Documentation

See [COMPILE_TIME_SAFETY_ENFORCEMENT.md](COMPILE_TIME_SAFETY_ENFORCEMENT.md) for:
- Detailed concept explanations
- Performance analysis
- Backward compatibility
- Testing guidelines

---

## 💡 Examples

**Full examples**: [`kernel_traits_concepts_example.h`](examples/kernel_traits_concepts_example.h)

**Test suite**: [`test_compile_time_safety.cpp`](../bench/test_compile_time_safety.cpp)

---

## ❓ FAQ

### Q: Do I need to update existing kernels?

**A**: No! Existing kernels work without modification. Add validation incrementally.

---

### Q: What if I need mutable state?

**A**: Traits should be stateless. Move state to `Args` or `Context`:

```cpp
// ❌ Don't do this
struct BadTraits {
    int counter = 0;  // Mutable state
};

// ✅ Do this instead
struct GoodTraits {
    struct Args {
        int counter;  // State lives in Args
    };
};
```

---

### Q: How do I test my traits?

```cpp
// Add this to your test file
#include "src/core/autotune/kernel_traits_concepts.h"

VALIDATE_KERNEL_TRAITS(MyKernelTraits);

void test_my_traits() {
    auto candidates = MyKernelTraits::generate_candidates();
    assert(!candidates.empty());  // Non-empty
    
    MyKernelTraits::Args args = {...};
    for (const auto& cfg : candidates) {
        assert(MyKernelTraits::is_valid_config(cfg, args));  // All valid
    }
}
```

---

### Q: What's the performance overhead?

**A**: **Zero in release builds!** All checks are:
- Compile-time (no runtime cost), or
- Debug-only (removed in release builds)

---

### Q: Can I disable validation?

**A**: Remove the `VALIDATE_KERNEL_TRAITS()` line, but **not recommended**.
Compile-time checks catch bugs early!

---

## 🎯 Summary

1. **Add one line**: `VALIDATE_KERNEL_TRAITS(MyKernelTraits);`
2. **Fix compile errors** (clear messages tell you what's wrong)
3. **Deploy with confidence** (zero runtime overhead)

**Questions?** Check [COMPILE_TIME_SAFETY_ENFORCEMENT.md](COMPILE_TIME_SAFETY_ENFORCEMENT.md)
