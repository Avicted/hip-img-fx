project('hip-img-fx', 'cpp',
  version: '1.0.1',
  default_options: [
    'cpp_std=c++20',
    'b_lto=true',
    'warning_level=2',
    'buildtype=release',
  ]
)

add_project_arguments('-D__HIP_PLATFORM_AMD__', language: 'cpp')
add_project_arguments('-DHIP_IMG_FX_VERSION="' + meson.project_version() + '"', language: 'cpp')

app_warning_args = [
  '-Wall', 
  '-Wextra', 
  '-Wpedantic',
  '-Wno-system-headers',
]

app_src = files(
  'src/app/main.cpp',
  'src/app/process.cpp',
  'src/cli/cli_parser.cpp',
  'src/core/gpu_utils.cpp',
  'src/core/image.cpp',
  'src/core/autotune/tuning_config.cpp',
  'src/core/autotune/cache_store.cpp',
  'src/filters/grayscale.hip.cpp',
  'src/filters/grayscale_autotune.hip.cpp',
  'src/filters/negative.hip.cpp',
  'src/filters/negative_autotune.hip.cpp',
  'src/filters/gaussian_blur.hip.cpp',
  'src/filters/gaussian_blur_autotune.hip.cpp',
)

# RX 6900 XT; change as needed
gpu_arch = '--offload-arch=gfx1030' 

inc_src = include_directories('.', 'include')
inc_sys = include_directories(['vendor', '/opt/rocm/include'], is_system: true)

executable(
  'hip-img-fx',
  app_src,
  cpp_args: app_warning_args + ['-fopenmp', '-march=native', '-ffast-math', gpu_arch],
  link_args: ['-fopenmp', '-march=native', '-ffast-math', gpu_arch],
  include_directories: [inc_src, inc_sys],
  install: true,
)

bench_src = files(
  'bench/run_bench.cpp',
  'src/core/gpu_utils.cpp',
  'src/core/image.cpp',
  'src/core/autotune/tuning_config.cpp',
  'src/core/autotune/cache_store.cpp',
  'src/filters/grayscale.hip.cpp',
  'src/filters/grayscale_autotune.hip.cpp',
  'src/filters/negative.hip.cpp',
  'src/filters/negative_autotune.hip.cpp',
  'src/filters/gaussian_blur.hip.cpp',
  'src/filters/gaussian_blur_autotune.hip.cpp',
)

executable(
  'hip-img-fx-bench',
  bench_src,
  cpp_args: app_warning_args + ['-fopenmp', '-march=native', '-ffast-math', gpu_arch],
  link_args: ['-fopenmp', '-march=native', '-ffast-math', gpu_arch],
  include_directories: [inc_src, inc_sys],
  install: false,
)

# ============================================================================
# PUBLIC API INSTALLATION
# ============================================================================

# Install public headers
install_headers(
  'include/hip-img-fx/autotune/orchestrator.h',
  'include/hip-img-fx/autotune/tuning_config.h',
  'include/hip-img-fx/autotune/types.h',
  'include/hip-img-fx/autotune/kernel_traits_concepts.h',
  'include/hip-img-fx/autotune/benchmarker.h',
  'include/hip-img-fx/autotune/cache_store.h',
  'include/hip-img-fx/autotune/embedded_cache.h',
  'include/hip-img-fx/autotune/hip_event.h',
  subdir: 'hip-img-fx/autotune'
)

# Generate pkg-config file
pkg = import('pkgconfig')
pkg.generate(
  name: 'hip-img-fx',
  description: 'HIP autotuning framework for GPU kernels',
  version: meson.project_version(),
  subdirs: 'hip-img-fx',
  # Note: HIP is provided by system ROCm installation
)
