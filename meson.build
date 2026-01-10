project('hip-img-fx', 'cpp',
  version: '0.2.0',
  default_options: [
    'cpp_std=c++20',
    'b_lto=true',
    'warning_level=2',
  ]
)

add_project_arguments('-D__HIP_PLATFORM_AMD__', language: 'cpp')

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

inc_src = include_directories('.')
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

# Validation test for grayscale migration
validate_src = files(
  'bench/validate_grayscale_migration.cpp',
  'src/core/gpu_utils.cpp',
  'src/core/autotune/tuning_config.cpp',
  'src/core/autotune/cache_store.cpp',
  'src/filters/grayscale.hip.cpp',
  'src/filters/grayscale_autotune.hip.cpp',
)

executable(
  'validate-grayscale-migration',
  validate_src,
  cpp_args: app_warning_args + ['-fopenmp', '-march=native', '-ffast-math', gpu_arch],
  link_args: ['-fopenmp', '-march=native', '-ffast-math', gpu_arch],
  include_directories: [inc_src, inc_sys],
  install: false,
)

# Performance benchmark for grayscale migration
benchmark_migration_src = files(
  'bench/benchmark_grayscale_migration.cpp',
  'src/core/gpu_utils.cpp',
  'src/core/autotune/tuning_config.cpp',
  'src/core/autotune/cache_store.cpp',
  'src/filters/grayscale.hip.cpp',
  'src/filters/grayscale_autotune.hip.cpp',
)

executable(
  'benchmark-grayscale-migration',
  benchmark_migration_src,
  cpp_args: app_warning_args + ['-fopenmp', '-march=native', '-ffast-math', gpu_arch],
  link_args: ['-fopenmp', '-march=native', '-ffast-math', gpu_arch],
  include_directories: [inc_src, inc_sys],
  install: false,
)

# Tier-1 improvements integration test
tier1_test_src = files(
  'bench/test_tier1_improvements.cpp',
  'src/core/gpu_utils.cpp',
  'src/core/autotune/tuning_config.cpp',
  'src/core/autotune/cache_store.cpp',
  'src/filters/grayscale.hip.cpp',
  'src/filters/grayscale_autotune.hip.cpp',
  'src/filters/negative.hip.cpp',
  'src/filters/negative_autotune.hip.cpp',
)

executable(
  'test-tier1-improvements',
  tier1_test_src,
  cpp_args: app_warning_args + ['-fopenmp', '-march=native', '-ffast-math', gpu_arch],
  link_args: ['-fopenmp', '-march=native', '-ffast-math', gpu_arch],
  include_directories: [inc_src, inc_sys],
  install: false,
)
