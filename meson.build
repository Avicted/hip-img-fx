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
# TESTS
# ============================================================================

gtest_dep = dependency('gtest', main: false, required: false)

if gtest_dep.found()
  test_src = files(
    'tests/test_main.cpp',
    'tests/test_helpers.cpp',
    'tests/test_filters_cpu.cpp',
    'tests/test_filters_gpu.cpp',
    'tests/test_image_io.cpp',
    'tests/test_cli_parser.cpp',
    'tests/test_autotune_config.cpp',
    'tests/test_cache_store.cpp',
    'tests/test_gpu_integration.cpp',
    'tests/test_image_formats.cpp',
    'tests/test_gpu_utils.cpp',
    'tests/test_autotune_extended.cpp',
    'tests/test_process.cpp',
    'tests/test_image_edge_cases.cpp',
    'tests/test_gpu_utils_extended.cpp',
    'tests/test_process_extended.cpp',
    'tests/test_helpers_extended.cpp',
    'tests/test_filters_cpu_extended.cpp',
    'tests/test_tuning_config_extended.cpp',
    'tests/test_cli_parser_extended.cpp',
    'tests/test_filters_traits.cpp',
    'tests/test_generate_candidates.cpp',
    'tests/test_process_errors.cpp',
    'tests/test_main_app.cpp',
    'tests/test_tuning_options_spec.cpp',
    'tests/test_tuning_config_spec.cpp',
    'tests/test_cache_store_spec.cpp',
    'tests/test_hip_event_spec.cpp',
    'tests/test_benchmarker_spec.cpp',
    'tests/test_orchestrator_spec.cpp',
    'tests/test_kernel_traits_concepts_spec.cpp',
    'tests/test_benchmark_runner.cpp',
    
    # Include source files needed for tests
    'src/core/gpu_utils.cpp',
    'src/core/image.cpp',
    'src/core/autotune/tuning_config.cpp',
    'src/core/autotune/cache_store.cpp',
    'src/cli/cli_parser.cpp',
    'src/app/process.cpp',
    'src/app/main.cpp',  # Include main.cpp for app_main() tests
    'src/filters/grayscale.hip.cpp',
    'src/filters/grayscale_autotune.hip.cpp',
    'src/filters/negative.hip.cpp',
    'src/filters/negative_autotune.hip.cpp',
    'src/filters/gaussian_blur.hip.cpp',
    'src/filters/gaussian_blur_autotune.hip.cpp',
    'bench/run_bench.cpp',
  )
  
  # Determine test build arguments (add coverage if enabled)
  test_cpp_args = app_warning_args + ['-fopenmp', '-march=native', gpu_arch, '-DBUILDING_TESTS']
  test_link_args = ['-fopenmp', '-march=native', gpu_arch]
  
  if get_option('b_coverage')
    test_cpp_args += ['--coverage']
    test_link_args += ['--coverage']
  endif
  
  test_exe = executable(
    'hip-img-fx-tests',
    test_src,
    cpp_args: test_cpp_args,
    link_args: test_link_args,
    include_directories: [inc_src, inc_sys],
    dependencies: [gtest_dep],
    install: false,
  )
  
  # Register with Meson test framework
  test('unit_tests', test_exe, timeout: 120)
  
  # CPU-only tests (for CI without GPU)
  test('unit_tests_cpu_only', test_exe, 
       args: ['--gtest_filter=-*GPU*'],
       timeout: 60,
       suite: 'cpu')
  
else
  warning('GoogleTest not found - tests will not be built. Install with: sudo apt-get install libgtest-dev')
endif

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
