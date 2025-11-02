project('hip-img-fx', 'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++20',
    'b_lto=true',
    'buildtype=release',
  ]
)

add_project_arguments('-D__HIP_PLATFORM_AMD__', language: 'cpp')

app_sources = files(
  'src/main.cpp',
  'src/cli_parser.cpp',
  'src/process.cpp',
  'src/gpu_utils.cpp',
  'src/image.cpp',
  'src/filters/grayscale.hip.cpp',
  'src/filters/negative.hip.cpp',
  'src/filters/gaussian_blur.hip.cpp',
)


gpu_arch = '--offload-arch=gfx1030'

executable(
  'hip-img-fx',
  app_sources,
  cpp_args: ['-fopenmp', '-march=native', '-ffast-math', gpu_arch],
  link_args: ['-fopenmp', '-march=native', '-ffast-math', gpu_arch],
  include_directories: [include_directories('.', 'vendor', '/opt/rocm/include')],
  install: true,
)
