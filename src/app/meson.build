app_sources = files(
  'main.cpp',
  'cli_parser.cpp',
  '../core/gpu_utils.cpp',
  '../core/image.cpp',
  '../core/filters/grayscale.hip.cpp',
  '../core/filters/negative.hip.cpp',
  '../core/filters/gaussian_blur.hip.cpp',
)

executable(
  'hip-img-fx',
  app_sources,
  include_directories: [include_directories('../core', '/opt/rocm/include')],
  cpp_args: ['-O2']
)