core_sources = files(
  'gpu_utils.cpp',
  'image.cpp',
  'filters/grayscale.hip.cpp',
  'filters/negative.hip.cpp',
)

core_lib = static_library(
  'hipimgfx_core',
  core_sources,
  include_directories: include_directories('.', 'vendor/', '/opt/rocm/include'),
  cpp_args: ['-O2']
)
