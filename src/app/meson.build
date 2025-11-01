# app_sources = files('main.cpp', 'cli_parser.cpp')
app_sources = files('main.cpp')

executable(
  'hip-img-fx',
  app_sources,
  include_directories: [include_directories('../core', '/opt/rocm/include')],
  link_with: core_lib
)
