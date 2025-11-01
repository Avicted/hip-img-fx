project('hip-img-fx', 'cpp',
  version: '0.1.0',
  default_options: ['cpp_std=c++20', 'buildtype=debugoptimized']
)

add_project_arguments('-D__HIP_PLATFORM_AMD__', language: 'cpp')

subdir('src/core')
subdir('src/app')
# subdir('tests')
