from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='pymeshedup',
        ext_modules=[cpp_extension.CppExtension(
            'pycmeshedup',
            ['src/binding.cc', 'src/mc.cc'],
            include_dirs=['/usr/local/include/eigen3'],
            extra_compile_args=['-O3'],
            library_dirs=['/usr/lib/x86_64-linux-gnu/'],
            libraries=['GLEW'],
            )],
        cmdclass={'build_ext': cpp_extension.BuildExtension})
