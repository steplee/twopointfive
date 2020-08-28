from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='pymeshedup',
        ext_modules=[cpp_extension.CppExtension(
            'pycmeshedup',
            ['src/binding.cc', 'src/mc.cc'],
            include_dirs=['/usr/local/include/eigen3'],
            #extra_cflags=['-O3'],
            )],
        cmdclass={'build_ext': cpp_extension.BuildExtension})
