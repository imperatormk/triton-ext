"""Build metal_utils ObjC++ extension at install time."""
import os
import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class MetalBuildExt(build_ext):
    """Custom build_ext that compiles ObjC++ with Metal + libtorch flags."""

    def build_extension(self, ext):
        import torch
        torch_dir = os.path.dirname(torch.__file__)
        torch_lib = os.path.join(torch_dir, "lib")
        torch_inc = os.path.join(torch_dir, "include")

        ext.include_dirs.append(torch_inc)
        ext.include_dirs.append(
            os.path.join(torch_inc, "torch", "csrc", "api", "include"))
        ext.library_dirs.append(torch_lib)
        ext.libraries.extend(["torch", "c10", "torch_python"])
        ext.extra_link_args.append(f"-Wl,-rpath,{torch_lib}")

        super().build_extension(ext)


metal_utils = Extension(
    name="triton_apple_backend.metal_utils",
    sources=[os.path.join("triton_apple_backend", "metal_utils.m")],
    language="objc++",
    extra_compile_args=[
        "-ObjC++",
        "-std=c++17",
        "-O3",
        "-Wno-deprecated-declarations",
    ],
    extra_link_args=[
        "-framework",
        "Metal",
        "-framework",
        "Foundation",
        "-undefined",
        "dynamic_lookup",
    ],
    include_dirs=[sysconfig.get_path("include")],
)

setup(
    name="triton-apple-backend",
    version="0.1.0",
    packages=["triton_apple_backend"],
    ext_modules=[metal_utils],
    cmdclass={"build_ext": MetalBuildExt},
    entry_points={
        "triton.backends": ["apple = triton_apple_backend"],
    },
)
