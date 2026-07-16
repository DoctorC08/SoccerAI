from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Define and compile soccer_env.cpp
soccer_extension = Pybind11Extension(
    name="src.envs.soccer_envs.soccer_sim",
    sources=["src/envs/soccer_envs/soccer_env.cpp"],
    cxx_std=17, # Automatically injects -std=c++17
    extra_compile_args=["-O3"], # Maximize math performance
)

setup(
    packages=find_packages(include=["src", "src.*"]),
    ext_modules=[soccer_extension],
    cmdclass={"build_ext": build_ext}, 
)
