from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Define and compile soccer_env.cpp
soccer_extension = Pybind11Extension(
    name="SoccerAI.envs.soccer_envs.soccer_sim",
    sources=["src/envs/soccer_envs/soccer_env.cpp"],
    cxx_std=17, # Automatically injects -std=c++17
    extra_compile_args=["-O3"], # Maximizes math performance
)

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[soccer_extension],
    cmdclass={"build_ext": build_ext}, # Connects pybind11's smart compilation checker
)