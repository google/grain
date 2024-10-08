py_library(
    name = "setup",
    srcs = ["setup.py"],
    srcs_version = "PY3",
)

load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@rules_python//python:packaging.bzl", "py_wheel", 'py_package')

compile_pip_requirements(
    name = "requirements",
    src = "requirements.in",
    requirements_txt = "requirements_lock.txt",
    requirements_darwin = "requirements_darwin.txt",
    requirements_linux = "requirements_linux.txt",
)

py_package(
    name="grain_pkg",
    packages=["grain"],
    deps=["//grain:core", "//grain:python", "//grain:python_experimental", "//grain:python_lazy_dataset"]
)

py_wheel(
    name = "grain_whl",
    distribution = "grain",
    version = "0.2.2",
    platform = select({
        "@platforms//os:macos": "macosx_14_0_arm64",
        "@platforms//os:linux": "manylinux2014_x86_64",
    }),
    deps = [
        "grain_pkg"
    ],
)
