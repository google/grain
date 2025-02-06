load("@rules_python//python:pip.bzl", "compile_pip_requirements")

py_library(
    name = "setup",
    srcs = ["setup.py"],
    srcs_version = "PY3",
)

compile_pip_requirements(
    name = "requirements",
    requirements_in = "test_requirements.in",
    requirements_txt = "test_requirements_lock.txt",
)
