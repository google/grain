#!/bin/sh
# build wheel for python version specified in $PYTHON_VERSION

set -e -x

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/grain}"

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

main() {
  # Remove .bazelrc if it already exists
  [ -e .bazelrc ] && rm .bazelrc

  # Only use __init__.py files that are present in the source. Without this,
  # Bazel will create empty __init__.py files for all subdirectories,
  # occasionally(!) overriding the source __init__.py files.
  write_to_bazelrc "build --incompatible_default_to_explicit_init_py"

  # Enable host OS specific configs. For instance, "build:linux" will be used
  # automatically when building on Linux.
  write_to_bazelrc "build --enable_platform_specific_config"
  # Bazel 7.0.0 no longer supports dynamic symbol lookup on macOS. To resolve
  # undefined symbol errors in macOS arm64 builds, explicitly add the necessary
  # linker flags until dependencies are well defined. See
  # https://github.com/bazelbuild/bazel/issues/19730.
  write_to_bazelrc "build:macos --linkopt=-Wl,-undefined,dynamic_lookup"
  write_to_bazelrc "build:macos --host_linkopt=-Wl,-undefined,dynamic_lookup"

  write_to_bazelrc "build --@rules_python//python/config_settings:python_version=${PYTHON_VERSION}"
  # Set platform-wise file extension for extension modules.
  case "$(uname)" in
    CYGWIN*|MINGW*|MSYS_NT*)
      INCLUDE_EXT="*.pyd"
      ;;
    *)
      INCLUDE_EXT="*.so"
      # Also reduce noise during build.
      write_to_bazelrc "build --cxxopt=-Wno-deprecated-declarations --host_cxxopt=-Wno-deprecated-declarations"
      write_to_bazelrc "build --cxxopt=-Wno-parentheses --host_cxxopt=-Wno-parentheses"
      write_to_bazelrc "build --cxxopt=-Wno-sign-compare --host_cxxopt=-Wno-sign-compare"
      ;;
  esac

  write_to_bazelrc "test --@rules_python//python/config_settings:python_version=${PYTHON_VERSION}"
  write_to_bazelrc "test --action_env PYTHON_VERSION=${PYTHON_VERSION}"
  write_to_bazelrc "test --test_timeout=300"

  # Make sure that we use the exact versions of dependencies specified in
  # MODULE.bazel file.
  write_to_bazelrc "common --check_direct_dependencies=error"

  bazel clean
  bazel build ... --action_env PYTHON_BIN_PATH="${PYTHON_BIN}" --action_env MACOSX_DEPLOYMENT_TARGET='11.0'

  if [ "$RUN_TESTS" = true ] ; then
    bazel test --verbose_failures --test_output=errors ... --action_env PYTHON_BIN_PATH="${PYTHON_BIN}"
  fi

  DEST="${OUTPUT_DIR}"'/all_dist'
  mkdir -p "${DEST}"

  printf '=== Destination directory: %s\n' "${DEST}"

  TMPDIR="$(mktemp -d -t tmp.XXXXXXXXXX)"

  printf '%s : "=== Using tmpdir: %s\n' "$(date)" "${TMPDIR}"

  printf "=== Copy grain files\n"

  cp README.md "${TMPDIR}"
  cp setup.py "${TMPDIR}"
  if [ "$IS_NIGHTLY" == true ]; then
    if [ "$(uname)" = "Darwin" ] ; then
      sed -i '' 's/^name = "grain"$/name = "grain-nightly"/' pyproject.toml
    else
      sed -i 's/^name = "grain"$/name = "grain-nightly"/' pyproject.toml
    fi
  fi
  cp pyproject.toml "${TMPDIR}"
  cp LICENSE "${TMPDIR}"
  # rsync on Windows runner can't correctly figure out absolute paths
  # so we use an intermediate folder for rsync execution, and after that
  # copy to the destination folder.
  rsync -avm -L --exclude="__pycache__/*" grain tmp_folder
  rsync -avm -L  --include="${INCLUDE_EXT}" --include="*_pb2.py" \
    --exclude="*.runfiles" --exclude="*_obj" --include="*/" --exclude="*" \
    bazel-bin/grain tmp_folder
  cp -av -L tmp_folder/. "${TMPDIR}"
  rm -r tmp_folder

  previous_wd="$(pwd)"
  cd "${TMPDIR}"
  printf '%s : "=== Building wheel\n' "$(date)"

  if [ "$IS_NIGHTLY" == true ]; then
    WHEEL_BLD_ARGS="egg_info --tag-build=.dev --tag-date"
  fi
  WHEEL_BLD_ARGS="${WHEEL_BLD_ARGS} bdist_wheel --python-tag py3${PYTHON_MINOR_VERSION}"
  if [ "$(uname)" == "Darwin" ]; then
    WHEEL_BLD_ARGS="${WHEEL_BLD_ARGS} --plat-name macosx_11_0_$(uname -m)"
  fi
  "$PYTHON_BIN" setup.py $WHEEL_BLD_ARGS

  if [ -n "${AUDITWHEEL_PLATFORM}" ]; then
    printf '%s : "=== Auditing wheel\n' "$(date)"
    auditwheel repair --plat "${AUDITWHEEL_PLATFORM}" -w dist dist/*.whl
    cp 'dist/'*manylinux*.whl "${DEST}"
  else
    cp 'dist/'*.whl "${DEST}"
  fi

  printf '%s : "=== Listing wheel\n' "$(date)"
  ls -lrt "${DEST}"/*.whl
  cd "${previous_wd}"

  printf '%s : "=== Output wheel file is in: %s\n' "$(date)" "${DEST}"

  $PYTHON_BIN -m pip install ${OUTPUT_DIR}/all_dist/grain*.whl
  $PYTHON_BIN -m pip install jax
  if (( "${PYTHON_MINOR_VERSION}" < 13 )); then
    $PYTHON_BIN -m pip install tensorflow==2.20.0rc0
    $PYTHON_BIN grain/_src/core/smoke_test_with_tf.py
  fi

  pushd "${OUTPUT_DIR}"
  $PYTHON_BIN -m pytest --pyargs grain -k "TreeJaxTest or FirstFitPackIterDatasetTest or JaxImportTest or TFImportTest"
  popd
}

main "$@"