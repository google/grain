# Steps to make pip package

```
copybara third_party/py/grain/oss/copy.bara.sky local .. --folder-dir=$HOME/tmp/grain --ignore-noop
#docker pull tensorflow/tensorflow:custom-op-ubuntu16
docker build 
docker run -it -v /tmp/grain:/tmp/grain tensorflow/tensorflow:custom-op-ubuntu16 /bin/bash
bazel build --crosstool_top=@sigbuild-r2.9-python3.9_config_cuda//crosstool:toolchain ...
```