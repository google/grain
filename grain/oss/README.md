# Steps to build a new grain pip package

1. Update the version number in setup.py

2. In workspace, run

```
./grain/oss/runner.sh
```

3. Wheels are in `/tmp/grain/all_dist`.

4. Upload to PyPI:

```
python3 -m pip install --upgrade twine
python3 -m twine upload /tmp/grain/all_dist/*-any.whl
```

Authenticate with Twine by following https://pypi.org/help/#apitoken and editing
your `~/.pypirc`.

5. Draft the new release in github: https://github.com/google/grain/releases. Tag the release commit with the version number.