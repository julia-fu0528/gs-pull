```python 
pixi init

pixi install
pixi run install

# Optional additional installs
pixi run install_splat
pixi run install_neuralpull
pixi run install_udf

```

export PYTHONNOUSERSITE=1
unset PYTHONPATH
pixi run python -m pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4' --no-build-isolation