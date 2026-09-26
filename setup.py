import re
from pathlib import Path

from setuptools import setup
from setuptools_rust import Binding, RustExtension

version = re.search(r'^version = "(.+)"', Path("Cargo.toml").read_text(), re.M)[1]

setup(
    version=version,
    packages=["butler_portugal"],
    package_dir={"": "python"},
    rust_extensions=[
        RustExtension(
            "butler_portugal.butler_portugal",
            binding=Binding.NoBinding,
            py_limited_api=True,
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    zip_safe=False,
)
