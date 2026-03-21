import pathlib

import setuptools

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")


def read_requirements(file_path: pathlib.Path) -> list[str]:
    requirements: list[str] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


base_packages = setuptools.find_packages(
    include=[
        "core",
        "core.*",
        "tracking",
        "tracking.*",
        "offside",
        "offside.*",
        "projection",
        "projection.*",
    ]
)


setuptools.setup(
    name="sports-main",
    version="0.2.0",
    python_requires=">=3.8",
    description="Soccer tracking project with shared core state management.",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=base_packages,
    include_package_data=True,
    install_requires=read_requirements(HERE / "requirements.txt"),
    extras_require={
        "tests": [
            "pytest",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Typing :: Typed",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
)
