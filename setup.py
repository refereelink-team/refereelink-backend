import pathlib

import setuptools


HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="soccer-analysis",
    version='0.2.0',
    python_requires=">=3.10",
    description="Soccer real-time analysis and assisted officiating system",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/caysonyin/SC",
    author="SC Team",
    license='MIT',
    packages=setuptools.find_packages(include=['app', 'app.*']),
    include_package_data=True,
    install_requires=[
        "supervision",
        "numpy",
        "opencv-python",
        "tqdm",
        "torch",
        "ultralytics",
        "PySide6==6.10.2",
        "fastapi",
        "uvicorn",
        "pydantic",
        "psutil",
    ],
    extras_require={
        'tests': [
            'pytest',
            'pytest-asyncio',
            'httpx',
        ],
        'web': [],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Typing :: Typed',
        'Operating System :: POSIX',
        'Operating System :: Unix',
    ]
)
