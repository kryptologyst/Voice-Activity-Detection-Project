"""Setup script for Voice Activity Detection package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vad-project",
    version="1.0.0",
    author="VAD Project Team",
    author_email="vad-project@example.com",
    description="A modern implementation of Voice Activity Detection for research and education",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vad-project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Education",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
            "pre-commit>=3.3.0",
        ],
        "demo": [
            "streamlit>=1.25.0",
            "gradio>=3.40.0",
        ],
        "tracking": [
            "wandb>=0.15.0",
            "mlflow>=2.5.0",
        ],
        "serving": [
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vad-train=scripts.train:main",
            "vad-evaluate=scripts.evaluate:main",
            "vad-demo=streamlit run demo/streamlit_app.py",
        ],
    },
    include_package_data=True,
    package_data={
        "vad": ["configs/*.yaml", "configs/**/*.yaml"],
    },
    keywords=[
        "voice activity detection",
        "speech processing",
        "audio analysis",
        "machine learning",
        "pytorch",
        "research",
        "education",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/vad-project/issues",
        "Source": "https://github.com/yourusername/vad-project",
        "Documentation": "https://github.com/yourusername/vad-project#readme",
    },
)
