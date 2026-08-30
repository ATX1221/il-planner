import setuptools

setuptools.setup(
    name="il-planner",
    version="0.1.0",
    description="A from-scratch imitation-learning planner for the nuPlan benchmark.",
    python_requires=">=3.9",
    packages=setuptools.find_packages(include=["il_planner", "il_planner.*"]),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
)
