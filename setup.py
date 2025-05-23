from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="melee_rl",
        version="0.0.1",
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        description="A Super Smash Bros. Melee reinforcement learning environment",
        python_requires=">=3.7",
    )