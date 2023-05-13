from setuptools import setup, find_packages

setup(
    name='boston_311',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'tensorflow',
        'scikit_learn',
        'matplotlib'
    ],
    entry_points={
        'console_scripts': [
            #'my_script_name=my_package_name.my_script_file:main'
            # Add any other console scripts you want to install here
        ]
    },
    # Metadata about your project
    author='Alex Alcivar',
    author_email='alex.g.alcivar49@gmail.com',
    description='A package for training machine learning models on Boston 311 data',
    long_description='A package for training machine learning models on Boston 311 data',
    url='https://github.com/mindfulcoder49/Boston_311',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
