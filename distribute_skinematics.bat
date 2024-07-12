REM cd scikit-kinematics
REM nvim CHANGES.txt README.md pyproject.toml docs\conf.py docs\index.rst skinematics\__init__.py skinematics\sensors\__init__.py
del dist\*
poetry build

REM python setup.py sdist bdist_egg bdist_wininst upload
REM python setup.py sdist bdist_wheel --universal
REM twine upload dist/* --verbose
REM cd ..
