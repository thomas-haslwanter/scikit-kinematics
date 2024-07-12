cd scikit-kinematics
REM nvim CHANGES.txt README.md pyproject.toml docs\conf.py docs\index.rst skinematics\__init__.py skinematics\sensors\__init__.py
del dist\*
python setup.py sdist bdist_egg bdist_wininst upload
python setup.py sdist bdist_wheel --universal
twine upload dist/* --verbose
cd ..
