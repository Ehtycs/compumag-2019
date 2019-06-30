# Npyfem

Simple finite element solver toolbox to be used in elmech and to learn how to
use numpy/scipy/jupyter

Project structure proposed here
http://as.ynchrono.us/2007/12/filesystem-structure-of-python-project_21.html

- npyfem ( referred to as "project folder/<project>")
  - README.md
  - tests
    - <unit tests (some day)>
  - run_tests.sh
  - resources
    - <SMALL meshfiles/geofiles etc. for testing/demo purposes>
  - docs
    - <comprehensive documentation!>
  - npyfem
    - <source files>
  - examples
    - <computational examples>


## Howto start (development phase):
- Install python3 in your preferred way
- Install dependencies: meshio, scipy, numpy, pygmsh
- Add project folder path to PYTHONPATH environment variable
  in order to import npyfem from arbitrary locations
- Create a script and import what you want
```
from npyfem.npmesh import NPMesh
# etc...
```

Unit tests (hopefully some day) are in "<project>/tests" folder. All tests can be run using
"<project>/run_tests.sh" or just "python3 -m unittest discover -s tests"
in "<project>/" folder.
