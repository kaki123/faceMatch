

set -ex



nosetests sklearn --exe
conda inspect linkages -p $PREFIX scikit-learn
conda inspect objects -p $PREFIX scikit-learn
exit 0
