#!/bin/bash
#PBS -N mnist_conv_metacentrum
#PBS -l nodes=1:ppn=8:debian8
#PBS -l mem=10gb
#PBS -l walltime=1d
#PBS -l scratch=5gb
#PBS -j oe
#PBS -m e
# chybovy vystup pripoji ke standarnimu vystupu a posle mail pri skonceni ulohy
# direktivy si upravte/umazte dle potreb sveho vypoctu
 
# nastaveni uklidu SCRATCHE pri chybe nebo ukonceni
# (pokud nerekneme jinak, uklidime po sobe)
trap 'clean_scratch' TERM EXIT

set -e

#nastavení pracovního aresáře pro vstupní/výstupní data 
DATADIR="$PBS_O_WORKDIR"

# Get input data
cd $DATADIR
file="script.py"
args="--threads 8"
wget "http://www.ms.mff.cuni.cz/~skopeko/files/dl/mnist_conv.py" -O "$file"
cp $DATADIR/"$file" $SCRATCHDIR

# přechod do pracovního adresáře a zahájení výpočtu 
cd $SCRATCHDIR

# nahrání požadovaného modulu
module add python-2.7.10-gcc
module add python27-modules-gcc

# Install tensorflow
# Installs a precompiled binary tensorflow wheel for the debian8 system at MetaCentrum.
tf_test="`echo "import tensorflow as tf; print tf.__version__" | python 2>&1`"
if `echo $tf_test | grep '^0[.]11[.].*' -q > /dev/null 2>&1`; then
    echo "Tensorflow $tf_test installed and ready."
else
    echo "Tensorflow not installed, trying to install prebuilt..."
    orig_dir="`pwd`"
    rm -rf tf_install 2>&1 >/dev/null
    mkdir tf_install
    cd tf_install
    wget "http://www.ms.mff.cuni.cz/~skopeko/files/tensorflow-0.11.0rc2-cp27-none-linux_x86_64.whl"
    pip install --upgrade --user tensorflow*.whl
    cd "$orig_dir"
fi

# Verify tensorflow
tf_test="`echo "import tensorflow as tf; print tf.__version__" | python 2>&1`"
if `echo $tf_test | grep '^0[.]11[.].*' -q > /dev/null 2>&1`; then
    echo "Tensorflow $tf_test installed and ready."
else
    echo "Tensorflow failed to install."
    echo "$tf_test"
    exit 2
fi

output="out.log"
logdir="logs"
echo "Running python script..."
eval python "$file" "$args" > "$output" 2>&1
echo "Python script finished."
cat "$output"

# vykopírování výsledků ze scratche (pokud selže, nechá data ve SCRATCHDIR a informuje uživatele)
cp $SCRATCHDIR/$output $DATADIR || export CLEAN_SCRATCH=false
cp -r $SCRATCHDIR/$logdir $DATADIR || export CLEAN_SCRATCH=false

