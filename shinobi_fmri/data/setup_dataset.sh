cd ./data/shinobi_behav
datalad get ./
cd ./data/shinobi
git checkout event_files
datalad get ./
cd ..
scp -r yharel@elm.criugm.qc.ca:/data/neuromod/DATA/games/shinobi_beh ./

# Additional lines to setup gym environment
cd shinobi/stimuli
# export aws etc...
datalad get ./
conda activate shinobi_env
python3 -m retro.import ./ShinobiIIIReturnOfTheNinjaMaster-Genesis
cd ../../..
pip install -r requirements.txt
python setup.py install
cd ../..
pip install -r requirements.txt
python setup.py install
