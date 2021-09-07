cd ./data/shinobi_behav
datalad get ./
cd ./data/shinobi
git checkout event_files
datalad get ./
cd ..
scp -r yharel@elm.criugm.qc.ca:/data/neuromod/DATA/games/shinobi_beh ./

# Additional lines to setup gym environment
cd shinobi/stimuli
datalad get ./
conda activate shinobi_env
python3 -m retro.import ./data/shinobi/stimuli/ShinobiIIIReturnOfTheNinjaMaster-Genesis
