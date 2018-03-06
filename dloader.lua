require "hdf5"
require "dset"
require "utils.TrainDataContainer"
require "utils.ValidDataContainer"
require "utils.GTrainDataContainer"
require "utils.GValidDataContainer"
local traindata = hdf5.open("datasrc/train.h5", "r")
local validata = hdf5.open("datasrc/valid.h5", "r")

if logger then
	logger:log("verbs:"..verbs..", nouns:"..nouns)
	logger:log("batches:"..ntrain.."(train), "..nvalid.."(valid)")
end

if usegraph then
	return {GTrainDataContainer(traindata, ntrain, nouns, window, rate), GValidDataContainer(validata, nvalid)}
else
	return {TrainDataContainer(traindata, ntrain, nouns, window, rate), ValidDataContainer(validata, nvalid)}
end
