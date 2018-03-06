require "hdf5"
require "tdset"
require "utils.TestDataContainer"
require "nn"
require "dpnn.Module"
require "dpnn.Serial"
require "deps.vecLookup"
require "deps.SequenceContainer"
require "deps.Applier"
require "rnn.Dropout"
require "nngraph"
require "cunn"

local tstdata = hdf5.open("datasrc/test.h5", "r")
local tdata = TestDataContainer(tstdata, ntest)
local tmod = torch.load(arg[1]).modules[1].modules[1]
local rsf = io.open(arg[2], "w")
for i, id in tdata:subiter() do
	local rs = tmod:forward(id)
	for _, s in ipairs(rs:reshape(rs:size(1)):totable()) do
		rsf:write(tostring(s).."\n")
	end
end
rsf:close()
tdata:close()
