require "cutorch"

local GValidDataContainer = torch.class('GValidDataContainer')

local function getLen(dset)
	local rs = 0
	for k, v in pairs(dset:all()) do
		rs = rs + 1
	end
	return rs
end

function GValidDataContainer:__init(dset, ndata)
	self.dset = dset
	self.ndata = ndata or getLen(self.dset)
	self.ans = torch.CudaTensor()
	if logger then
		logger:log("Valider: Use pre-defined samples")
	end
end

local function iter(din, curid)
	local dset, ans, ndata = unpack(din)
	if curid <= ndata then
		local data = dset:read(tostring(curid)):all():cudaLong()
		ans:resize(data:size(1)):fill(1)
		curid = curid + 1
		return curid, data, ans
	else
		return
	end
end

function GValidDataContainer:subiter()
	return iter, {self.dset, self.ans, self.ndata}, 1
end

function GValidDataContainer:close()
	self.dset:close()
end
