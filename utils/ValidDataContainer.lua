require "cutorch"

local ValidDataContainer = torch.class('ValidDataContainer')

local function getLen(dset)
	local rs = 0
	for k, v in pairs(dset:all()) do
		rs = rs + 1
	end
	return rs
end

function ValidDataContainer:__init(dset, ndata)
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
		local v = data:select(2, 1)
		return curid, {{v, data:select(2, 2)}, {v, data:select(2, 3)}}, ans
	else
		return
	end
end

function ValidDataContainer:subiter()
	return iter, {self.dset, self.ans, self.ndata}, 1
end

function ValidDataContainer:close()
	self.dset:close()
end
