require "cutorch"

local TestDataContainer = torch.class('TestDataContainer')

local function getLen(dset)
	local rs = 0
	for k, v in pairs(dset:all()) do
		rs = rs + 1
	end
	return rs
end

function TestDataContainer:__init(dset, ndata)
	self.dset = dset
	self.ndata = ndata or getLen(self.dset)
	if logger then
		logger:log("Tester: Iterator over testset")
	end
end

local function iter(din, curid)
	local dset, ndata = unpack(din)
	if curid <= ndata then
		local data = dset:read(tostring(curid)):all():cudaLong()
		curid = curid + 1
		return curid, {data:select(2, 1), data:select(2, 2)}
	else
		return
	end
end

function TestDataContainer:subiter()
	return iter, {self.dset, self.ndata}, 1
end

function TestDataContainer:close()
	self.dset:close()
end
