require "cutorch"
require "math"

local TrainDataContainer = torch.class('TrainDataContainer')

local function getLen(dset)
	local rs = 0
	for k, v in pairs(dset:all()) do
		rs = rs + 1
	end
	return rs
end

function TrainDataContainer:__init(dset, ndata, nouns, window, rate)
	self.dset = dset
	self.ndata = ndata or getLen(self.dset)
	self.nouns = nouns
	self.window = window
	self.rate = rate
	self.rnd = torch.IntTensor()
	self.ans = torch.CudaTensor()
	if logger then
		if self.window then
			if self.rate then
				logger:log("Trainer: Combined negative sampler, random rate:"..self.rate..", window size:"..self.window)
			else
				logger:log("Trainer: Near negative sampler, window size:"..self.window)
			end
		else
			logger:log("Trainer: Random negative sample equally with noun vocab")
		end
	end
end

local function iter(din, curid)
	local dset, ans, rnd, ndata, nouns, window, rate = unpack(din)
	if curid <= ndata then
		local d_real = dset:read(tostring(curid)):all()
		local data = d_real:cudaLong()
		local bsize = data:size(1)
		ans:resize(bsize):fill(1)
		rnd:resize(bsize)
		if window and ((rate==nil) or (rate and math.random()>rate)) then
			rnd:random(1, window)
			if math.random() > 0.5 then
				rnd:add(d_real:select(2, 2))
				rnd:maskedFill(rnd:gt(nouns), nouns)
			else
				rnd:csub(d_real:select(2, 2), rnd)
				rnd:maskedFill(rnd:lt(1), 1)
			end
		else
			rnd:random(1, nouns)
		end
		curid = curid + 1
		local v = data:select(2, 1)
		return curid, {{v, data:select(2, 2)}, {v, rnd:cudaLong()}}, ans
	else
		return
	end
end

function TrainDataContainer:subiter()
	return iter, {self.dset, self.ans, self.rnd, self.ndata, self.nouns, self.window, self.rate}, 1
end

function TrainDataContainer:close()
	self.dset:close()
end
