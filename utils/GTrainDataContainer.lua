require "cutorch"

local GTrainDataContainer = torch.class('GTrainDataContainer')

local function getLen(dset)
	local rs = 0
	for k, v in pairs(dset:all()) do
		rs = rs + 1
	end
	return rs
end

function GTrainDataContainer:__init(dset, ndata, nouns, window, rate)
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
		local data = dset:read(tostring(curid)):all()
		local bsize = data:size(1)
		ans:resize(bsize):fill(1)
		rnd:resize(bsize, 1)
		if window and ((rate==nil) or (rate and math.random()>rate)) then
			rnd:random(1, window)
			if math.random() > 0.5 then
				rnd:add(data:select(2, 2))
				rnd:maskedFill(rnd:gt(nouns), nouns)
			else
				rnd:csub(data:select(2, 2), rnd)
				rnd:maskedFill(rnd:lt(1), 1)
			end
		else
			rnd:random(1, nouns)
		end
		curid = curid + 1
		return curid, torch.cat(data, rnd, 2):cudaLong(), ans
	else
		return
	end
end

function GTrainDataContainer:subiter()
	return iter, {self.dset, self.ans, self.rnd, self.ndata, self.nouns, self.window, self.rate}, 1
end

function GTrainDataContainer:close()
	self.dset:close()
end
