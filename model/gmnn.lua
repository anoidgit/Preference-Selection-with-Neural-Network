require "nngraph"

local function initEmb(nc, vsize)
	if logger then
		logger:log("Uniform Embedding Init:"..nc..", "..vsize)
	end
	local rs = torch.FloatTensor(nc, vsize)
	rs:uniform():csub(0.5):div(vsize)
	rs[1]:zero()
	return rs
end

function getusegnn(vsize, hsize, maxn)
	local nmax = maxn or 2
	local vvec = nil
	local nvec = nil
	if pre_emb then
		vvec = ldvvec()
		nvec = ldnvec()
		if logger then
			logger:log("Pretrained Embedding Init:"..vvec:size(1)..", "..vvec:size(2))
			logger:log("Pretrained Embedding Init:"..nvec:size(1)..", "..nvec:size(2))
		end
	else
		vvec = initEmb(verbs, vsize)
		nvec = initEmb(nouns, vsize)
	end
	if logger then
		logger:log("Arch:Maxout, Layers:2")
		logger:log("Build with nngraph")
		logger:log("Vector size:"..vsize)
		logger:log("Hidden units:"..hsize)
		logger:log("Maxouts:"..nmax)
	end
	local vlook = nn.vecLookup(vvec)
	local nlook = nn.vecLookup(nvec)
	local input = nn.Identity()()
	local ve = nn.Select(2, 1)(vlook(nn.Narrow(2, 1, 1)(input)))
	local ne = nlook(nn.Narrow(2, 2, 2)(input))
	local npe = nn.Select(2, 1)(ne)
	local nne = nn.Select(2, 2)(ne)
	local vnp = nn.JoinTable(2, 2)({ve, npe})
	local vnn = nn.JoinTable(2, 2)({ve, nne})
	if pdrop then
		local dM = nn.Dropout(pdrop, nil, nil, true)
		vnp = dM(vnp)
		vnn = dM(vnn)
	end
	local nnmod_core=nn.Sequential()
		:add(nn.Maxout(vsize*2, hsize, nmax))
		:add(nn.Maxout(hsize, 1, nmax))
	local outputs = {nnmod_core(vnp), nnmod_core:clone("weight", "bias")(vnn)}
	return nn.gModule({input}, outputs)
end
