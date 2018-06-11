local function initEmb(nc, vsize)
	if logger then
		logger:log("Uniform Embedding Init:"..nc..", "..vsize)
	end
	local rs = torch.FloatTensor(nc, vsize)
	rs:uniform():csub(0.5):div(vsize)
	rs[1]:zero()
	return rs
end

function getusenn(vsize, hsize)
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
		logger:log("Arch:feed-forward, Layers:2")
		logger:log("Vector size:"..vsize)
		logger:log("Hidden units:"..hsize)
	end
	local vlook = nn.vecLookup(vvec)
	local nlook = nn.vecLookup(nvec)
	local look = nn.ParallelTable()
		:add(vlook)
		:add(nlook)
	local nnmod_core=nn.Sequential()
		:add(look)
		:add(nn.JoinTable(2, 2))
	if pdrop then
		nnmod_core:add(nn.Dropout(pdrop, nil, nil, true))
	end
	nnmod_core:add(nn.Linear(vsize*2, hsize))
		:add(nn.Tanh())
		:add(nn.Linear(hsize, 1))
	local nnmod=nn.Applier(nnmod_core)
	return nnmod
end
