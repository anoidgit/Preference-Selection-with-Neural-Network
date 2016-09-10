function getmaxout(inputs,outputs,nlinear)
	local ncyc=nlinear or 2
	return nn.Sequential():add(nn.Linear(inputs,outputs*ncyc)):add(nn.Reshape(ncyc,outputs,true)):add(nn.Max(2))
end

function getnn()
	local inputs = sizvec*2;
	local HUs = inputs;
	local outputs = 1;

	local id2vec=nn.ParallelTable();
	id2vec:add(nn.vecLookup(vwvec));
	id2vec:add(nn.vecLookup(nwvec));

	local inputmod=nn.Sequential();
	inputmod:add(id2vec);
	inputmod:add(nn.JoinTable(2));

	local nnmod_p1=nn.Sequential()
		:add(inputmod)
		:add(getmaxout(inputs, HUs))
		--:add(nn.Tanh())
		:add(getmaxout(HUs, outputs));

	local prl=nn.ParallelTable();
	prl:add(nnmod_p1);
	prl:add(nnmod_p1:clone('weight','bias'));

	local nnmod=nn.Sequential();
	nnmod:add(prl);

	return nnmod
end
