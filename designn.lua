require "deps.vecLookup"
require "deps.SequenceContainer"
require "deps.Applier"
require "rnn.Dropout"
require "gcrit"

function getnn()
	--return getonn()
	return getnnn()
end

function getonn()
	wvec = nil
	--local lmod = loadObject("modrs/nnmod.asc").module
	local lmod = torch.load("modrs/nnmod.asc").modules[1]:get(1).module
	return lmod
end

function getnnn()

	require "model.snn"
	require "model.gsnn"
	--require "model.mnn"
	--require "model.gmnn"
	if usegraph then
		return getusegnn(vsize, hsize, maxn)
	else
		return getusenn(vsize, hsize, maxn)
	end

end

function setupvec(modin, value)
	modin:apply(function(m)
		if torch.isTypeOf(m, 'nn.vecLookup') then
			m.updatevec = value
		end
	end)
end

function setnormvec(modin, value)
	modin:apply(function(m)
		if torch.isTypeOf(m, 'nn.vecLookup') then
			m.usenorm = value
		end
	end)
end

function dupvec(modin)
	setupvec(modin,false)
end

function upvec(modin)
	setupvec(modin,true)
end

function dnormvec(modin)
	setnormvec(modin,false)
end

function normvec(modin)
	setnormvec(modin,true)
end

function ldvvec()
	local embf = hdf5.open("datasrc/v.h5", "r")
	local rs = embf:read("emb"):all():float()
	embf:close()
	return rs
end

function ldnvec()
	local embf = hdf5.open("datasrc/n.h5", "r")
	local rs = embf:read("emb"):all():float()
	embf:close()
	return rs
end
