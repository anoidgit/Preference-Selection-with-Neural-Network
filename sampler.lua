function getsample(batch)
	local cotf={}
	local cotl={}
	local wrtf={}
	for i=1,batch do
		local sdata=mword[math.random(nsam)]
		table.insert(cotf,sdata[1])
		table.insert(cotl,sdata[2])
		table.insert(wrtf,sdata[3])
	end
	local comt=torch.Tensor(cotf)
	return {{comt,torch.Tensor(cotl)},{comt,torch.Tensor(wrtf)}}
end
