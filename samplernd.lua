function getsample(batch)
	local cotf={}
	local cotl={}
	local wrtf={}
	for i=1,batch do
		local sdata=mword[math.random(nsam)]
		local sverb=sdata[1]
		table.insert(cotf,sverb)
		table.insert(cotl,sdata[2])
		local snn=math.random(cnoun)
		while fbdset[sverb][snn] do
			snn=math.random(cnoun)
		end
		table.insert(wrtf,snn)
	end
	local comt=torch.Tensor(cotf)
	return {{comt,torch.Tensor(cotl)},{comt,torch.Tensor(wrtf)}}
end
