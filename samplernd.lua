function loadfbdseq(fname)
	local file=io.open(fname)
	local num=file:read("*n")
	local rs={}
	while num do
		local tmpt={}
		local ind=file:read("*n")
		for i=1,num-1 do
			local vi=file:read("*n")
			tmpt[vi]=true
		end
		rs[ind]=tmpt
		num=file:read("*n")
	end
	file:close()
	return rs
end

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

fbdset=loadfbdseq('datasrc/fbd.txt')
cnoun=nwvec:size(1)-1
