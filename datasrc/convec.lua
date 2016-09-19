torch.setdefaulttensortype('torch.FloatTensor')

function convfile(fsrc,frs,lsize,useint)
	local file=io.open(fsrc)
	local num=file:read("*n")
	local rs={}
	while num do
		table.insert(rs,num)
		num=file:read("*n")
	end
	file:close()
	if useint then
		ts=torch.IntTensor(rs)
	else
		ts=torch.Tensor(rs)
	end
	ts:resize(#rs/lsize,lsize)
	file=torch.DiskFile(frs,'w')
	file:writeObject(ts)
	file:close()
end

function gvec(nvec,vecsize,frs)
	local file=torch.DiskFile(frs,"w")
	file:writeObject(torch.randn(nvec,vecsize))
	file:close()
end

function convweight(fsrc,frs)
	local file=io.open(fsrc)
	local num=file:read("*n")
	local rs={}
	while num do
		table.insert(rs,num)
		num=file:read("*n")
	end
	file:close()
	local ts=torch.IntTensor(rs)
	ts:resize(#rs,1)
	file=torch.DiskFile(frs,'w')
	file:writeObject(ts)
	file:close()
end

--convfile("trainer.txt","traineg.asc",3,true)
--convfile("tester.txt","testeg.asc",3,true)
--convfile("dever.txt","deveg.asc",3,true)

--do not forget unk
--gvec(4772,64,"vrvec.asc")
--gvec(44778,64,"nrvec.asc")

--convweight("deverweight.txt","dw.asc")
--convweight("testerweight.txt","tw.asc")

