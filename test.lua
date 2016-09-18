torch.setdefaulttensortype('torch.FloatTensor')

function eva(mlpin, x)
	mlpin:evaluate()
	local ind=1
	local eind=x[1]:size(1)
	local comv
	local rs=nil
	while ind+batchsize<eind do
		comv=x[1]:narrow(1,ind,batchsize)
		local brs=mlpin:forward({{comv,x[2]:narrow(1,ind,batchsize)},{comv,x[3]:narrow(1,ind,batchsize)}})
		if rs then
			rs[1]:cat(brs[1],1)
			rs[2]:cat(brs[2],1)
		else
			rs=brs
		end
		ind=ind+batchsize
	end
	local exlen=eind-ind+1
	comv=x[1]:narrow(1,ind,exlen)
	local brs=mlpin:forward({{comv,x[2]:narrow(1,ind,exlen)},{comv,x[3]:narrow(1,ind,exlen)}})
	if rs then
		rs[1]:cat(brs[1],1)
		rs[2]:cat(brs[2],1)
	else
		rs=brs
	end
	return rs
end

function loadObject(fname)
	local file=torch.DiskFile(fname)
	local objRd=file:readObject()
	file:close()
	return objRd
end

function saveObject(fname,objWrt)
	if not torch.isTensor(objWrt) then
		objWrt:lightSerial()
	end
	local file=torch.DiskFile(fname,'w')
	file:writeObject(objWrt)
	file:close()
end

function loadDev(fposi)
	local pm=loadObject(fposi)
	return {pm:select(2,1),pm:select(2,2),pm:select(2,3)}
end

require "nn"
require "vecLookup"
require "PartialNN"

batchsize=8192

function grs(nnmod,ftest)
	local devin=loadDev(ftest)
	local rst=eva(nnmod,devin)
	local frs=torch.gt(rst[1]-rst[2],0)
	return torch.sum(frs)/frs:size(1)
end

nnmod=loadObject('devnnmod.asc')
print(grs(nnmod,'../datasrc/testeg.asc'))
print(grs(nnmod,'../datasrc/deveg.asc'))

