function loadObject(fname)
	local file=torch.DiskFile(fname)
	local objRd=file:readObject()
	file:close()
	return objRd
end

function loadDev(fposi)
	local pm=loadObject(fposi)
	return {pm:select(2,1),pm:select(2,2),pm:select(2,3)}
end

function loadTrain(ftrain)
	return loadObject(ftrain)
end

vwvec=loadObject('datasrc/vrvec.asc')
nwvec=loadObject('datasrc/nrvec.asc')
sizvec=nwvec:size(2)

mword=loadTrain('datasrc/traineg.asc')

devin=loadDev('datasrc/deveg.asc')

nsam=mword:size(1)

