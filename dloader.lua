vwvec=loadObject('datasrc/vrvec.asc')
nwvec=loadObject('datasrc/nrvec.asc')
sizvec=nwvec:size(2)

mword=loadTrain('datasrc/traineg.asc')

devin=loadDev('datasrc/deveg.asc')

nsam=mword:size(1)
