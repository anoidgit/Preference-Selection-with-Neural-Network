print("set default tensor type to float")
torch.setdefaulttensortype('torch.FloatTensor')

function gradUpdate(mlpin, x, y, criterionin, learningRate)
	local pred=mlpin:forward(x)
	local err=criterionin:forward(pred, y)
	sumErr=sumErr+err
	local gradCriterion=criterionin:backward(pred, y)
	mlpin:zeroGradParameters()
	mlpin:backward(x, gradCriterion)
	mlpin:updateGradParameters(0.875)
	mlpin:updateParameters(learningRate)
	mlpin:maxParamNorm(-1)
end

function evaDev(mlpin, x, criterionin)
	local tmod=mlpin:clone()
	tmod:evaluate()
	local bsiz=512
	local tart=torch.Tensor(bsiz):fill(1)
	local ind=1
	local serr=0
	local eind=#x
	local cfwd=1
	local comv
	while ind+bsiz<eind do
		comv=x[1]:narrow(1,ind,bsiz)
		serr=serr+criterionin:forward(tmod:forward({{comv,x[2]:narrow(1,ind,bsiz)},{comv,x[3]:narrow(1,ind,bsiz)}}), tart)
		ind=ind+bsiz
		cfwd=cfwd+1
	end
	local exlen=eind-ind+1
	comv=x[1]:narrow(1,ind,exlen)
	serr=serr+criterionin:forward(tmod:forward({{comv,x[2]:narrow(1,ind,exlen)},{comv,x[3]:narrow(1,ind,exlen)}}), torch.Tensor(exlen):fill(1))
	return serr/cfwd
end

function getresmodel(modelcap,scale,usegraph)
	local rtm=nn.ConcatTable()
	rtm:add(modelcap)
	if not scale or scale==1 then
		rtm:add(nn.Identity())
	elseif type(scale)=='number' then
		rtm:add(nn.Sequential():add(nn.Identity()):add(nn.MulConstant(scale,true)))
	else
		rtm:add(nn.Sequential():add(nn.Identity()):add(scale))
	end
	local rsmod=nn.Sequential():add(rtm):add(nn.CAddTable())
	if usegraph then
		local input=nn.Identity()()
		local output=rsmod(input)
		return nn.gModule({input},{output})
	else
		return rsmod
	end
end

function getmaxout(inputs,outputs,nlinear)
	local ncyc=nlinear or 2
	return nn.Sequential():add(nn.Linear(inputs,outputs*ncyc)):add(nn.Reshape(ncyc,outputs,true)):add(nn.Max(2))
end

function graphmodule(module_graph)
	local input=nn.Identity()()
	local output=module_graph(input)
	return nn.gModule({input},{output})
end

function loadfbdseq(fname)
	local file=io.open(fname)
	local num=file:read("*n")
	local rs={}
	while num do
		local tmpt={}
		local ind=file:read("*n")
		for i=1,num-1 do
			local vi=file:read("*n")
			tmpt[vi]=1
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
		table.insert(cotf,sdata[1])
		table.insert(cotl,sdata[2])
		table.insert(wrtf,sdata[3])
	end
	local comt=torch.Tensor(cotf)
	return {{comt,torch.Tensor(cotl)},{comt,torch.Tensor(wrtf)}}
end

function inirand(cyc)
	cyc=cyc or 8
	for i=1,cyc do
		local sdata=math.random(nsam)
	end
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

function loadDev(fposi,fneg)
	local pm=loadObject(fposi)
	local nm=loadObject(fneg)
	return {pm:select(2,1),pm:select(2,2),nm:select(2,2)}
end

function loadTrain(fposi,fneg)
	local pm=loadObject(fposi)
	local nm=loadObject(fneg)
	return torch.cat(pm,nm:select(2,2))
end

print("load settings")
batchsize=1024
marguse=1
ieps=256
modlr=0.5
sizvec=50

vwvec=loadObject('datasrc/vrvec.asc')
nwvec=loadObject('datasrc/nrvec.asc')

print("load training data")
mword=loadTrain('datasrc/train.asc','datasrc/traineg.asc')

devin=loadDev('datasrc/dev.asc','datasrc/devneg.asc')

nsam=mword:size(1)

target=torch.Tensor(batchsize):fill(1)
sumErr=0
crithis={}
cridev={}
erate=0
edevrate=0
storemini=1
storedevmini=1
minerrate=marguse
mindeverrate=minerrate

print("load packages")
require "nn"
require "nngraph"
require "dpnn"
require "vecLookup"
require "gnuplot"

function getnn()
	local inputs = 100;
	local HUs = 100;
	local outputs = 1;

	local id2vec=nn.ParallelTable();
	id2vec:add(nn.vecLookup(vwvec));
	id2vec:add(nn.vecLookup(nwvec));

	local inputmod=nn.Sequential();
	inputmod:add(id2vec);
	inputmod:add(nn.JoinTable(2));

	local nnmod_p1=nn.Sequential()
		:add(inputmod)
		:add(nn.Linear(inputs, HUs))
		:add(nn.Tanh())
		:add(nn.Linear(HUs, outputs));

	local prl=nn.ParallelTable();
	prl:add(nnmod_p1);
	prl:add(nnmod_p1:clone('weight','bias'));

	local nnmod=nn.Sequential();
	nnmod:add(prl);

	return nnmod
end

function train()

	print("design neural networks")
	local nnmod=getnn()

	print(nnmod)
	nnmod:training()

	print("design criterion")
	local critmod = nn.MarginRankingCriterion(marguse);

	print("init train")
	local epochs=1
	local lr=modlr
	inirand()
	print("Init model Dev:"..evaDev(nnmod,devin,critmod))
	collectgarbage()

	print("start pre train")
	for tmpi=1,32 do
		for tmpi=1,ieps do
			input=getsample(batchsize)
			gradUpdate(nnmod,input,target,critmod,lr)
		end
		local erate=sumErr/ieps
		table.insert(crithis,erate)
		print("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate)
		sumErr=0
		epochs=epochs+1
	end

	epochs=1
	icycle=1

	aminerr=0
	lrdecayepochs=1

	while true do
		print("start innercycle:"..icycle)
		for innercycle=1,256 do
			for tmpi=1,ieps do
				local input=getsample(batchsize)
				gradUpdate(nnmod,input,target,critmod,lr)
			end
			local erate=sumErr/ieps
			table.insert(crithis,erate)
			local edevrate=evaDev(nnmod,devin,critmod)
			table.insert(cridev,edevrate)
			print("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate..",Dev:"..edevrate)
			local modsavd=false
			if edevrate<mindeverrate then
				print("new minimal dev error found,save model")
				mindeverrate=edevrate
				saveObject("srs/devnnmod"..storedevmini..".asc",nnmod)
				storedevmini=storedevmini+1
				modsavd=true
			end
			if erate<minerrate then
				minerrate=erate
				aminerr=0
				if not modsavd then
					print("new minimal error found,save model")
					saveObject("srs/nnmod"..storemini..".asc",nnmod)
					storemini=storemini+1
				end
			else
				if aminerr>=4 then
					aminerr=0
					lrdecayepochs=lrdecayepochs+1
					lr=modlr/(lrdecayepochs)
				end
				aminerr=aminerr+1
			end
			sumErr=0
			epochs=epochs+1
		end

		print("save neural network trained")
		saveObject("srs/nnmod.asc",nnmod)

		print("save criterion history trained")
		local critensor=torch.Tensor(crithis)
		saveObject("srs/crit.asc",critensor)
		local critdev=torch.Tensor(cridev)
		saveObject("srs/critdev.asc",critdev)

		print("plot and save criterion")
		gnuplot.plot(critensor)
		gnuplot.figprint("srs/crit.png")
		gnuplot.figprint("srs/crit.eps")
		gnuplot.plotflush()
		gnuplot.plot(critdev)
		gnuplot.figprint("srs/critdev.png")
		gnuplot.figprint("srs/critdev.eps")
		gnuplot.plotflush()

		critensor=torch.Tensor()
		critdev=torch.Tensor()

		print("task finished!Minimal error rate:"..minerrate)

		print("wait for test, neural network saved at nnmod*.asc")

		icycle=icycle+1

		print("collect garbage")
		collectgarbage()

	end
end

train()
