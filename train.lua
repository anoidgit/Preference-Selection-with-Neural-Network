print("set default tensor type to float")
torch.setdefaulttensortype('torch.FloatTensor')

function gradUpdate(mlpin, x, y, criterionin, learningRate)
	local pred=mlpin:forward(x)
	local err=criterionin:forward(pred, y)
	sumErr=sumErr+err
	local gradCriterion=criterionin:backward(pred, y)
	mlpin:zeroGradParameters()
	mlpin:backward(x, gradCriterion)
	mlpin:updateGradParameters(0.9)
	mlpin:updateParameters(learningRate)
	mlpin:maxParamNorm(-1)
end

function evaDev(mlpin, x)
	mlpin:evaluate()
	local ind=1
	local eind=x[1]:size(1)
	local comv
	local rs=nil
	while ind+batchsize<eind do
		comv=x[1]:narrow(1,ind,batchsize)
		local brs=mlpin:forward({{comv,x[2]:narrow(1,ind,batchsize)},{comv,x[3]:narrow(1,ind,batchsize)}})
		if rs then
			rs={rs[1]:cat(brs[1],1),rs[2]:cat(brs[2],1)}
		else
			rs={brs[1]:clone(),brs[2]:clone()}
		end
		ind=ind+batchsize
	end
	local exlen=eind-ind+1
	comv=x[1]:narrow(1,ind,exlen)
	local brs=mlpin:forward({{comv,x[2]:narrow(1,ind,exlen)},{comv,x[3]:narrow(1,ind,exlen)}})
	if rs then
		rs={rs[1]:cat(brs[1],1),rs[2]:cat(brs[2],1)}
	else
		rs={brs[1]:clone(),brs[2]:clone()}
	end
	local frs=torch.gt(rs[2]-rs[1],0)
	mlpin:training()
	return torch.sum(frs)/frs:size(1)
end

function inirand(cyc)
	cyc=cyc or 8
	for i=1,cyc do
		local sdata=math.random(nsam)
	end
end

function saveObject(fname,objWrt)
	local tmpod=nil
	if not torch.isTensor(objWrt) then
		tmpod=nn.Serial(objWrt)
		tmpod:lightSerial()
	else
		tmpod=objWrt
	end
	local file=torch.DiskFile(fname,'w')
	file:writeObject(tmpod)
	file:close()
end

print("load settings")
require"conf"

print("load data")
require "dloader"

target=torch.Tensor(batchsize):fill(1)
sumErr=0
crithis={}
cridev={}
critest={}
erate=0
storemini=1
storedevmini=1
storetestmini=1
minerrate=marguse
mindeverrate=minerrate
mintesterrate=minerrate

print("load packages")
require "nn"
require "dpnn"
require "vecLookup"
require "gnuplot"

if usernd then
	require "samplernd"
else
	require "sampler"
end

function train()

	print("design neural networks and criterion")
	require "designn"
	local nnmod=getnn()

	print(nnmod)
	nnmod:training()

	local critmod=getcrit()

	print("init train")
	local epochs=1
	local lr=modlr
	inirand()
	print("Init model Dev:"..evaDev(nnmod,devin))
	collectgarbage()

	print("start pre train")
	for tmpi=1,warmcycle do
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
		for innercycle=1,gtraincycle do
			for tmpi=1,ieps do
				local input=getsample(batchsize)
				gradUpdate(nnmod,input,target,critmod,lr)
			end
			local erate=sumErr/ieps
			table.insert(crithis,erate)
			local edevrate=evaDev(nnmod,devin)
			local etestrate=evaDev(nnmod,testin)
			table.insert(cridev,edevrate)
			table.insert(critest,etestrate)
			print("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate..",Dev:"..edevrate..",Test:"..etestrate)
			local modsavd=false
			if etestrate<mintesterrate then
				print("new minimal test error found,save model")
				mintesterrate=etestrate
				saveObject("modrs/testnnmod"..storetestmini..".asc",nnmod)
				storetestmini=storetestmini+1
				if storetestmini>csave then
					storetestmini=1
				end
				modsavd=true
			end
			if edevrate<mindeverrate then
				if not modsavd then
					print("new minimal dev error found,save model")
					mindeverrate=edevrate
					saveObject("modrs/devnnmod"..storedevmini..".asc",nnmod)
					storedevmini=storedevmini+1
					if storedevmini>csave then
						storedevmini=1
					end
					modsavd=true
				end
			end
			if erate<minerrate then
				minerrate=erate
				aminerr=0
				if not modsavd then
					print("new minimal error found,save model")
					saveObject("modrs/nnmod"..storemini..".asc",nnmod)
					storemini=storemini+1
					if storemini>csave then
						storemini=1
					end
				end
			else
				if aminerr>=expdecaycycle then
					aminerr=0
					if lrdecayepochs>lrdecaycycle then
						modlr=lr
						lrdecayepochs=1
					end
					lrdecayepochs=lrdecayepochs+1
					lr=modlr/(lrdecayepochs)
				end
				aminerr=aminerr+1
			end
			sumErr=0
			epochs=epochs+1
		end

		print("save neural network trained")
		saveObject("modrs/nnmod.asc",nnmod)

		print("save criterion history trained")
		local critensor=torch.Tensor(crithis)
		saveObject("modrs/crit.asc",critensor)
		local critdev=torch.Tensor(cridev)
		saveObject("modrs/critdev.asc",critdev)

		print("plot and save criterion")
		gnuplot.plot(critensor)
		gnuplot.figprint("modrs/crit.png")
		gnuplot.figprint("modrs/crit.eps")
		gnuplot.plotflush()
		gnuplot.plot(critdev)
		gnuplot.figprint("modrs/critdev.png")
		gnuplot.figprint("modrs/critdev.eps")
		gnuplot.plotflush()

		critensor=torch.Tensor()
		critdev=torch.Tensor()

		print("task finished!Minimal error rate:"..minerrate.."	"..mindeverrate.."	"..mintesterrate)

		print("wait for test, neural network saved at nnmod*.asc")

		icycle=icycle+1

		print("collect garbage")
		collectgarbage()

	end
end

train()
