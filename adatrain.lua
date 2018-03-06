print("load settings")
require"aconf"

runid = runid.."_Adam_nngraph:"..tostring(usegraph).."_resetOptim:"..tostring(resetOptim).."_window:"..tostring(window).."_rate:"..tostring(rate).."_"..vsize.."_"..hsize.."_"..tostring(pdrop).."_"..modlr.."_"..earlystop.."_"..extra_header

require "utils.Logger"
require "paths"
paths.mkdir(logd)
if cntrain then
	logmod = "a"
else
	logmod = "w"
end
logger = Logger(logd.."/"..runid..".log", nil, nil, logmod)

if extra_header then
	logger:log(extra_header)
end

logger:log("set default tensor type to float")
torch.setdefaulttensortype('torch.FloatTensor')

logger:log("Use Adam")

function feval()
	return _inner_err, _inner_gradParams
end

function gradUpdate(mlpin, x, y, criterionin, lr, optm)

	_inner_gradParams:zero()

	local pred=mlpin:forward(x)
	_inner_err=criterionin:forward(pred, y)
	sumErr=sumErr+_inner_err
	local gradCriterion=criterionin:backward(pred, y)
	mlpin:backward(x, gradCriterion)

	optm(feval, _inner_params, {learningRate = lr})

	--mlpin:maxParamNorm(2)

end

function evaDev(mlpin, criterionin, devdata)
	mlpin:evaluate()
	local serr=0
	local nd=0
	for i, id, td in devdata:subiter() do
		xlua.progress(i, nvalid)
		local ps, ns = unpack(mlpin:forward(id))
		nd=nd+ps:size(1)
		serr=serr+ps:le(ns):sum()
	end
	mlpin:training()
	return serr/nd
end

--[[function evaDev(mlpin, criterionin, devdata)
	mlpin:evaluate()
	local serr=0
	for i, id, td in devdata:subiter() do
		xlua.progress(i, nvalid)
		serr=serr+criterionin:forward(mlpin:forward(id), td)
	end
	mlpin:training()
	return serr/nvalid
end]]

function inirand(cyc)
	local rs
	for i=1, (cyc or 8) do
		rs = math.random()
	end
	return rs
end

function saveObject(fname,objWrt)
	if torch.isTensor(objWrt) then
		torch.save(fname,objWrt)
	else
		objWrt:clearState()
		torch.save(fname,nn.Serial(objWrt):mediumSerial())
	end
	--[[local file=torch.DiskFile(fname,'w')
	file:writeObject(tmpod)
	file:close()]]
end

logger:log("pre load package")
require "nn"
require "dpnn.Module"
require "dpnn.Serial"

sumErr=0
crithis={}
cridev={}

if cycs then
	bnnmod=nil
	bdevnnmod=nil
	bupdnm=false
	bupdevnm=false
end

function train()

	local erate=0
	local edevrate=0
	local storemini=1
	local storedevmini=1
	local minerrate=starterate
	local minvaliderrate=minerrate

	logger:log("prepare environment")
	local savedir="modrs/"..runid.."/"
	paths.mkdir(savedir)

	logger:log("load optim")
	require "getoptim"
	local optmethod=getoptim()

	logger:log("load data")
	local traind, devd = unpack(require "dloader")

	logger:log("design neural networks and criterion")
	require "cunn"

	require "designn"
	local nnmod=getnn()

	logger:log(nnmod)
	nnmod:training()

	local critmod=getcrit()

	nnmod:cuda()
	critmod:cuda()

	logger:log("init train")
	local epochs=1
	local lr=modlr
	inirand()

	logger:log("turn off embeddings update")
	dupvec(nnmod)
	logger:log("turn off embeddings norm")
	dnormvec(nnmod)
	
	minvaliderrate=evaDev(nnmod,critmod,devd)
	logger:log("Init model Dev:"..minvaliderrate)

	_inner_params, _inner_gradParams=nnmod:getParameters()

	collectgarbage()

	logger:log("start pre train")
	for tmpi=1,warmcycle do
		for tmpj=1,ieps do
			for i, id, td in traind:subiter() do
				gradUpdate(nnmod, id, td, critmod, lr, optmethod)
				xlua.progress(i, ntrain)
			end
		end
		local erate=sumErr/ntrain
		if erate<=minerrate then
			minerrate=erate
		end
		table.insert(crithis,erate)
		logger:log("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate)
		sumErr=0
		epochs=epochs+1
	end

	if warmcycle>0 then
		logger:log("save neural network trained")
		saveObject(savedir.."nnmod.asc",nnmod)
	end

	logger:log("turn on embeddings update")
	upvec(nnmod)

	epochs=1
	local icycle=1

	local aminerr=1
	local aminvaliderr=1
	local lrdecayepochs=1

	local cntrun=true

	collectgarbage()

	while cntrun do
		logger:log("start innercycle:"..icycle)
		for innercycle=1,gtraincycle do
			for tmpi=1,ieps do
				for i, id, td in traind:subiter() do
					gradUpdate(nnmod, id, td, critmod, lr, optmethod)
					xlua.progress(i, ntrain)
				end
			end
			local erate=sumErr/ntrain
			table.insert(crithis,erate)
			local edevrate=evaDev(nnmod,critmod,devd)
			table.insert(cridev,edevrate)
			logger:log("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate..",Dev:"..edevrate)
			--logger:log("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate)
			local modsavd=false
			if edevrate<=minvaliderrate then
				minvaliderrate=edevrate
				aminvaliderr=1
				if cycs then
					nnmod:float()
					bdevnnmod=nnmod:clone()
					nnmod:cuda()
					bupdevnm=true
				else
					saveObject(savedir.."devnnmod"..storedevmini..".asc",nnmod)
					storedevmini=storedevmini+1
					if storedevmini>csave then
						storedevmini=1
					end
				end
				modsavd=true
				logger:log("new minimal dev error found, model saved")
			else
				if earlystop and aminvaliderr>earlystop then
					logger:log("early stop")
					cntrun=false
					break
				end
				aminvaliderr=aminvaliderr+1
			end
			if erate<=minerrate then
				minerrate=erate
				aminerr=1
				if not modsavd then
					if cycs then
						nnmod:float()
						bnnmod=nnmod:clone()
						nnmod:cuda()
						bupdnm=true
					else
						saveObject(savedir.."nnmod"..storemini..".asc",nnmod)
						storemini=storemini+1
						if storemini>csave then
							storemini=1
						end
					end
					logger:log("new minimal error found, model saved")
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
					if resetOptim then
						logger:log("Reset Optimizer")
						optmethod=getoptim()
						collectgarbage()
					end
				end
				aminerr=aminerr+1
			end
			sumErr=0
			if cycs and epochs%savecycle==0 then
				if bupdevnm then
					logger:log("flush dev mod")
					torch.save(savedir.."devnnmod"..storedevmini..".asc",bdevnnmod)
					bdevnnmod=nil
					storedevmini=storedevmini+1
					if storedevmini>csave then
						storedevmini=1
					end
					bupdevnm=false
				end
				if bupdnm then
					logger:log("flush mod")
					torch.save(savedir.."nnmod"..storemini..".asc",bnnmod)
					bnnmod=nil
					storemini=storemini+1
					if storemini>csave then
						storemini=1
					end
					bupdnm=false
				end
			end
			epochs=epochs+1
		end

		logger:log("save neural network trained")
		saveObject(savedir.."nnmod.asc",nnmod)

		logger:log("save criterion history trained")
		local critensor=torch.Tensor(crithis)
		saveObject(savedir.."crit.asc",critensor)
		local critdev=torch.Tensor(cridev)
		saveObject(savedir.."critdev.asc",critdev)

		--[[logger:log("plot and save criterion")
		gnuplot.plot(critensor)
		gnuplot.figlogger:log(savedir.."crit.png")
		gnuplot.figlogger:log(savedir.."crit.eps")
		gnuplot.plotflush()
		gnuplot.plot(critdev)
		gnuplot.figlogger:log(savedir.."critdev.png")
		gnuplot.figlogger:log(savedir.."critdev.eps")
		gnuplot.plotflush()]]

		critensor=nil
		critdev=nil

		logger:log("task finished!Minimal error rate:"..minerrate.."	"..minvaliderrate)
		--logger:log("task finished!Minimal error rate:"..minerrate)

		logger:log("wait for test, neural network saved at nnmod*.asc")

		icycle=icycle+1

		logger:log("collect garbage")
		collectgarbage()

	end
	traind:close()
	devd:close()
end

train()
