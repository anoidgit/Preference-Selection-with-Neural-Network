starterate=math.huge--warning:only used as init erate, not asigned to criterion

logd="logs"

runid="180412_nyt_snn"

extra_header="batchsize:1024_unk:1_minwordfreq:1_mincocfreq:1_cooccur.weight:1_shfc:64_gpu:GTX1080.on"

pre_emb=true--use pretrained embedding

vsize=200
hsize=200
maxn=2
pdrop=nil

window=50
rate=0.1

ieps=1
warmcycle=0
expdecaycycle=4
gtraincycle=64

modlr=1/1024

earlystop=32

csave=3

lrdecaycycle=4

resetOptim=true--reset Optimizer after learning rate decay
usegraph=false

cycs=false--warning:this option need a lot of memory
savecycle=32
