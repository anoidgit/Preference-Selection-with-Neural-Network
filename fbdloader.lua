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