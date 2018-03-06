local Applier, parent = torch.class('nn.Applier', 'nn.SequenceContainer')

function Applier:updateOutput(input)
	self.output = {}
	if self.train then
		for _, v in ipairs(input) do
			table.insert(self.output, self:net(_):updateOutput(v))
		end
	else
		for _, v in ipairs(input) do
			table.insert(self.output, self:net(_):updateOutput(v):clone())
		end
	end
	return self.output
end

function Applier:updateGradInput(input, gradOutput)
	self.gradInput = {}
	for _, v in ipairs(input) do
		table.insert(self.gradInput, self:net(_):updateGradInput(v, gradOutput[_]))
	end
	return self.gradInput
end

function Applier:accGradParameters(input, gradOutput, scale)
	for _, v in ipairs(input) do
		self:net(_):accGradParameters(v, gradOutput[_], scale)
	end
end
