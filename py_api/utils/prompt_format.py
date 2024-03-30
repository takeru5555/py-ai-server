# this is a module for constructing the proper prompt format
#   for a specified llm model, from a list of so-called "prompt parts"
#   which are a shorthand format for constructing a prompt that
#   can be constructed into any format

# parts is a list of objects
#   each object is a prompt part
# each prompt part has at least a `val` propertyn and optionally:
#   - `use`: boolean indicating whether to include this part in the prompt
#   - `pre`: string to add before the part
#   - `suf`: string to add after the part

# further, different models support different elements of the prompt (e.g. some have a system/instruction section and some don't, in which case it's to be included in the user section)

# make Formatter class with methods like `flexible`, `Alpaca` and `ChatML`

import fnmatch, os
from py_api.models.llm.llm_api import PromptPart, PromptParts

class Formatter:
	# TODO use consistent casing of role names
	# (e.g. user vs USER)
	def flexible(
		self,
		user,
		system: str = '',
		prefix_response='',
		prior_msgs=[],
		user_role='user',
		assistant_role='assistant',
	):
		prompt = ''
		if system != '':
			prompt += system.strip() + '\n'
		if len(prior_msgs) > 0:
			for msg in prior_msgs:
				prompt += msg['role'].capitalize(
				) + ': ' + msg['content'] + '\n'
			prompt += user_role.capitalize() + ': '
		prompt += user.strip() + '\n'
		if prefix_response == '':
			if len(prior_msgs) > 0:
				prompt += assistant_role.capitalize() + ': '
			prompt += 'RESPONSE:\n'
		else:
			prompt += prefix_response
		return prompt

	def Alpaca(
		self,
		user: str,
		system: str = '',
		prefix_response='',
		prior_msgs=[],
		user_role='user',
		assistant_role='assistant'
	):
		prompt = ''
		if system != '':
			if len(prior_msgs) > 0:
				prompt += '### Instruction:\n'
			prompt += system.strip() + '\n\n'
		if len(prior_msgs) > 0:
			for msg in prior_msgs:
				prompt += msg['role'].capitalize(
				) + ': ' + msg['content'] + '\n\n'
			prompt += user_role.capitalize() + ': '
		else:
			prompt += '### Instruction:\n'
		prompt += user.strip() + '\n'
		r = '' if len(
			prior_msgs
		) == 0 else f' {assistant_role.capitalize()}'
		prompt += f'### Response:\n'
		if prefix_response != '':
			prompt += prefix_response
		return prompt

	def Alpaca_Input(
		self,
		user: str,
		system: str = '',
		prefix_response='',
		prior_msgs=[],
		user_role='user',
		assistant_role='assistant'
	):
		"""Alpaca variation with the system prompt as the instructions."""
		prompt = ''
		if system != '':
			prompt += '### Instruction:\n' + system.strip() + '\n\n'
		prompt += '### Input:\n'
		if len(prior_msgs) > 0:
			for msg in prior_msgs:
				prompt += msg['role'].capitalize(
				) + ': ' + msg['content'] + '\n\n'
			prompt += user_role.capitalize() + ': '
		prompt += user.strip() + '\n'
		r = '' if len(
			prior_msgs
		) == 0 else f' {assistant_role.capitalize()}'
		prompt += f'###{r} Response:\n'
		if prefix_response != '':
			prompt += prefix_response
		return prompt

	def ChatML(
		self,
		user: str,
		system: str = '',
		prefix_response='',
		prior_msgs=[],
		user_role='user',
		assistant_role='assistant'
	):
		prompt = ''
		u = user_role.lower()
		a = assistant_role.lower()
		if system != '':
			prompt += '<|im_start|>system\n' + system.strip(
			) + '<|im_end|>\n'
		if len(prior_msgs) > 0:
			for msg in prior_msgs:
				prompt += f'<|im_start|>{msg["role"]}\n' + msg[
					"content"] + '<|im_end|>\n'
		prompt += f'<|im_start|>{u}\n' + user.strip(
		) + '<|im_end|>\n'
		prompt += f'<|im_start|>{a}\n'
		if prefix_response != '':
			prompt += prefix_response
		return prompt

	def MistralInstruct(
		self,
		user: str,
		system: str = '',
		prefix_response='',
		prior_msgs=[],
		user_role='user',
		assistant_role='assistant'
	):
		# <s>[INST] {prompt} [/INST]
		prompt = '<s>'
		if system != '':
			prompt += '[INST] ' + system.strip()
		if len(prior_msgs) > 0:
			prompt += '\n[/INST]\n'
			for msg in prior_msgs:
				wrap = msg['role'] == user_role
				m = f'{msg["role"]}: {msg["content"]}'
				prompt += f'[INST] {m} [/INST]\n' if wrap else m + '\n'
			prompt += '[INST] '
		prompt += user.strip() + ' [/INST]'
		if prefix_response != '':
			prompt += '\n' + prefix_response
		return prompt

	def UserAssistant(
		self,
		user: str,
		system: str = '',
		prefix_response='',
		prior_msgs=[],
		user_role='user',
		assistant_role='assistant'
	):
		prompt = ''
		if system != '':
			prompt += system.strip() + '\n'
		if len(prior_msgs) > 0:
			# if roles are not standard, use them to map to standard User/Assistant roles
			is_standard = True
			if user_role != 'user' or assistant_role != 'assistant':
				is_standard = False
			for msg in prior_msgs:
				role = msg['role']
				if not is_standard:
					if role == user_role:
						role = 'USER'
					elif role == assistant_role:
						role = 'ASSISTANT'
				prompt += role + ':\n' + msg['content'].strip() + '\n'
		prompt += 'USER:\n' + user.strip() + '\n'
		prompt += 'ASSISTANT:\n'
		if prefix_response != '':
			prompt += prefix_response
		return prompt

	def UserAssistantNewlines(
		self,
		user: str,
		system: str = '',
		prefix_response='',
		prior_msgs=[],
		user_role='user',
		assistant_role='assistant'
	):
		# this one's not supposed to have system
		prompt = ''
		if system != '':
			prompt += system.strip() + '\n\n'
		u = user_role.capitalize()
		a = assistant_role.capitalize()
		if len(prior_msgs) > 0:
			for msg in prior_msgs:
				prompt += f'{msg["role"].capitalize()}:\n{msg["content"].strip()}\n\n'
		prompt += f'### {u}:\n' + user.strip() + '\n\n'
		prompt += f'### {a}:\n'
		if prefix_response != '':
			prompt += prefix_response
		return prompt

	# this one returns a list of messages instead
	# not sure what to do with this exactly
	def OpenAI(
		self,
		user: str,
		system: str = '',
		prefix_response='',
		prior_msgs=[],
		user_role='user',
		assistant_role='assistant'
	):
		# not sure how to implement prefix_response
		prompt = []
		if system != '':
			prompt.append({
				'role': 'system',
				'content': system.strip()
			})
		if len(prior_msgs) > 0:
			for msg in prior_msgs:
				prompt.append({
					'role': msg['role'],
					'content': msg['content']
				})
		prompt.append({'role': user_role, 'content': user.strip()})
		if prefix_response != '':
			prompt.append({
				'role': assistant_role,
				'content': prefix_response
			})
		return prompt

def parts_to_str(parts: list[PromptPart]):
	s = ''
	for part in parts:
		hasStr = part.val != None
		shouldUse = part.use != None and part.use
		hasPre = part.pre != None
		pre = part.pre if hasPre else ''
		hasSuf = part.suf != None
		suf = part.suf if hasSuf else ''
		if hasStr and shouldUse:
			partStr = f'{pre}{part.val}{suf}'
			s += partStr
	return s

model_formats = {
	'*dolphin-2*': 'ChatML',
	'*emerhyst-20b*': 'Alpaca',
	'*luna-ai-llama2*': 'UserAssistant',
	'*mistral-7b-instruct*': 'MistralInstruct',
	'*mythalion-13b*': 'Alpaca',
	'*mythomax-l2-13b*': 'Alpaca',
	'*openhermes-2.*-mistral-7b*': 'ChatML',
	'*platypus-30b*': 'Alpaca',
	'*solar-10.7b-instruct*': 'UserAssistantNewlines',
}

def get_model_format(model: str) -> str:
	fmt = None
	model = model.lower()
	for model_format in model_formats:
		if fnmatch.fnmatch(model, model_format):
			fmt = model_formats[model_format]
			break
	if fmt == None:
		raise Exception(f'Model {model} not supported.')
	return fmt

def parts_to_prompt(
	parts: PromptParts, model: str, prefix_response=''
) -> str:
	formatter = None
	# is model a path? get just the model name
	if '/' in model:
		model = os.path.basename(model)
	fmt = get_model_format(model)
	formatter = staticmethod(getattr(Formatter(), fmt))
	user = parts_to_str(parts.user)
	if hasattr(parts, 'system') and len(parts.system) > 0:
		system = parts_to_str(parts.system)
	else:
		system = ''
	prior_msgs = []
	if len(parts.prior_msgs
					) > 0 and not isinstance(parts.prior_msgs[0], dict):
		prior_msgs = [msg.model_dump() for msg in parts.prior_msgs]
	return formatter(
		user, system, prefix_response, prior_msgs=prior_msgs
	)

def parts_to_messages(
	parts: PromptParts,
	prefix_response=''
) -> list[dict[str, str]]:
	# only works for OpenAI format
	u = 'user'
	a = 'assistant'
	prior_msgs = parts.prior_msgs
	# convert prior_msgs to list of dicts if necessary
	if len(prior_msgs) > 0 and not isinstance(prior_msgs[0], dict):
		prior_msgs = [msg.model_dump() for msg in prior_msgs]
	assert isinstance(prior_msgs, list)
	msg = prior_msgs[-1] if len(prior_msgs) > 0 else None
	if msg != None:
		assert isinstance(msg, dict)
		u = msg['role']

	return Formatter().OpenAI(
		parts_to_str(parts.user), parts_to_str(parts.system),
		prefix_response, prior_msgs, u, a
	)
