from uuid import uuid4
import time

def text_completion(
	text_result: str,
	result={},
	model_name='',
	tokens=0,
	max_tokens=0,
	finish_reason='length'
):
	# if result is provided, it should be the same shape as the result from here
	id = result['id'] if 'id' in result else uuid4().hex
	created = result['created'] if 'created' in result else int(
		time.time()
	)
	model_name = result['model'
											] if 'model' in result else model_name
	if tokens > 0 and tokens < max_tokens:
		finish_reason = 'stop'

	has_usage = 'usage' in result
	prompt_tokens = result['usage'][
		'prompt_tokens'
	] if has_usage and 'prompt_tokens' in result['usage'] else 0
	completion_tokens = result['usage'][
		'completion_tokens'
	] if has_usage and 'completion_tokens' in result['usage'] else 0
	if tokens > 0 and completion_tokens == 0:
		completion_tokens = tokens
	total_tokens = result['usage'][
		'total_tokens'
	] if has_usage and 'total_tokens' in result['usage'] else 0

	return {
		'id':
		id,
		'object':
		'text_completion',
		'created':
		created,
		'model':
		model_name,
		'choices': [{
			'text': text_result,
			'index': 0,
			'finish_reason': finish_reason,
		}],
		'usage': {
			'prompt_tokens': prompt_tokens,
			'completion_tokens': completion_tokens,
			'total_tokens': total_tokens,
		}
	}
