# ALFWorld
ALFWORLD_INIT_TEXT_OBS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}]. 
Your response should be a valid json file in the following format: \n{{\n\"thoughts\": \"first describe what do you see in the image using the text description, then carefully think about which action to complete the task.\", \n\"action\": \"an admissible action\"\n}}
"""

ALFWORLD_TEXT_OBS = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} actions you took and the corresponding environment feedback: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}]. 
Your response should be a valid json file in the following format: \n{{\n\"thoughts\": \"first describe what do you see in the image using the text description, then carefully think about which action to complete the task.\", \n\"action\": \"an admissible action\"\n}}
"""