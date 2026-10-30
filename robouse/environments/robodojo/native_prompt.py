import json


def initial_message(prompt, goal, observation, blocks):
    instruction = json.loads(goal.text)["instruction"]
    state = blocks[0]["text"]
    if not json.loads(state).get("ok"):
        raise ValueError("INITIAL_OBSERVATION_FAILED")
    images = [b for b in blocks if b["type"] == "image"]
    if len(images) != 3:
        raise ValueError("THREE_NATIVE_PRELOAD_IMAGES_REQUIRED")
    message = f"\n# This episode\n\nTask: {instruction}\n\nThe first observation is already taken, so do not call `goal` or `observe` to start. Its compact state:\n\n{state}\n\nThe three images attached to this message are this observation's top, left_wrist, right_wrist images, in that order.\n"
    return {
        "role": "user",
        "content": [{"type": "text", "text": prompt + message}, *images],
    }
