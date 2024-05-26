import time

PRE_PROMPT = "You are a helpful personal assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as a Personal Assistant."

def debounce(wait):
    def decorator(fn):
        last_call = [0]

        def debounced(*args, **kwargs):
            current_time = time.time()
            if current_time - last_call[0] >= wait:
                last_call[0] = current_time
                return fn(*args, **kwargs)
        
        return debounced
    
    return decorator
