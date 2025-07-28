import re
import time
from config import logger

class ParserState:
    __slots__ = ['answer', 'thought', 'in_think', 'in_answer', 'start_time', 'last_pos', 'total_think_time']
    def __init__(self):
        self.answer = ""
        self.thought = ""
        self.in_think = False
        self.in_answer = False
        self.start_time = 0
        self.last_pos = 0
        self.total_think_time = 0.0

def format_time(seconds_float):
    total_seconds = int(round(seconds_float))
    hours = total_seconds // 3600
    remaining_seconds = total_seconds % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def parse_response(text, state):
    buffer = text[state.last_pos:]
    state.last_pos = len(text)
    
    while buffer:
        if not state.in_think and not state.in_answer:
            think_start = buffer.find('<think>')
            reasoning_start = buffer.find('<reasoning>')
            answer_start = buffer.find('<answer>')
            
            starts = []
            if think_start != -1:
                starts.append((think_start, '<think>', 7, 'think'))
            if reasoning_start != -1:
                starts.append((reasoning_start, '<reasoning>', 11, 'think'))
            if answer_start != -1:
                starts.append((answer_start, '<answer>', 8, 'answer'))
            
            if not starts:
                state.answer += buffer
                break
            
            start_pos, start_tag, tag_length, mode = min(starts, key=lambda x: x[0])
            
            state.answer += buffer[:start_pos]
            if mode == 'think':
                state.in_think = True
                state.start_time = time.perf_counter()
            else:
                state.in_answer = True
            buffer = buffer[start_pos + tag_length:]
            
        elif state.in_think:
            think_end = buffer.find('</think>')
            reasoning_end = buffer.find('</reasoning>')
            
            ends = []
            if think_end != -1:
                ends.append((think_end, '</think>', 8))
            if reasoning_end != -1:
                ends.append((reasoning_end, '</reasoning>', 12))
            
            if ends:
                end_pos, end_tag, tag_length = min(ends, key=lambda x: x[0])
                state.thought += buffer[:end_pos]
                duration = time.perf_counter() - state.start_time
                state.total_think_time += duration
                state.in_think = False
                buffer = buffer[end_pos + tag_length:]
                if end_tag == '</reasoning>':
                    state.answer += buffer
                    break
            else:
                state.thought += buffer
                break
                
        elif state.in_answer:
            answer_end = buffer.find('</answer>')
            if answer_end != -1:
                state.answer += buffer[:answer_end]
                state.in_answer = False
                buffer = buffer[answer_end + 9:]
            else:
                state.answer += buffer
                break
    
    elapsed = time.perf_counter() - state.start_time if state.in_think else 0
    return state, elapsed

def format_response(state, elapsed):
    answer_part = state.answer
    collapsible = []
    collapsed = "<details open>"

    if state.thought or state.in_think:
        if state.in_think:
            total_elapsed = state.total_think_time + elapsed
            formatted_time = format_time(total_elapsed)
            status = f"💭 Thinking for {formatted_time}"
        else:
            formatted_time = format_time(state.total_think_time)
            status = f"✅ Thought for {formatted_time}"
            collapsed = "<details>"
        collapsible.append(
            f"{collapsed}<summary>{status}</summary>\n\n<div class='thinking-container'>\n{state.thought}\n</div>\n</details>"
        )
    return collapsible, answer_part

def remove_tags(text):
    if text is None:
        return None
    return re.sub(r'<[^>]+>', ' ', text).strip()