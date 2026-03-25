from typing import List, Dict, Tuple
import os
import re
import multiprocessing
from multiprocessing.connection import Connection
import io
from contextlib import redirect_stdout, redirect_stderr
import sys
from env.code_utils import PyExecutor, extract_python_code

def _transform_asserts(code: str) -> str:
    """
    Transform assert statements to provide better error messages.
    
    Converts: assert func(x) == y
    To: _result = func(x); _expected = y; assert _result == _expected, f"Got {_result!r}, expected {_expected!r}"
    """
    lines = code.split('\n')
    transformed = []
    counter = 0
    
    for line in lines:
        stripped = line.strip()
        # Match: assert expr == expected or assert expr != expected
        # Allow for optional trailing comment: # ...
        match = re.match(r'^assert\s+(.+?)\s*(==|!=)\s*(.+?)(?:\s*#.*)?$', stripped)
        if match:
            expr, op, expected = match.groups()
            indent = line[:len(line) - len(line.lstrip())]
            result_var = f"_assert_result_{counter}"
            expected_var = f"_assert_expected_{counter}"
            counter += 1
            # Generate transformed code with better error message
            # Evaluate both sides and show them in the error
            transformed.append(f"{indent}{result_var} = {expr}")
            transformed.append(f"{indent}{expected_var} = {expected}")
            if op == '==':
                transformed.append(f'{indent}assert {result_var} {op} {expected_var}, f"AssertionError: Got {{{result_var}!r}}, expected {{{expected_var}!r}}"')
            else:
                transformed.append(f'{indent}assert {result_var} {op} {expected_var}, f"AssertionError: Got {{{result_var}!r}}, should not equal {{{expected_var}!r}}"')
        else:
            transformed.append(line)
    
    return '\n'.join(transformed)


def _run_code_internal(code: str, conn: Connection):
    """Execution helper for the dynamic environment to return tracebacks/output."""
    try:
        # Transform assert statements for better error messages
        code = _transform_asserts(code)
        
        # Redirect stdout/stderr to capture output if any
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        # Pre-import only typing (required for function signatures)
        # Other imports (math, re, collections, etc.) must be handled by the model
        # per MBPP benchmark requirements
        global_ns = {}
        exec("from typing import List, Dict, Tuple, Optional, Any, Set, Union, Callable", global_ns)
        
        # Execute the code
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, global_ns)
        
        output = stdout.getvalue() + stderr.getvalue()
        if not output:
            conn.send((True, "[Executed Successfully with No Output]"))
        else:
            conn.send((True, output))
            
    except AssertionError as e:
        # For assertion errors, provide a cleaner message
        err_msg = str(e) if str(e) else "AssertionError: Assertion failed (no details available)"
        conn.send((False, err_msg))
    except Exception as e:
        import traceback
        # Get clean traceback, only keep <string> frames but preserve error type
        tb_lines = traceback.format_exc().split('\n')
        clean_lines = []
        skip_next = False
        for line in tb_lines:
            # Skip lines that reference internal files (not <string>)
            if 'File "' in line and '<string>' not in line:
                skip_next = True
                continue
            if skip_next and line.strip().startswith('exec('):
                skip_next = False
                continue
            skip_next = False
            clean_lines.append(line)
        
        err_msg = '\n'.join(clean_lines)
        conn.send((False, err_msg))
    finally:
        conn.close()

class MBPPEnv:
    def __init__(self, configs: Dict | None = None):
        self.configs = configs or {}
        self.executor = PyExecutor()

    def set_env(self, task_config: Dict) -> Tuple[str, str]:
        if task_config.get('answer') is None:
            # For MBPP, 'answer' might be the test list or the solution code
            pass
        if task_config.get("prompt") is None:
            raise ValueError('Please provide the prompt for the task')

        self.task_config = task_config
        self._reset()

        # Build the user prompt: problem text + first test case
        prompt = task_config["prompt"]
        test_list = task_config.get("test_list", [])
        
        if test_list:
            # Format: "problem text\nassert func_name(...) == ..."
            user_prompt = f"{prompt}\n{test_list[0]}"
        else:
            user_prompt = prompt

        return user_prompt

    def _reset(self):
        self.done = False
        self.reward = 0.0

    def step(self, action: str) -> Tuple[str, float, bool]:
        """
        Process an action (model output) and return observation.
        
        Args:
            action: The model's output containing <execute> or <answer> tags
            
        Returns:
            Tuple of (observation, reward, done)
        """
        action = self.preprocess_action(action)
        action_type, action_content = self._process_action(action)
        observation = ""

        if action_type == "execute":
            success, result = self._execute_code(action_content)
            observation = f"<observation>\n{result}\n</observation>"
            self.done = False
            self.reward = 0.0
        elif action_type == "answer":
            observation = ""
            self.done = True
            self.reward = self._check_answer_final(action_content)
        else:
            observation = "<observation>\nInvalid action format. Use <execute>...</execute> to run code or <answer>...</answer> to submit your solution.\n</observation>"
            self.done = False
            self.reward = 0.0

        return observation, self.reward, self.done

    def _execute_code(self, code: str, timeout: int = None) -> Tuple[bool, str]:
        # Allow timeout to be configured via environment variable for multiprocessing contexts
        if timeout is None:
            timeout = int(os.environ.get("REACT_EXEC_TIMEOUT", "5"))
        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(target=_run_code_internal, args=(code, child_conn))
        p.start()
        p.join(timeout)
        
        if p.is_alive():
            p.kill()
            p.join()
            return False, "TimeoutError: Execution timed out."
            
        if parent_conn.poll():
            return parent_conn.recv()
        else:
            return False, "RuntimeError: Execution failed unexpectedly."

    @classmethod
    def preprocess_action(cls, action: str) -> str:
        """
        Preprocess action to extract relevant content.
        Truncates to the first complete action block, preferring <execute>.
        """
        # If both tags are present, we look for the first closing tag to decide which one to keep
        exec_idx = action.find("</execute>")
        ans_idx = action.find("</answer>")
        
        if exec_idx != -1 and ans_idx != -1:
            # If both exist, prefer the one that comes first (execute)
            return action[:exec_idx + len("</execute>")]
        
        if exec_idx != -1:
            return action[:exec_idx + len("</execute>")]
        
        if ans_idx != -1:
            return action[:ans_idx + len("</answer>")]
        
        return action

    @classmethod
    def _process_action(cls, action: str) -> Tuple[str, str]:
        """
        Parse the action to determine type and content.
        
        Returns:
            Tuple of (action_type, action_content)
            action_type is one of: "execute", "answer", "invalid"
        """
        action = action.strip()
        
        # Check for execute tag (prefer the first one)
        exec_match = re.search(r"<execute>\s*(.*?)\s*</execute>", action, re.DOTALL)
        if exec_match:
            code = exec_match.group(1).strip()
            # Remove markdown code block wrapper if present
            code = re.sub(r"^```python\s*", "", code)
            code = re.sub(r"^```\s*", "", code)
            code = re.sub(r"```$", "", code)
            return "execute", code.strip()
        
        # Check for answer tag (if no execute was found)
        ans_match = re.search(r"<answer>\s*(.*?)\s*</answer>", action, re.DOTALL)
        if ans_match:
            code = ans_match.group(1).strip()
            # Remove markdown code block wrapper if present
            if "```python" in code:
                code_match = re.search(r"```python\s*(.*?)\s*```", code, re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip()
            elif "```" in code:
                code_match = re.search(r"```\s*(.*?)\s*```", code, re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip()
            return "answer", code
            
        return "invalid", action

    def _check_answer_final(self, answer: str) -> float:
        # answer is the function code provided in <answer>
        # We test it against the test_list in task_config
        tests = self.task_config.get("test_list", [])
        if not tests:
            return 1.0 # No tests to check?
            
        success, _, states = self.executor.execute(answer, tests)
        return 1.0 if success else 0.0

    def feedback(self) -> float:
        return self.reward

    @classmethod
    def compute_reward(cls, completions: List[str], envs: List['MBPPEnv'] = None, answer: List[List[str]] = None, **kwargs) -> List[float]:
        # Reward function for RL (e.g. GRPO)
        scores = []
        executor = PyExecutor()
        for i, completion in enumerate(completions):
            if envs is not None:
                tests = envs[i].task_config.get('test_list', [])
            elif answer is not None:
                tests = answer[i]
            else:
                raise ValueError("Both 'envs' and 'answer' are missing in reward computation.")

            matches = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            if not matches:
                scores.append(0.0)
                continue

            extracted_code = matches[-1].strip()
            # If it's wrapped in triple backticks inside <answer>, unwrap it
            if "```python" in extracted_code:
                extracted_code_match = re.search(r"```python(.*?)```", extracted_code, re.DOTALL)
                if extracted_code_match:
                    extracted_code = extracted_code_match.group(1).strip()
            
            success, _, _ = executor.execute(extracted_code, tests)
            if success:
                scores.append(1.0)
            else:
                scores.append(0.0)
        return scores
