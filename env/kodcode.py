import io
import multiprocessing
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from multiprocessing.connection import Connection
from typing import Dict, List, Tuple, Optional, Any

from env.base_env import StaticEnv, DynamicEnv
from env.code_utils import PyExecutor, extract_python_code


def _run_code_internal(code: str, conn: Connection):
    """
    Execution helper for the dynamic environment to return tracebacks/output.
    Kept similar to MBPP/HumanEval for consistency.
    """
    try:
        stdout = io.StringIO()
        stderr = io.StringIO()

        # Pre-import only typing helpers; models should import anything else they need.
        global_ns: Dict[str, Any] = {}
        exec("from typing import List, Dict, Tuple, Optional, Any, Set, Union, Callable, Iterable, Sequence", global_ns)

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, global_ns)

        output = stdout.getvalue() + stderr.getvalue()
        conn.send((True, output if output else "[Executed Successfully with No Output]"))
    except Exception:
        import traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_list = traceback.extract_tb(exc_traceback)
        clean_frames = [frame for frame in tb_list if frame.filename == "<string>"]
        cleaned_tb = "".join(traceback.format_list(clean_frames)) if clean_frames else ""
        err_msg = f"Traceback (most recent call last):\n{cleaned_tb}{exc_type.__name__}: {exc_value}"
        conn.send((False, err_msg))
    finally:
        conn.close()

class KodCodeEnv(StaticEnv):

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def _rename_func(cls, answer: str, function_name: str) -> str:
        """
        Replace the name of the first function in `answer` with `function_name`.
        Only modifies the function name, keeps everything else intact.
        """
        pattern = r"def\s+(\w+)\s*\("

        new_answer = re.sub(pattern, f"def {function_name}(", answer, count=1)
        return new_answer

    @classmethod
    def compute_reward(cls, completions: List[str], test: List[str], test_info: List, **kwargs) -> List[float]:
 
        py_executor = PyExecutor()
        scores = []
        for completion, t, tf in zip(completions, test, test_info): 
            func_blocks = extract_python_code(completion.strip())
            collected_answer = '\n'.join(func_blocks)
            function_name = None
            if isinstance(tf, list) and tf:
                function_name = tf[0].get("function_name")
            renamed_answer = cls._rename_func(collected_answer, function_name) if function_name else collected_answer
            _, _, results = py_executor.execute(renamed_answer, [t])

            score = sum(results) / len(results) if results else 0.0
            scores.append(score)
        
        return scores


class KodCodeReActEnv:
    """
    Dynamic environment for KodCode-V1 ReAct code generation with Python execution tool.
    Mirrors MBPP/HumanEval behavior but evaluates final answers using KodCode tests.
    """

    def __init__(self):
        super().__init__()
        self.executor = PyExecutor()

    def set_env(self, task_config: Dict) -> Tuple[str, str]:
        if task_config.get("prompt") is None:
            raise ValueError("Please provide the 'prompt' for the task")
        if task_config.get("test") is None:
            raise ValueError("Please provide the 'test' for the task")
        if task_config.get("test_info") is None:
            raise ValueError("Please provide the 'test_info' for the task")

        self.task_config = task_config
        self._reset()

        # from data.kodcode.builder import KODCODE_SYSTEM_PROMPT
        user_prompt = task_config["prompt"]
        return user_prompt

    def _reset(self) -> None:
        self.done = False
        self.reward = 0.0

    def step(self, action: str) -> Tuple[str, float, bool]:
        action = self.preprocess_action(action)
        action_type, action_content = self._process_action(action)

        if action_type == "execute":
            success, result = self._execute_code(action_content)
            observation = f"<observation>\n{result}\n</observation>"
            self.done = False
            self.reward = 0.0
            return observation, self.reward, self.done

        if action_type == "answer":
            self.done = True
            self.reward = self._check_answer_final(action_content)
            return "", self.reward, self.done

        observation = (
            "<observation>\nInvalid action format. Use <execute>...</execute> to run code "
            "or <answer>...</answer> to submit your solution.\n</observation>"
        )
        self.done = False
        self.reward = 0.0
        return observation, self.reward, self.done

    def _execute_code(self, code: str, timeout: Optional[int] = None) -> Tuple[bool, str]:
        if timeout is None:
            timeout = int(os.environ.get("REACT_EXEC_TIMEOUT", "5"))
        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(target=_run_code_internal, args=(code, child_conn))
        p.start()
        p.join(timeout + 1)

        if p.is_alive():
            p.kill()
            p.join()
            return False, "TimeoutError: Execution timed out."

        if parent_conn.poll():
            return parent_conn.recv()
        return False, "RuntimeError: Execution failed unexpectedly."

    @classmethod
    def preprocess_action(cls, action: str) -> str:
        exec_idx = action.find("</execute>")
        ans_idx = action.find("</answer>")

        if exec_idx != -1 and ans_idx != -1:
            return action[: exec_idx + len("</execute>")]
        if exec_idx != -1:
            return action[: exec_idx + len("</execute>")]
        if ans_idx != -1:
            return action[: ans_idx + len("</answer>")]
        return action

    @classmethod
    def _process_action(cls, action: str) -> Tuple[str, str]:
        action = action.strip()

        exec_match = re.search(r"<execute>\s*(.*?)\s*</execute>", action, re.DOTALL)
        if exec_match:
            code = exec_match.group(1).strip()
            code = re.sub(r"^```python\s*", "", code)
            code = re.sub(r"^```\s*", "", code)
            code = re.sub(r"```$", "", code)
            return "execute", code.strip()

        ans_match = re.search(r"<answer>\s*(.*?)\s*</answer>", action, re.DOTALL)
        if ans_match:
            code = ans_match.group(1).strip()
            if "```python" in code:
                m = re.search(r"```python\s*(.*?)\s*```", code, re.DOTALL)
                if m:
                    code = m.group(1).strip()
            elif "```" in code:
                m = re.search(r"```\s*(.*?)\s*```", code, re.DOTALL)
                if m:
                    code = m.group(1).strip()
            return "answer", code

        return "invalid", action

    def _check_answer_final(self, answer: str) -> float:
        test_code = self.task_config.get("test", "")
        test_info = self.task_config.get("test_info", None)

        # KodCode tests often import from "solution"; PyExecutor strips that import.
        # We still need to ensure the function name matches what tests expect.
        function_name = None
        if isinstance(test_info, list) and test_info:
            function_name = test_info[0].get("function_name")

        final_code = answer.strip()
        if function_name:
            final_code = KodCodeEnv._rename_func(final_code, function_name)

        try:
            success, _, _ = self.executor.execute(final_code, [test_code])
            return 1.0 if success else 0.0
        except Exception:
            return 0.0

    def feedback(self) -> float:
        return self.reward

    @classmethod
    def compute_reward(cls, completions: List[str], test: List[str], test_info: List, **kwargs) -> List[float]:
        """
        Static reward API for offline evaluation, mirroring KodCodeEnv.compute_reward.
        """
        return KodCodeEnv.compute_reward(completions=completions, test=test, test_info=test_info, **kwargs)
