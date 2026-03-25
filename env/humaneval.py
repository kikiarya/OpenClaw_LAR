from typing import List, Dict, Tuple
import re
import multiprocessing
from multiprocessing.connection import Connection
import io
import signal
from contextlib import redirect_stdout, redirect_stderr

from env.code_utils import PyExecutor, extract_python_code


def _run_code_internal(code: str, conn: Connection, timeout: int = None):
    """Execution helper for the dynamic environment to return tracebacks/output."""
    try:
        if timeout:
            def handler(signum, frame):
                raise TimeoutError("Execution timed out.")
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)

        stdout = io.StringIO()
        stderr = io.StringIO()
        global_ns = {}
        
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, global_ns)
        
        output = stdout.getvalue() + stderr.getvalue()
        if not output:
            conn.send((True, "[Executed Successfully with No Output]"))
        else:
            conn.send((True, output))
            
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        conn.send((False, err_msg))
    finally:
        if timeout:
            signal.alarm(0)
        conn.close()


class HumanEvalEnv:
    """
    Dynamic environment for HumanEval code generation with Python execution tool.
    
    Supports iterative code development via <execute> tags for testing
    and <answer> tags for final submission.
    """

    def __init__(self):
        super().__init__()
        self.executor = PyExecutor()

    def set_env(self, task_config: Dict) -> Tuple[str, str]:
        """
        Initialize the environment with a HumanEval task.
        
        Args:
            task_config: Dict with keys 'prompt', 'test', 'entry_point'
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        if task_config.get("prompt") is None:
            raise ValueError("Please provide the 'prompt' for the task")

        self.task_config = task_config
        self._reset()

        user_prompt = f"Complete the following code: \n\n```python\n{task_config['prompt']}\n```"
        return user_prompt

    def _reset(self):
        """Reset environment state."""
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
            if success:
                observation = f"<observation>\n{result}\n</observation>"
            else:
                # For failed executions, the observation should contain the *entire*
                # raw error output (e.g., full Python traceback) without extra prefixes.
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

    def _execute_code(self, code: str, timeout: int = 5) -> Tuple[bool, str]:
        """Execute code in a sandboxed environment with timeout."""
        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(target=_run_code_internal, args=(code, child_conn, timeout))
        p.start()
        p.join(timeout + 1) # Give it 1 extra second to send the traceback
        
        if p.is_alive():
            p.kill()
            p.join()
            # If we're here, the child didn't send anything back in time (e.g. hung in C code)
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
            # If both exist, prefer the one that comes first, or prefer execute if they are same?
            # Actually, per plan: "if </execute> exists, keep everything through the first </execute>"
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
        """
        Check if the submitted answer passes the HumanEval tests.
        
        Returns:
            1.0 if all tests pass, 0.0 otherwise
        """
        test_code = self.task_config.get("test", "")
        entry_point = self.task_config.get("entry_point", "")
        
        if not test_code or not entry_point:
            return 0.0
        
        # Construct full test code
        full_code = f"{answer}\n\n{test_code}\n\ncheck({entry_point})"
        
        try:
            success, _ = self._execute_code(full_code, timeout=10)
            return 1.0 if success else 0.0
        except Exception:
            return 0.0

    def feedback(self) -> float:
        """Return the current reward."""
        return self.reward

    @classmethod
    def compute_reward(
        cls,
        completions: List[str],
        test: List[str],
        entry_point: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute reward for HumanEval completions (static evaluation).
        
        Args:
            completions: List of model completions (code solutions)
            test: List of test code strings
            entry_point: List of function names to test
            
        Returns:
            List of reward scores (1.0 for pass, 0.0 for fail)
        """
        py_executor = PyExecutor()
        scores = []

        for completion, test_code, func_name in zip(completions, test, entry_point):
            # Extract code from the completion
            answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            if answer_match:
                code = answer_match.group(1).strip()
                if "```python" in code:
                    code_blocks = extract_python_code(code)
                    code = "\n".join(code_blocks) if code_blocks else code
            else:
                code_blocks = extract_python_code(completion)
                if code_blocks:
                    code = "\n".join(code_blocks)
                else:
                    code = completion.strip()

            # Construct and run the test
            full_code = f"{code}\n\n{test_code}\n\ncheck({func_name})"
            
            try:
                success = py_executor.evaluate(func_name, code, f"{test_code}\n\ncheck({func_name})")
                scores.append(1.0 if success else 0.0)
            except Exception:
                scores.append(0.0)

        return scores
