import os
import tempfile
import docker
import ast

class _AssertionCounter(ast.NodeTransformer):
    def _increment(self, index: int) -> ast.AugAssign:
        return ast.AugAssign(
            target=ast.Subscript(
                value=ast.Name(id="__assert_counts", ctx=ast.Load()),
                slice=ast.Constant(index),
                ctx=ast.Store(),
            ),
            op=ast.Add(),
            value=ast.Constant(1),
        )

    def visit_Assert(self, node: ast.Assert):
        return [
            self._increment(1),  # total executed assertions
            ast.Try(
                body=[node],
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id="Exception", ctx=ast.Load()),
                        name=None,
                        body=[ast.Pass()],
                    )
                ],
                # handlers = [
                #     ast.ExceptHandler(
                #         type=ast.Name(id="Exception", ctx=ast.Load()),
                #         name="exc",
                #         body=[
                #             ast.Expr(ast.Call(
                #                 func=ast.Name(id="print", ctx=ast.Load()),
                #                 args=[
                #                     ast.Constant(
                #                         value=f"First failure at test line {node.lineno}: "
                #                             f"{ast.unparse(node.test)}"
                #                     ),
                #                     ast.Name(id="exc", ctx=ast.Load()),
                #                     ast.Call(
                #                         func=ast.Name(id="locals", ctx=ast.Load()),
                #                         args=[],
                #                         keywords=[],
                #                     ),
                #                 ],
                #                 keywords=[],
                #             )),
                #             ast.Raise(exc=None, cause=None),
                #         ],
                #     )
                # ],
                orelse=[self._increment(0)],  # passed assertions
                finalbody=[],
            ),
        ]

class DockerSandbox:
    def __init__(self, image_name: str = "python:3.10-slim", timeout_seconds: int = 10):
        self.client = docker.from_env()
        self.image_name = image_name
        self.timeout = timeout_seconds

    def evaluate(self, solution_code: str, test_code: str, normalized=True, power=2) -> float:
        list_of_lines = test_code.split("\n")
        list_of_assert = [assert_statement for assert_statement in list_of_lines if "assert" in assert_statement]
        assert_stuff = ""
        for assert_str in list_of_assert:
            assert_stuff += self._assert_code_str_generation(assert_str)
        full_script = f"""\
{solution_code}

score = 0
{assert_stuff}

print(score)
exit(0)
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding="utf-8") as f:
            f.write(full_script)
            temp_file_path = f.name

        # Get the actual random filename inside the mounted /app directory
        script_name = os.path.basename(temp_file_path)

        container = None
        try:
            container = self.client.containers.run(
                image=self.image_name,
                command=f"python /app/{script_name}",
                volumes={os.path.dirname(temp_file_path): {'bind': '/app', 'mode': 'ro'}},
                working_dir='/app',
                network_mode="none",
                mem_limit="256m",
                cpu_period=100000,
                cpu_quota=50000,
                pids_limit=64,
                detach=True,
                remove=False
            )
            result = container.wait(timeout=self.timeout)
            logs = container.logs(stdout=True, stderr=True).decode('utf-8')
            try:
                score = float(logs.strip())
                norm_score = score / len(list_of_assert)
            except ValueError:
                raise Exception
            
            if result.get('StatusCode', -1) == 0:
                if normalized:
                    # 0.05 points for evaluate able to run, though max is 1.05
                    return pow(norm_score, power) + 0.05
                else:
                    return score
            return 0.0
        except Exception:
             return 0.0
        finally:
            if container:
                try: container.remove(force=True)
                except Exception: pass
            if os.path.exists(temp_file_path):
                try: os.unlink(temp_file_path)
                except Exception: pass

    def _assert_code_str_generation(self, assert_str):
        return f"""
try:
    {assert_str}
    score += 1
except AssertionError:
    pass
"""


    def evaluate_humaneval(
        self,
        solution_code: str,
        test_code: str,
        candidate_name_hint: str | None = None,
        problem_prompt: str = "",
    ) -> tuple[int, int]:
        try:
            test_tree = ast.parse(test_code)
            solution_tree = ast.parse(solution_code)
        except SyntaxError as exc:
            print(f"Invalid HumanEval code: {exc}")
            return 0, 1

        expected_total = sum(
            isinstance(node, ast.Assert)
            for node in ast.walk(test_tree)
        )

        generated_functions = [
            node.name
            for node in solution_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if candidate_name_hint in generated_functions:
            entry_point = candidate_name_hint
        elif "solution" in generated_functions:
            entry_point = "solution"
        elif len(generated_functions) == 1:
            entry_point = generated_functions[0]
        else:
            print("Could not determine which generated function to test.")
            return 0, expected_total

        if not any(
            isinstance(node, ast.FunctionDef) and node.name == "check"
            for node in test_tree.body
        ):
            print("HumanEval test code does not define check(candidate).")
            return 0, expected_total

        has_check_invocation = any(
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "check"
            for node in test_tree.body
        )

        test_tree = _AssertionCounter().visit(test_tree)
        ast.fix_missing_locations(test_tree)

        test_runner = "" if has_check_invocation else f"\ncheck({entry_point})\n"
        full_script = (
            "__assert_counts = [0, 0]\n"
            f"{problem_prompt}\n"
            f"{solution_code}\n"
            f"{ast.unparse(test_tree)}\n"
            f"{test_runner}"
            'print(f"__ASSERT_COUNTS__:{__assert_counts[0]}:{__assert_counts[1]}")\n'
        )

        temp_file_path = None
        container = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                temp_file.write(full_script)
                temp_file_path = temp_file.name

            container = self.client.containers.run(
                image=self.image_name,
                command=f"python /app/{os.path.basename(temp_file_path)}",
                volumes={
                    os.path.dirname(temp_file_path): {
                        "bind": "/app",
                        "mode": "ro",
                    }
                },
                working_dir="/app",
                network_mode="none",
                mem_limit="256m",
                cpu_period=100000,
                cpu_quota=50000,
                pids_limit=64,
                detach=True,
                remove=False,
            )

            result = container.wait(timeout=self.timeout)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8")

            marker = "__ASSERT_COUNTS__:"
            count_line = next(
                (line for line in reversed(logs.splitlines()) if line.startswith(marker)),
                None,
            )

            if result.get("StatusCode", -1) != 0 or count_line is None:
                print(f"HumanEval evaluation failed:\n{logs}")
                return 0, expected_total

            passed, total = map(int, count_line[len(marker):].split(":"))
            return passed, total

        except Exception as ex:
            print(f"Docker excpetion: {ex}")
            return 0, expected_total

        finally:
            if container is not None:
                container.remove(force=True)
            if temp_file_path is not None and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
