import os
import tempfile
import docker

class DockerSandbox:
    def __init__(self, image_name: str = "python:3.10-slim", timeout_seconds: int = 3):
        self.client = docker.from_env()
        self.image_name = image_name
        self.timeout = timeout_seconds

    def evaluate(self, solution_code: str, test_code: str, normalized=True) -> float:
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
                    return norm_score
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
