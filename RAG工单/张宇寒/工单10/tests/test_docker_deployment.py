import shutil
import subprocess
import unittest
from pathlib import Path

import app as web


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("docker"), "Docker CLI未安装")
class DockerConfigurationTests(unittest.TestCase):
    def test_local_fallback_identifies_workorder10_and_uses_host_port(self):
        self.assertEqual(web.PORT, 8021)
        self.assertIn("工单10", web.app.title)
        page = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn("工单10 · Docker部署版", page)

    def test_compose_config_is_valid_and_hardened(self):
        result = subprocess.run(
            ["docker", "compose", "config"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        normalized = result.stdout
        self.assertIn('published: "8021"', normalized)
        self.assertIn("target: 8020", normalized)
        self.assertIn("read_only: true", normalized)
        self.assertIn("no-new-privileges:true", normalized)
        self.assertIn("healthcheck:", normalized)

    def test_build_context_never_contains_env_or_caches(self):
        patterns = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        self.assertIn(".env", patterns)
        self.assertIn("__pycache__", patterns)
        self.assertIn(".git", patterns)

    def test_dockerfile_runs_as_non_root_and_has_healthcheck(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("USER appuser", dockerfile)
        self.assertIn("HEALTHCHECK", dockerfile)
        self.assertIn('CMD ["uvicorn"', dockerfile)


if __name__ == "__main__":
    unittest.main()
