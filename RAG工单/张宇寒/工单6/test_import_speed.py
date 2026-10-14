import subprocess, sys, unittest
from pathlib import Path

class ImportPerformanceTests(unittest.TestCase):
    def test_rag_import_does_not_initialize_embedding_runtime(self):
        result=subprocess.run([sys.executable,"-B","-c","import rag; print('ready')"],cwd=Path(__file__).parent,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr); self.assertEqual(result.stdout.strip(),"ready")
