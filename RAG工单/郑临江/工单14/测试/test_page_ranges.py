"""工单14：定位到明确源码边界错误，验证单页、末页和全范围。"""
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parents[1]/'研发/ragflow-main/common/industrial_page_ranges.py'
spec=importlib.util.spec_from_file_location('industrial_pages',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_requested_figure_page_is_not_dropped():assert mod.normalize_pdf_page_range(7,7,8)==(6,7)
def test_last_page_is_included():assert mod.normalize_pdf_page_range(8,8,8)==(7,8)
def test_all_pages():assert mod.normalize_pdf_page_range(1,100000,8)==(0,8)
