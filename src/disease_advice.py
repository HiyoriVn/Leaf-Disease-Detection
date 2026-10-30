"""Safe, label-exact Markdown advice reader for the three crop models."""
from __future__ import annotations
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "disease_advise"


def get_advice(crop: str, model_label: str) -> str | None:
    if crop not in ("rice", "cassava", "coffee"):
        return None
    path = DIR / f"{crop}.md"
    if not path.is_file():
        return None
    content = path.read_text(encoding="utf-8")
    marker = "## " + model_label
    lines = content.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == marker), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start+1:end]).strip()


def show_advice(crop: str, label: str) -> None:
    advice = get_advice(crop, label)
    if advice:
        with st.expander("Khuyến nghị chăm sóc / xử lý — cần xác nhận thực địa", expanded=False):
            st.markdown(advice)
            st.caption("Thông tin tham khảo, không phải đơn thuốc. Kiểm tra nhãn đăng ký và hướng dẫn nông nghiệp tại địa phương.")
    else:
        st.caption("Chưa có khuyến nghị theo nhãn này; không tự suy diễn cách điều trị.")
