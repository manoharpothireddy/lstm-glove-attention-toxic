import os
import importlib
import io
import csv
import time
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf

from src.attention import AttentionPooling


st.set_page_config(page_title="Toxic Comment Classifier", layout="wide")

UI_BUILD = "2026-02-22-ui-v2"


def _safe_import_config(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


_DEFAULT_CONFIG_MODULE = os.environ.get("TOX_CONFIG", "config_best")
_available_configs = []
for _name in ["config_best", "config"]:
    if _safe_import_config(_name) is not None:
        _available_configs.append(_name)
if _DEFAULT_CONFIG_MODULE not in _available_configs:
    _available_configs = [_DEFAULT_CONFIG_MODULE] + _available_configs


def _pick_best_config_module() -> str:
    for name in ["config_best", "config"]:
        cfg = _safe_import_config(name)
        if cfg is None:
            continue
        try:
            if (cfg.ARTIFACTS_DIR / "model.keras").exists():
                return name
        except Exception:
            continue
    return _available_configs[0] if _available_configs else "config_best"


@st.cache_resource
def _load_model_and_tokenizer(config_module: str):
    cfg = importlib.import_module(config_module)
    model = tf.keras.models.load_model(
        cfg.ARTIFACTS_DIR / "model.keras",
        custom_objects={"AttentionPooling": AttentionPooling},
    )

    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
        (cfg.ARTIFACTS_DIR / "tokenizer.json").read_text(encoding="utf-8")
    )
    labels = (cfg.ARTIFACTS_DIR / "labels.json").read_text(encoding="utf-8")
    labels = __import__("json").loads(labels)
    return cfg, model, tokenizer, labels


def _chip_html(text: str, color: str) -> str:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<span style="display:inline-block;padding:0.22rem 0.55rem;border-radius:999px;'
        f'border:1px solid {color};margin-right:0.4rem;margin-bottom:0.4rem;'
        f'background:rgba(255,255,255,0.02);font-size:0.85rem;">{safe}</span>'
    )


def _render_author_assist(
    *,
    cfg,
    model,
    tokenizer,
    labels,
    config_module: str,
    auto_score: bool,
    threshold: float,
    preset: str,
):
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="topbar">'
        f'<div>'
        f'<div class="subtle">Be respectful.</div>'
        f"</div>"
        f'<div class="userline"><div class="avatar">{_initials(st.session_state["username"])}</div><div><div class="feed-author">u/{st.session_state["username"]}</div><div class="subtle">Author assist mode</div></div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="brand">Write a comment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tagline">You’ll get real-time feedback before posting.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.4, 1])

    with left:
        default_text = ""
        if preset != "(custom)":
            default_text = preset.split(": ", 1)[1]
        if not st.session_state.get("single_text") and default_text:
            st.session_state["single_text"] = default_text

        st.markdown("**Your comment**")
        text = st.text_area(
            "",
            value=st.session_state.get("single_text", ""),
            key="single_text",
            placeholder="Type your comment here...",
            height=170,
            label_visibility="collapsed",
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            score_clicked = st.button("Check", type="primary", use_container_width=True)
        with c2:
            clear_clicked = st.button("Clear", use_container_width=True)

        if clear_clicked:
            st.session_state["single_text"] = ""
            st.session_state["single_last_text"] = None
            st.session_state["single_last_profile"] = None
            st.session_state["single_last_probs"] = None
            st.session_state["single_last_latency_ms"] = None
            st.session_state["single_last_threshold"] = None
            st.rerun()

    with right:
        st.markdown("<div class=\"card\">", unsafe_allow_html=True)
        st.markdown("**Safety check**")
        st.markdown(
            "<div class=\"muted\">This panel shows the model’s risk estimate for your comment. Open Details to see label probabilities.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    should_score = False
    log_history = False
    if auto_score:
        if score_clicked:
            should_score = True
            log_history = bool(text.strip())
        elif text.strip() and (
            st.session_state["single_last_text"] != text
            or st.session_state["single_last_profile"] != config_module
        ):
            should_score = True
    else:
        if score_clicked:
            should_score = True
            log_history = bool(text.strip())

    if should_score:
        t0 = time.perf_counter()
        probs = _predict([text], model, tokenizer, max_len=int(cfg.MAX_LEN))[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        st.session_state["single_last_text"] = text
        st.session_state["single_last_profile"] = config_module
        st.session_state["single_last_probs"] = probs
        st.session_state["single_last_latency_ms"] = float(latency_ms)
        st.session_state["single_last_threshold"] = float(threshold)

        if log_history:
            max_prob = float(np.max(probs))
            severity_label, severity_color, _ = _severity_from_max_prob(max_prob)
            top_idx = int(np.argmax(probs))
            top_label = labels[top_idx]
            st.session_state["history_by_user"][st.session_state["username"]].insert(
                0,
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "username": st.session_state["username"],
                    "text": text,
                    "severity": severity_label,
                    "max_probability": float(max_prob),
                    "color": severity_color,
                    "top_label": top_label,
                },
            )

    cached_probs = st.session_state.get("single_last_probs")
    have_probs = cached_probs is not None and st.session_state.get("single_last_profile") == config_module

    if have_probs:
        probs = cached_probs
        max_prob = float(np.max(probs))
        severity_label, severity_color, _ = _severity_from_max_prob(max_prob)

        top_idx = int(np.argmax(probs))
        top_label = labels[top_idx]

        flagged = bool(max_prob >= threshold)
        dot_color = severity_color

        with right:
            st.markdown("<div class=\"card\">", unsafe_allow_html=True)
            st.markdown(_risk_pill_html(severity_label, dot_color, max_prob), unsafe_allow_html=True)
            if st.session_state.get("single_last_latency_ms") is not None:
                st.caption(f"Latency: {st.session_state['single_last_latency_ms']:.0f} ms")

            if flagged:
                st.markdown(
                    "<div class=\"hint\"><strong>Suggestion</strong><br/>Consider rewriting to be more respectful before posting.</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Most influential label: `{top_label}`")
            else:
                st.markdown(
                    "<div class=\"hint\"><strong>Looks OK</strong><br/>This comment seems unlikely to violate guidelines.</div>",
                    unsafe_allow_html=True,
                )

            with st.expander("Details (label breakdown)", expanded=False):
                order = np.argsort(-probs)
                rows = []
                for idx in order:
                    p = float(probs[idx])
                    rows.append(
                        {
                            "label": labels[idx],
                            "probability": round(p, 6),
                            "above_threshold": bool(p >= threshold),
                        }
                    )
                st.dataframe(rows, use_container_width=True, height=260)
                st.download_button(
                    "Download details (CSV)",
                    data=_to_csv_bytes(rows),
                    file_name="toxicity_details.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.divider()
            f1, f2 = st.columns(2)
            with f1:
                wrong_clicked = st.button("This is wrong", use_container_width=True)
            with f2:
                ok_clicked = st.button("This is helpful", use_container_width=True)

            if wrong_clicked or ok_clicked:
                st.session_state["feedback_by_user"][st.session_state["username"]].append(
                    {
                        "text": text,
                        "profile": config_module,
                        "max_probability": float(max_prob),
                        "severity": severity_label,
                        "feedback": "wrong" if wrong_clicked else "helpful",
                    }
                )
                st.success("Thanks — feedback recorded locally for this session.")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        with right:
            st.markdown("<div class=\"card\">", unsafe_allow_html=True)
            st.info("Start typing to get real-time feedback.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("**Recent checks**")
    active_history = st.session_state.get("history_by_user", {}).get(st.session_state["username"], [])
    if active_history:
        for item in active_history[:20]:
            st.markdown("<div class=\"feed-item\">", unsafe_allow_html=True)
            st.markdown(
                f'<div class="feed-meta">'
                f'<div><span class="feed-author">u/{item["username"]}</span> <span class="feed-time">• {item["timestamp"]}</span></div>'
                f'{_risk_pill_html(item["severity"], item["color"], float(item["max_probability"]))}'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="feed-text">{item["text"]}</div>', unsafe_allow_html=True)
            st.caption(f"Top label: `{item['top_label']}`")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.caption("No checks yet. Write a comment and click Check.")

    with st.expander("Session feedback log", expanded=False):
        active_feedback = st.session_state.get("feedback_by_user", {}).get(st.session_state["username"], [])
        if active_feedback:
            st.dataframe(active_feedback, use_container_width=True)
            st.download_button(
                "Download feedback log (CSV)",
                data=_to_csv_bytes(active_feedback),
                file_name="feedback_log.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("No feedback submitted yet.")

    st.markdown("</div>", unsafe_allow_html=True)




def _predict(texts, model, tokenizer, max_len: int):
    seq = tokenizer.texts_to_sequences(texts)
    x = tf.keras.preprocessing.sequence.pad_sequences(
        seq,
        maxlen=max_len,
        padding="post",
        truncating="post",
    )
    probs = model.predict(x, verbose=0)
    return probs


def _label_color(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "#16a34a"
    if prob >= threshold * 0.75:
        return "#f59e0b"
    return "#64748b"


def _render_probability_row(label: str, prob: float, threshold: float):
    left, right = st.columns([1, 3], vertical_alignment="center")
    with left:
        st.markdown(f"**{label}**")
        st.caption(f"{prob:.4f}")
    with right:
        st.progress(float(prob), text=None)


def _severity_from_max_prob(max_prob: float) -> tuple[str, str, float]:
    if max_prob >= 0.90:
        return "High", "#ef4444", 1.0
    if max_prob >= 0.60:
        return "Medium", "#f59e0b", 0.7
    if max_prob >= 0.30:
        return "Low", "#22c55e", 0.45
    return "Minimal", "#64748b", 0.2


def _to_csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _initials(name: str) -> str:
    parts = [p for p in name.strip().split() if p]
    if not parts:
        return "U"
    if len(parts) == 1:
        return parts[0][:1].upper()
    return (parts[0][:1] + parts[-1][:1]).upper()


def _risk_pill_html(severity_label: str, color: str, max_prob: float) -> str:
    return (
        f'<div class="pill" style="border-color: rgba(255,255,255,0.14);">'
        f'<span class="pill-dot" style="background:{color}"></span>'
        f'<strong>{severity_label} risk</strong>'
        f'<span class="muted">max {max_prob:.2f}</span>'
        f"</div>"
    )


def _ensure_state():
    st.session_state.setdefault("single_text", "")
    st.session_state.setdefault("single_last_text", None)
    st.session_state.setdefault("single_last_profile", None)
    st.session_state.setdefault("single_last_probs", None)
    st.session_state.setdefault("single_last_latency_ms", None)
    st.session_state.setdefault("single_last_threshold", None)
    st.session_state.setdefault("users", ["manohar"])
    st.session_state.setdefault("active_user", "manohar")
    st.session_state.setdefault("history_by_user", {})
    st.session_state.setdefault("feedback_by_user", {})
    st.session_state.setdefault("adding_user", False)
    st.session_state.setdefault("confirming_delete", False)
    st.session_state.setdefault("new_user_name", "")

    active = st.session_state.get("active_user")
    if not active:
        active = st.session_state["users"][0]
        st.session_state["active_user"] = active

    st.session_state["history_by_user"].setdefault(active, [])
    st.session_state["feedback_by_user"].setdefault(active, [])

    st.session_state.setdefault("username", active)


st.markdown(
    """
<style>
  .wrap { max-width: 980px; margin: 0 auto; }
  .brand { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.1rem; }
  .tagline { color: rgba(255,255,255,0.65); margin-bottom: 1.0rem; }
  .topbar { display:flex; align-items:center; justify-content:space-between; gap: 10px; margin-bottom: 0.75rem; }
  .subtle { color: rgba(255,255,255,0.65); font-size: 0.95rem; }
  .avatar { width: 34px; height: 34px; border-radius: 999px; display:flex; align-items:center; justify-content:center; font-weight: 800; background: rgba(59,130,246,0.25); border: 1px solid rgba(59,130,246,0.35); }
  .userline { display:flex; align-items:center; gap: 10px; }
  .pill { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0.65rem; border-radius: 999px; border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.03); font-size: 0.95rem; }
  .pill-dot { width: 10px; height: 10px; border-radius: 999px; }
  .card { border: 1px solid rgba(255,255,255,0.10); border-radius: 14px; padding: 14px 14px; background: rgba(255,255,255,0.02); }
  .muted { color: rgba(255,255,255,0.65); }
  .hint { padding: 10px 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.10); background: rgba(255,255,255,0.02); }
  .feed-item { border: 1px solid rgba(255,255,255,0.10); border-radius: 14px; padding: 12px 12px; background: rgba(255,255,255,0.015); margin-bottom: 10px; }
  .feed-meta { display:flex; align-items:center; justify-content:space-between; gap: 10px; margin-bottom: 6px; }
  .feed-author { font-weight: 750; }
  .feed-time { color: rgba(255,255,255,0.55); font-size: 0.9rem; }
  .feed-text { white-space: pre-wrap; line-height: 1.35; }
</style>
""",
    unsafe_allow_html=True,
)

_ensure_state()

with st.sidebar:
    st.header("Author Assist")
    st.subheader("Profile")
    users = st.session_state.get("users", ["manohar"])
    active_user = st.selectbox("User", users, index=max(users.index(st.session_state.get("active_user", users[0])), 0))
    st.session_state["active_user"] = active_user
    st.session_state["username"] = active_user

    if not st.session_state.get("adding_user") and not st.session_state.get("confirming_delete"):
        btn_add, btn_del = st.columns([1, 1])
        with btn_add:
            if st.button("Add user", use_container_width=True):
                st.session_state["adding_user"] = True
                st.session_state["new_user_name"] = ""
                st.rerun()
        with btn_del:
            if len(users) > 1:
                if st.button("Delete user", use_container_width=True, type="secondary"):
                    st.session_state["confirming_delete"] = True
                    st.rerun()
            else:
                st.button("Delete user", use_container_width=True, disabled=True)

    elif st.session_state.get("confirming_delete"):
        st.warning(f'Delete **u/{active_user}** and all their data?')
        c_yes, c_no = st.columns([1, 1])
        with c_yes:
            confirm_clicked = st.button("Yes, delete", type="primary", use_container_width=True)
        with c_no:
            deny_clicked = st.button("Cancel", use_container_width=True)

        if deny_clicked:
            st.session_state["confirming_delete"] = False
            st.rerun()

        if confirm_clicked:
            u = active_user
            st.session_state["users"].remove(u)
            st.session_state["history_by_user"].pop(u, None)
            st.session_state["feedback_by_user"].pop(u, None)
            st.session_state["active_user"] = st.session_state["users"][0]
            st.session_state["username"] = st.session_state["users"][0]
            st.session_state["confirming_delete"] = False
            # Clear cached scoring for deleted user
            st.session_state["single_last_text"] = None
            st.session_state["single_last_probs"] = None
            st.session_state["single_last_profile"] = None
            st.session_state["single_last_latency_ms"] = None
            st.session_state["single_last_threshold"] = None
            st.rerun()

    else:
        st.text_input("New username", key="new_user_name")
        c_add, c_cancel = st.columns([1, 1])
        with c_add:
            add_clicked = st.button("Create", type="primary", use_container_width=True)
        with c_cancel:
            cancel_clicked = st.button("Cancel", use_container_width=True)

        if cancel_clicked:
            st.session_state["adding_user"] = False
            del st.session_state["new_user_name"]
            st.rerun()

        if add_clicked:
            u = (st.session_state.get("new_user_name") or "").strip()
            if not u:
                st.warning("Please enter a username.")
            else:
                if u not in st.session_state["users"]:
                    st.session_state["users"].append(u)
                st.session_state["active_user"] = u
                st.session_state["username"] = u
                st.session_state["history_by_user"].setdefault(u, [])
                st.session_state["feedback_by_user"].setdefault(u, [])
                st.session_state["adding_user"] = False
                del st.session_state["new_user_name"]
                st.rerun()

    config_module = _pick_best_config_module()
    st.caption(f"Model: `{config_module}`")
    auto_score = st.toggle("Real-time scoring", value=True)
    threshold = st.slider(
        "Flag threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
    )
    st.divider()
    preset = st.selectbox(
        "Example comments",
        [
            "(custom)",
            "Neutral: I love this product.",
            "Insult: You are an idiot.",
            "Threat: I will kill you.",
            "Hate: People like you should not exist.",
        ],
        index=0,
    )
    st.divider()
    st.caption(f"UI build: `{UI_BUILD}`")
    st.caption("This is a demo assistant. For real moderation, keep a human in the loop.")

cfg, model, tokenizer, labels = _load_model_and_tokenizer(config_module)

_render_author_assist(
    cfg=cfg,
    model=model,
    tokenizer=tokenizer,
    labels=labels,
    config_module=config_module,
    auto_score=auto_score,
    threshold=float(threshold),
    preset=preset,
)
