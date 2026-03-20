from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import urlparse


DEFAULT_NO_REPLY_TOKEN = "<NO_REPLY>"
DEFAULT_SYSTEM_PROMPT = (
    "你是一个微信私聊消息回复助手。"
    "请像真人微信聊天一样自然、简短地回复。"
    "只基于提供的未读消息和上下文回复，不要编造事实。"
    "不要输出 Markdown、标题、引号、解释或签名。"
    "如果不应该回复，只输出 {no_reply_token}。"
)


class ConfigError(RuntimeError):
    """配置错误。"""


@dataclass
class AssistantConfig:
    env_file: Path
    api_url: str
    api_key: str
    model: str
    api_header_user_agent: str
    api_header_origin: str
    api_header_referer: str
    timeout_seconds: float
    temperature: float
    no_reply_token: str
    system_prompt: str
    scan_interval_seconds: float
    load_delay_seconds: float
    operation_delay_seconds: float
    read_step_delay_seconds: float
    between_sessions_delay_seconds: float
    post_send_delay_seconds: float
    context_message_count: int
    max_history_messages: int
    search_pages: int
    is_maximize: bool
    send_delay_seconds: float
    close_wechat_on_exit: bool
    min_reply_interval_seconds: float
    exclude_friends: set[str]
    state_file: Path
    verbose: bool


class ReplyStateStore:
    """避免同一批未读消息被短时间重复回复。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def should_skip(
        self,
        friend: str,
        signature: str,
        min_reply_interval_seconds: float,
        now: float,
    ) -> bool:
        record = self.data.get(friend)
        if not isinstance(record, dict):
            return False
        previous_signature = str(record.get("signature", ""))
        previous_timestamp = float(record.get("timestamp", 0))
        if previous_signature != signature:
            return False
        return (now - previous_timestamp) < min_reply_interval_seconds

    def mark_processed(
        self,
        friend: str,
        signature: str,
        unread_messages: list[str],
        reply: str | None,
    ) -> None:
        self.data[friend] = {
            "signature": signature,
            "timestamp": time.time(),
            "unread_messages": unread_messages,
            "reply": reply or "",
        }
        self._save()


class OpenAICompatibleChatClient:
    """最小可用的 OpenAI-compatible chat completions 客户端。"""

    def __init__(self, config: AssistantConfig) -> None:
        self.api_url = config.api_url
        self.api_key = config.api_key
        self.model = config.model
        self.api_header_user_agent = config.api_header_user_agent
        self.api_header_origin = config.api_header_origin
        self.api_header_referer = config.api_header_referer
        self.timeout_seconds = config.timeout_seconds
        self.temperature = config.temperature
        self.no_reply_token = config.no_reply_token
        self.system_prompt = config.system_prompt

    @staticmethod
    def normalize_api_url(api_url: str) -> str:
        parsed = urlparse(api_url)
        path = parsed.path.rstrip("/")
        if path.endswith("/chat/completions"):
            return api_url
        if path == "":
            return api_url.rstrip("/") + "/v1/chat/completions"
        if path == "/v1":
            return api_url.rstrip("/") + "/chat/completions"
        return api_url

    def generate_reply(
        self,
        friend: str,
        unread_messages: list[str],
        context_messages: list[str],
    ) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "friend_name": friend,
                            "recent_context_messages": context_messages,
                            "current_unread_messages": unread_messages,
                            "requirements": [
                                "直接输出最终要发送的一段微信消息",
                                "默认使用自然口语中文",
                                "如果不该回复，只输出 no_reply_token",
                                "不要附加解释",
                            ],
                            "no_reply_token": self.no_reply_token,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": self.api_header_user_agent,
            "Origin": self.api_header_origin,
            "Referer": self.api_header_referer,
        }
        request_url = self.normalize_api_url(self.api_url)
        req = request.Request(request_url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                content_type = response.headers.get("Content-Type", "")
                response_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"AI 接口请求失败，HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"AI 接口网络异常: {exc}") from exc

        if "html" in content_type.lower() or response_body.lstrip().lower().startswith(
            "<!doctype html"
        ):
            raise RuntimeError(
                "AI 接口返回了 HTML 页面，通常说明 AI_API_URL 配置成了网站首页而不是接口地址。"
                f"当前请求地址: {request_url}。"
                "请优先检查 .env 中的 AI_API_URL，推荐填写完整接口地址，例如 "
                "https://your-domain/v1/chat/completions"
            )
        try:
            data = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"AI 接口返回了非 JSON 内容: {response_body}") from exc
        return self._extract_text(data)

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"AI 接口返回缺少 choices: {data}")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    texts.append(text)
            return "\n".join(texts).strip()
        raise RuntimeError(f"无法解析 AI 返回内容: {data}")


@dataclass
class MockSession:
    friend: str
    unread_messages: list[str]
    context_messages: list[str]


@dataclass
class ActiveChatCandidate:
    friend: str
    unread_messages: list[str]
    context_messages: list[str]


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.replace("\\n", "\n").strip()


def load_env_file(path: Path) -> None:
    if not path.exists():
        raise ConfigError(f"未找到配置文件: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            continue
        os.environ.setdefault(key.strip(), strip_quotes(value.strip()))


def get_env_str(name: str, default: str | None = None) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        if default is not None:
            return default
        raise ConfigError(f"缺少配置项: {name}")
    return value


def get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"{name} 必须是布尔值，当前为: {value}")


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是整数，当前为: {value}") from exc


def get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是数字，当前为: {value}") from exc


def parse_friend_list(raw_value: str) -> set[str]:
    return {item.strip() for item in raw_value.split(",") if item.strip()}


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def load_config(env_file: Path) -> AssistantConfig:
    load_env_file(env_file)
    base_dir = env_file.parent
    no_reply_token = get_env_str("AI_NO_REPLY_TOKEN", DEFAULT_NO_REPLY_TOKEN)
    raw_prompt = get_env_str("AI_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    try:
        system_prompt = raw_prompt.format(no_reply_token=no_reply_token)
    except KeyError as exc:
        raise ConfigError(f"AI_SYSTEM_PROMPT 中包含未知占位符: {exc}") from exc

    context_message_count = max(0, get_env_int("WECHAT_CONTEXT_MESSAGE_COUNT", 6))
    max_history_messages = max(
        context_message_count + 1,
        get_env_int("WECHAT_MAX_HISTORY_MESSAGES", 18),
    )
    state_file = resolve_path(
        base_dir,
        get_env_str("WECHAT_STATE_FILE", ".wechat_ai_reply_state.json"),
    )
    exclude_friends = parse_friend_list(get_env_str("WECHAT_EXCLUDE_FRIENDS", ""))

    return AssistantConfig(
        env_file=env_file,
        api_url=get_env_str("AI_API_URL"),
        api_key=get_env_str("AI_API_KEY"),
        model=get_env_str("AI_MODEL"),
        api_header_user_agent=get_env_str(
            "AI_HEADER_USER_AGENT",
            (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 "
                "Safari/537.36"
            ),
        ),
        api_header_origin=get_env_str("AI_HEADER_ORIGIN", "https://asooai.com"),
        api_header_referer=get_env_str("AI_HEADER_REFERER", "https://asooai.com/"),
        timeout_seconds=max(5.0, get_env_float("AI_TIMEOUT_SECONDS", 45.0)),
        temperature=get_env_float("AI_TEMPERATURE", 0.7),
        no_reply_token=no_reply_token,
        system_prompt=system_prompt,
        scan_interval_seconds=max(1.0, get_env_float("WECHAT_SCAN_INTERVAL_SECONDS", 10.0)),
        load_delay_seconds=max(1.0, get_env_float("WECHAT_LOAD_DELAY_SECONDS", 4.5)),
        operation_delay_seconds=max(
            0.0, get_env_float("WECHAT_OPERATION_DELAY_SECONDS", 1.0)
        ),
        read_step_delay_seconds=max(
            0.0, get_env_float("WECHAT_READ_STEP_DELAY_SECONDS", 0.2)
        ),
        between_sessions_delay_seconds=max(
            0.0, get_env_float("WECHAT_BETWEEN_SESSIONS_DELAY_SECONDS", 1.2)
        ),
        post_send_delay_seconds=max(
            0.0, get_env_float("WECHAT_POST_SEND_DELAY_SECONDS", 1.0)
        ),
        context_message_count=context_message_count,
        max_history_messages=max_history_messages,
        search_pages=max(0, get_env_int("WECHAT_SEARCH_PAGES", 5)),
        is_maximize=get_env_bool("WECHAT_IS_MAXIMIZE", False),
        send_delay_seconds=max(0.1, get_env_float("WECHAT_SEND_DELAY_SECONDS", 0.2)),
        close_wechat_on_exit=get_env_bool("WECHAT_CLOSE_WECHAT_ON_EXIT", False),
        min_reply_interval_seconds=max(
            0.0,
            get_env_float("WECHAT_MIN_REPLY_INTERVAL_SECONDS", 15.0),
        ),
        exclude_friends=exclude_friends,
        state_file=state_file,
        verbose=get_env_bool("WECHAT_VERBOSE", True),
    )


def normalize_messages(messages: list[str]) -> list[str]:
    normalized: list[str] = []
    for message in messages:
        clean = " ".join(str(message).split())
        if clean:
            normalized.append(clean)
    return normalized


def split_messages_by_unread_count(
    latest_first_messages: list[str],
    unread_count: int,
    context_message_count: int,
) -> tuple[list[str], list[str]]:
    unread_latest_first = latest_first_messages[: max(0, unread_count)]
    context_latest_first = latest_first_messages[
        unread_count : unread_count + context_message_count
    ]
    unread_messages = list(reversed(unread_latest_first))
    context_messages = list(reversed(context_latest_first))
    return normalize_messages(unread_messages), normalize_messages(context_messages)


def build_message_signature(friend: str, unread_messages: list[str]) -> str:
    payload = json.dumps(
        {"friend": friend, "unread_messages": unread_messages},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_reply_text(reply: str, no_reply_token: str) -> str:
    clean = reply.strip().strip('"').strip("'").strip()
    if not clean:
        return no_reply_token
    if clean.casefold() == no_reply_token.casefold():
        return no_reply_token
    return clean


def pause(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def load_mock_sessions(mock_file: Path) -> list[MockSession]:
    if not mock_file.exists():
        raise ConfigError(f"未找到 mock 文件: {mock_file}")
    try:
        payload = json.loads(mock_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"mock 文件不是合法 JSON: {mock_file}") from exc

    sessions_raw = payload.get("sessions") if isinstance(payload, dict) else payload
    if not isinstance(sessions_raw, list):
        raise ConfigError("mock 文件内容必须是数组，或者包含 sessions 数组的对象")

    sessions: list[MockSession] = []
    for index, item in enumerate(sessions_raw, start=1):
        if not isinstance(item, dict):
            raise ConfigError(f"mock 第 {index} 项必须是对象")
        friend = str(item.get("friend", "")).strip()
        unread_messages_raw = item.get("unread_messages", [])
        context_messages_raw = item.get("context_messages", [])
        if not friend:
            raise ConfigError(f"mock 第 {index} 项缺少 friend")
        if not isinstance(unread_messages_raw, list):
            raise ConfigError(f"mock 第 {index} 项的 unread_messages 必须是数组")
        if not isinstance(context_messages_raw, list):
            raise ConfigError(f"mock 第 {index} 项的 context_messages 必须是数组")
        unread_messages = normalize_messages([str(message) for message in unread_messages_raw])
        context_messages = normalize_messages([str(message) for message in context_messages_raw])
        if not unread_messages:
            raise ConfigError(f"mock 第 {index} 项至少要有一条 unread_messages")
        sessions.append(
            MockSession(
                friend=friend,
                unread_messages=unread_messages,
                context_messages=context_messages,
            )
        )
    return sessions


def configure_pyweixin(config: AssistantConfig, global_config: Any) -> None:
    global_config.is_maximize = config.is_maximize
    global_config.close_weixin = False
    global_config.search_pages = config.search_pages
    global_config.load_delay = float(config.load_delay_seconds)
    global_config.send_delay = float(config.send_delay_seconds)
    global_config.clear = True


def get_windows_startup_help(error_message: str) -> str:
    return (
        f"{error_message}\n"
        "Windows 启动顺序建议：\n"
        "1. 完全退出微信，并在任务管理器里结束 Weixin.exe / WeChatAppEx.exe。\n"
        "2. 在启动微信前开启讲述人，快捷键 Win + Ctrl + Enter。\n"
        "3. 保持讲述人开启，然后再启动并登录微信 4.1+。\n"
        "4. 登录后先保持几分钟，再运行本脚本。\n"
        "5. 建议先执行 python wechat_ai_reply_assistant.py --dry-run 做验证。\n"
        "6. 确保微信界面语言是简体中文，并且微信与脚本使用相同权限级别。"
    )


def process_session(
    config: AssistantConfig,
    ai_client: OpenAICompatibleChatClient,
    state_store: ReplyStateStore,
    friend: str,
    unread_messages: list[str],
    context_messages: list[str],
    dry_run: bool,
    send_reply: Any = None,
) -> None:
    if friend in config.exclude_friends:
        if config.verbose:
            log(f"跳过排除联系人: {friend}")
        return

    unread_messages = normalize_messages(unread_messages)
    context_messages = normalize_messages(context_messages)
    if not unread_messages:
        if config.verbose:
            log(f"未从会话中提取到可用文本，跳过: {friend}")
        return

    signature = build_message_signature(friend, unread_messages)
    now = time.time()
    if state_store.should_skip(
        friend=friend,
        signature=signature,
        min_reply_interval_seconds=config.min_reply_interval_seconds,
        now=now,
    ):
        if config.verbose:
            log(f"检测到重复未读消息，已跳过: {friend}")
        return

    if config.verbose:
        log(f"准备回复私聊消息: {friend}")
        for index, message in enumerate(unread_messages, start=1):
            log(f"未读 {index}: {message}")

    reply = ai_client.generate_reply(
        friend=friend,
        unread_messages=unread_messages,
        context_messages=context_messages,
    )
    reply = normalize_reply_text(reply, config.no_reply_token)
    if reply == config.no_reply_token:
        state_store.mark_processed(
            friend=friend,
            signature=signature,
            unread_messages=unread_messages,
            reply=None,
        )
        if config.verbose:
            log(f"AI 判断当前无需回复: {friend}")
        return

    if dry_run or send_reply is None:
        log(f"[dry-run] 将回复 {friend}: {reply}")
    else:
        send_reply(friend, reply)
        pause(config.post_send_delay_seconds)
        log(f"已回复 {friend}: {reply}")

    state_store.mark_processed(
        friend=friend,
        signature=signature,
        unread_messages=unread_messages,
        reply=reply,
    )


def pull_messages_from_current_window(
    current_window: Any,
    number: int,
    tools: Any,
    lists: Any,
    read_step_delay_seconds: float,
) -> list[str]:
    messages: list[str] = []
    chat_list = current_window.child_window(**lists.FriendChatList)
    if not chat_list.exists(timeout=0.5):
        return messages

    items = chat_list.children(control_type="ListItem")
    if not items:
        return messages

    messages.append(items[-1].window_text())
    tools.activate_chatList(chat_list)
    pause(read_step_delay_seconds)
    while len(messages) < number:
        chat_list.type_keys("{UP}")
        pause(read_step_delay_seconds)
        selected = [
            listitem
            for listitem in chat_list.children(control_type="ListItem")
            if listitem.has_keyboard_focus()
        ]
        if not selected:
            break
        selected_item = selected[0]
        if selected_item.class_name() != "mmui::ChatItemView":
            messages.append(selected_item.window_text())
    chat_list.type_keys("{END}")
    pause(read_step_delay_seconds)
    return messages[-number:]


def send_reply_in_current_window(
    current_window: Any,
    reply: str,
    modules: dict[str, Any],
    config: AssistantConfig,
) -> None:
    edit_area = current_window.child_window(**modules["Edits"].CurrentChatEdit)
    if not edit_area.exists(timeout=0.5):
        raise RuntimeError("当前聊天输入框不可用，无法发送回复")

    edit_area.click_input()
    pause(config.operation_delay_seconds)
    edit_area.set_text("")
    pause(config.read_step_delay_seconds)
    if 0 < len(reply) < 2000:
        modules["SystemSettings"].copy_text_to_clipboard(reply)
    else:
        modules["SystemSettings"].convert_long_text_to_txt(reply)
    pause(config.read_step_delay_seconds)
    modules["pyautogui"].hotkey("ctrl", "v", _pause=False)
    time.sleep(config.send_delay_seconds)
    modules["pyautogui"].hotkey("alt", "s", _pause=False)


def detect_active_chat_candidate(
    main_window: Any,
    modules: dict[str, Any],
    config: AssistantConfig,
) -> ActiveChatCandidate | None:
    tools = modules["Tools"]
    if tools.is_group_chat(main_window):
        return None

    current_chat_text = main_window.child_window(**modules["Texts"].CurrentChatText)
    edit_area = main_window.child_window(**modules["Edits"].CurrentChatEdit)
    chat_list = main_window.child_window(**modules["Lists"].FriendChatList)
    if not current_chat_text.exists(timeout=0.2):
        return None
    if not edit_area.exists(timeout=0.2) or not chat_list.exists(timeout=0.2):
        return None

    friend = current_chat_text.window_text().strip()
    if not friend:
        return None

    items = chat_list.children(control_type="ListItem")
    if not items:
        return None

    trailing_incoming_messages_latest_first: list[str] = []
    max_incoming_messages = max(1, min(5, config.max_history_messages))
    for item in reversed(items):
        text = " ".join(item.window_text().split())
        if not text:
            continue
        if item.class_name() == "mmui::ChatItemView":
            continue
        is_my_bubble = tools.is_my_bubble(main_window, item, edit_area)
        pause(config.read_step_delay_seconds)
        if is_my_bubble:
            break
        trailing_incoming_messages_latest_first.append(text)
        if len(trailing_incoming_messages_latest_first) >= max_incoming_messages:
            break

    if not trailing_incoming_messages_latest_first:
        return None

    history_count = min(
        config.max_history_messages,
        len(trailing_incoming_messages_latest_first) + config.context_message_count,
    )
    latest_first_messages = pull_messages_from_current_window(
        current_window=main_window,
        number=history_count,
        tools=tools,
        lists=modules["Lists"],
        read_step_delay_seconds=config.read_step_delay_seconds,
    )
    unread_messages, context_messages = split_messages_by_unread_count(
        latest_first_messages=latest_first_messages,
        unread_count=len(trailing_incoming_messages_latest_first),
        context_message_count=config.context_message_count,
    )
    if not unread_messages:
        return None
    return ActiveChatCandidate(
        friend=friend,
        unread_messages=unread_messages,
        context_messages=context_messages,
    )


def run_once(
    config: AssistantConfig,
    ai_client: OpenAICompatibleChatClient,
    state_store: ReplyStateStore,
    modules: dict[str, Any],
    dry_run: bool,
) -> None:
    navigator = modules["Navigator"]
    messages = modules["Messages"]
    tools = modules["Tools"]
    scan_for_new_messages = modules["scan_for_new_messages"]

    main_window = navigator.open_weixin(is_maximize=config.is_maximize)
    pause(config.operation_delay_seconds)
    active_chat_candidate = detect_active_chat_candidate(
        main_window=main_window,
        modules=modules,
        config=config,
    )
    unread_sessions = scan_for_new_messages(
        main_window=main_window,
        is_maximize=config.is_maximize,
        close_weixin=False,
    )
    if not unread_sessions and active_chat_candidate is None:
        if config.verbose:
            log("未发现未读消息。")
        return

    processed_friends: set[str] = set()
    if (
        active_chat_candidate is not None
        and active_chat_candidate.friend not in unread_sessions
    ):
        try:
            process_session(
                config=config,
                ai_client=ai_client,
                state_store=state_store,
                friend=active_chat_candidate.friend,
                unread_messages=active_chat_candidate.unread_messages,
                context_messages=active_chat_candidate.context_messages,
                dry_run=dry_run,
                send_reply=lambda target_friend, reply: send_reply_in_current_window(
                    current_window=main_window,
                    reply=reply,
                    modules=modules,
                    config=config,
                ),
            )
            processed_friends.add(active_chat_candidate.friend)
            pause(config.between_sessions_delay_seconds)
        except Exception as exc:
            log(f"处理当前聊天窗口失败 {active_chat_candidate.friend}: {exc}")
            pause(config.between_sessions_delay_seconds)

    for friend, unread_count in unread_sessions.items():
        if friend in processed_friends:
            continue
        try:
            if (
                active_chat_candidate is not None
                and friend == active_chat_candidate.friend
            ):
                current_window = main_window
                unread_messages = active_chat_candidate.unread_messages
                context_messages = active_chat_candidate.context_messages
            else:
                current_window = navigator.open_dialog_window(
                    friend=friend,
                    is_maximize=config.is_maximize,
                    search_pages=config.search_pages,
                )
                pause(config.operation_delay_seconds)
                if tools.is_group_chat(current_window):
                    pause(config.between_sessions_delay_seconds)
                    continue

                history_count = min(
                    config.max_history_messages,
                    max(unread_count, unread_count + config.context_message_count),
                )
                latest_first_messages = pull_messages_from_current_window(
                    current_window=current_window,
                    number=history_count,
                    tools=tools,
                    lists=modules["Lists"],
                    read_step_delay_seconds=config.read_step_delay_seconds,
                )
                unread_messages, context_messages = split_messages_by_unread_count(
                    latest_first_messages=latest_first_messages,
                    unread_count=unread_count,
                    context_message_count=config.context_message_count,
                )
            process_session(
                config=config,
                ai_client=ai_client,
                state_store=state_store,
                friend=friend,
                unread_messages=unread_messages,
                context_messages=context_messages,
                dry_run=dry_run,
                send_reply=lambda target_friend, reply: send_reply_in_current_window(
                    current_window=current_window,
                    reply=reply,
                    modules=modules,
                    config=config,
                ),
            )
            pause(config.between_sessions_delay_seconds)
        except Exception as exc:
            log(f"处理会话失败 {friend}: {exc}")
            pause(config.between_sessions_delay_seconds)


def run_mock_once(
    config: AssistantConfig,
    ai_client: OpenAICompatibleChatClient,
    state_store: ReplyStateStore,
    mock_file: Path,
) -> None:
    sessions = load_mock_sessions(mock_file)
    if config.verbose:
        log(f"已加载 {len(sessions)} 条 mock 会话。")
    for session in sessions:
        try:
            process_session(
                config=config,
                ai_client=ai_client,
                state_store=state_store,
                friend=session.friend,
                unread_messages=session.unread_messages,
                context_messages=session.context_messages,
                dry_run=True,
                send_reply=None,
            )
        except Exception as exc:
            log(f"处理 mock 会话失败 {session.friend}: {exc}")


def import_wechat_modules() -> dict[str, Any]:
    if os.name != "nt":
        raise RuntimeError("该脚本只能在 Windows 环境运行。")
    from pyweixin import GlobalConfig, Messages, Navigator
    from pyweixin.Errors import NotFoundError, NotLoginError, NotStartError
    from pyweixin.Uielements import Edits, Lists, Texts
    from pyweixin.WeChatTools import Tools
    from pyweixin.WinSettings import SystemSettings
    from pyweixin.utils import scan_for_new_messages
    import pyautogui

    return {
        "Edits": Edits(),
        "GlobalConfig": GlobalConfig,
        "Lists": Lists(),
        "Messages": Messages,
        "Navigator": Navigator,
        "NotFoundError": NotFoundError,
        "NotLoginError": NotLoginError,
        "NotStartError": NotStartError,
        "SystemSettings": SystemSettings,
        "Texts": Texts(),
        "Tools": Tools,
        "pyautogui": pyautogui,
        "scan_for_new_messages": scan_for_new_messages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微信 AI 私聊回复助手")
    parser.add_argument(
        "--env-file",
        default=".env",
        help="配置文件路径，默认使用仓库根目录下的 .env",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只扫描并处理一轮未读消息",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅生成回复并打印，不真正发送到微信",
    )
    parser.add_argument(
        "--mock-file",
        help="mock 会话 JSON 文件路径。可在 macOS 上测试 AI 回复链路。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_file = Path(args.env_file).expanduser()
    if not env_file.is_absolute():
        env_file = Path.cwd() / env_file

    try:
        config = load_config(env_file)
    except ConfigError as exc:
        log(f"配置错误: {exc}")
        return 1

    if args.mock_file:
        mock_file = Path(args.mock_file).expanduser()
        if not mock_file.is_absolute():
            mock_file = Path.cwd() / mock_file
        ai_client = OpenAICompatibleChatClient(config)
        state_store = ReplyStateStore(config.state_file)
        log("微信 AI 回复助手已进入 mock 测试模式。")
        try:
            run_mock_once(
                config=config,
                ai_client=ai_client,
                state_store=state_store,
                mock_file=mock_file,
            )
        except Exception as exc:
            log(f"mock 测试失败: {exc}")
            return 1
        return 0

    try:
        modules = import_wechat_modules()
    except Exception as exc:
        log(f"导入 pyweixin 失败: {exc}")
        return 1

    configure_pyweixin(config, modules["GlobalConfig"])
    ai_client = OpenAICompatibleChatClient(config)
    state_store = ReplyStateStore(config.state_file)

    log("微信 AI 私聊回复助手已启动。")
    if config.exclude_friends:
        log(f"排除联系人: {', '.join(sorted(config.exclude_friends))}")

    try:
        while True:
            cycle_started_at = time.time()
            try:
                run_once(
                    config=config,
                    ai_client=ai_client,
                    state_store=state_store,
                    modules=modules,
                    dry_run=args.dry_run,
                )
            except (
                modules["NotStartError"],
                modules["NotLoginError"],
                modules["NotFoundError"],
            ) as exc:
                log(get_windows_startup_help(str(exc)))
                return 1
            if args.once:
                break
            elapsed = time.time() - cycle_started_at
            sleep_seconds = max(1.0, config.scan_interval_seconds - elapsed)
            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        log("已收到停止信号，正在退出。")
    finally:
        if config.close_wechat_on_exit:
            try:
                if modules["Tools"].is_weixin_running():
                    window = modules["Navigator"].open_weixin(
                        is_maximize=config.is_maximize
                    )
                    window.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
