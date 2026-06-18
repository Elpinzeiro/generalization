import argparse
import base64
import os
import re
from typing import Tuple

import requests
import urllib3
from dotenv import load_dotenv

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_CONFIG = {
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "key_env": "ANTHROPIC_API_KEY",
        "message_path": "/v1/messages",
        "headers": {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        },
    },
    "openai": {
        "base_url": "https://api.openai.com",
        "key_env": "OPENAI_API_KEY",
        "message_path": "/v1/responses",
        "headers": {
            "content-type": "application/json",
        },
    },
}


def _bit_reverse(e: int) -> int:
    return (
        (1 & e) << 7 | (2 & e) << 5 | (4 & e) << 3 | (8 & e) << 1
        | (16 & e) >> 1 | (32 & e) >> 3 | (64 & e) >> 5 | (128 & e) >> 7
    )


def _encode_char(e: int) -> str:
    HEX = "0123456789ABCDEF"
    if e == ord(" "):
        return "+"
    if (
        (e < ord("0") and e not in (ord("-"), ord(".")))
        or (ord("9") < e < ord("A"))
        or (ord("Z") < e < ord("a") and e != ord("_"))
        or e > ord("z")
    ):
        return f"%{HEX[e >> 4]}{HEX[e & 0xF]}"
    return chr(e)


def md6(text: str) -> str:
    return "".join(
        _encode_char(53 ^ _bit_reverse(ord(c)) ^ (0xFF & i))
        for i, c in enumerate(text)
    )


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def get_api_config(api_name: str) -> dict:
    if api_name not in API_CONFIG:
        raise ValueError(f"Unsupported api_name: {api_name}")
    return API_CONFIG[api_name]


def looks_like_proxy_warning(resp: requests.Response) -> bool:
    location = (resp.headers.get("Location", "") or "").lower()
    text = (resp.text or "").lower()

    markers = [
        "proxycontrolwarn",
        "sessionid",
        "httpwarning_",
        'id="pid"',
        'id="uid"',
    ]
    return any(m in location for m in markers) or any(m in text for m in markers)


def extract_proxy_params_from_response(
    session: requests.Session,
    resp: requests.Response,
    api_base_url: str,
    uid: str = "0",
) -> dict:
    location = resp.headers.get("Location", "")
    html = resp.text or ""
    source = location or html

    host_match = re.search(r'https?://([^/]+)/proxycontrolwarn/', source)
    if not host_match:
        raise ValueError(f"Could not determine proxy host from:\n{source[:500]}")
    proxy_host = host_match.group(1)

    pid_match = re.search(r'httpwarning_(\d+)', source) or re.search(
        r'id="pid"[^>]*value="([^"]+)"', html
    )
    if not pid_match:
        raise ValueError(f"Could not determine PID from:\n{source[:500]}")
    pid = pid_match.group(1)

    ori_url_b64 = b64(api_base_url)
    warn_url = (
        f"http://{proxy_host}/proxycontrolwarn/httpwarning_{pid}.html"
        f"?ori_url={ori_url_b64}&uid={uid}"
    )

    resp2 = session.get(warn_url, timeout=10)
    html2 = resp2.text

    session_id = re.search(r'id="sessionid"[^>]*value="([^"]+)"', html2)
    uid_match = re.search(r'id="uid"[^>]*value="([^"]+)"', html2)

    if not session_id:
        raise ValueError(f"Could not parse sessionid. Body preview:\n{html2[:500]}")

    return {
        "proxy_host": proxy_host,
        "session_id": session_id.group(1),
        "pid": pid,
        "uid": uid_match.group(1) if uid_match else uid,
        "ori_url_b64": ori_url_b64,
    }


def authorize_proxy(session: requests.Session, params: dict) -> requests.Session:
    payload = (
        f"ori_url={params['ori_url_b64']}"
        f"&sessionid={params['session_id']}"
        f"&pid={params['pid']}"
        f"&uid={params['uid']}"
    )
    obfuscated = md6(base64.b64encode(payload.encode()).decode())
    final = base64.b64encode(obfuscated.encode()).decode()

    resp = session.get(
        f"http://{params['proxy_host']}/proxycontrolwarn/check?{final}",
        timeout=10,
    )
    print(f"Proxy check: {resp.status_code}")
    return session


def build_openai_request(model_name: str, config: dict) -> Tuple[str, dict, dict]:
    api_key = os.environ[config["key_env"]]
    url = config["base_url"] + config["message_path"]
    headers = {
        **config["headers"],
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_name,
        "input": "hi",
        "max_output_tokens": 16,
    }
    return url, headers, payload


def build_anthropic_request(model_name: str, config: dict) -> Tuple[str, dict, dict]:
    api_key = os.environ[config["key_env"]]
    url = config["base_url"] + config["message_path"]
    headers = {
        **config["headers"],
        "x-api-key": api_key,
    }
    payload = {
        "model": model_name,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
    }
    return url, headers, payload


def perform_request_with_optional_bypass(
    api_name: str,
    model_name: str,
    config: dict,
) -> requests.Response:
    session = requests.Session()

    if api_name == "openai":
        url, headers, payload = build_openai_request(model_name, config)
    elif api_name == "anthropic":
        url, headers, payload = build_anthropic_request(model_name, config)
    else:
        raise ValueError(f"Unsupported api_name: {api_name}")

    resp = session.post(
        url,
        headers=headers,
        json=payload,
        timeout=30,
        verify=False,
        allow_redirects=False,
    )

    print(f"[probe] status={resp.status_code}")
    print(f"[probe] location={resp.headers.get('Location', '')!r}")
    print(f"[probe] body_head={resp.text[:300]!r}")

    if looks_like_proxy_warning(resp):
        print("[probe] Detected proxy warning, authorizing...")
        params = extract_proxy_params_from_response(
            session=session,
            resp=resp,
            api_base_url=config["base_url"],
        )
        print(
            f"Got params: proxy_host={params['proxy_host']}, "
            f"session_id={params['session_id']}, pid={params['pid']}"
        )

        authorize_proxy(session, params)
        print("Cookies after proxy auth:", session.cookies.get_dict())

        resp = session.post(
            url,
            headers=headers,
            json=payload,
            timeout=30,
            verify=False,
            allow_redirects=False,
        )
        print(f"[retry] status={resp.status_code}")
        print(f"[retry] body_head={resp.text[:300]!r}")

    return resp


def call_api_with_optional_bypass(api_name: str, model_name: str):
    config = get_api_config(api_name)
    resp = perform_request_with_optional_bypass(api_name, model_name, config)
    print(f"API: {resp.status_code}")
    print(resp.text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_name", type=str, default="anthropic")
    parser.add_argument("--model_name", type=str, default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    call_api_with_optional_bypass(args.api_name, args.model_name)


if __name__ == "__main__":
    main()