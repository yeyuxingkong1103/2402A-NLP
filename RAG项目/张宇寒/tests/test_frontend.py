from html.parser import HTMLParser
from urllib.parse import urlsplit
from pathlib import Path
import re
import subprocess

from fastapi.testclient import TestClient

from backend.app.main import app


class PageElements(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.elements = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def homepage():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    return PageElements(response.text)


def test_deferred_features_are_not_navigation_destinations():
    for _, attrs in homepage().elements:
        assert attrs.get("data-route-target") not in {"files", "documents"}
        assert attrs.get("href") not in {"#files", "#documents"}


def test_upload_controls_are_available_in_consultation_ui():
    elements = {attrs["id"]: attrs for _, attrs in homepage().elements if "id" in attrs}
    assert "hidden" not in elements["thinking-toggle-button"]
    assert "disabled" not in elements["thinking-toggle-button"]
    assert "disabled" not in elements["file-input"]
    assert ".pdf" in elements["file-input"]["accept"]
    assert ".docx" in elements["file-input"]["accept"]
    assert ".png" in elements["file-input"]["accept"]
    for name in ("material-drawer", "material-drawer-toggle"):
        assert "hidden" not in elements[name]
        assert "inert" not in elements[name]
    assert "hidden" in elements["files-page"]
    assert "inert" in elements["files-page"]


def test_consultation_toolbar_explains_each_control_with_its_own_aligned_hint():
    elements = {attrs["id"]: attrs for _, attrs in homepage().elements if "id" in attrs}

    assert elements["thinking-toggle-button"]["aria-describedby"] == "file-status"
    assert elements["ask-retrieval-mode"]["aria-describedby"] == "retrieval-mode-status"
    assert elements["composer-material-control"]["role"] == "group"
    assert elements["composer-retrieval-control"]["role"] == "group"


def test_homepage_has_an_accessible_question_form():
    elements = {attrs["id"]: attrs for _, attrs in homepage().elements if "id" in attrs}
    assert "home-question-form" in elements
    assert elements["home-question-input"]["aria-label"]
    assert elements["home-question-input"]["maxlength"] == "4000"
    assert "required" in elements["home-question-input"]


def test_question_composers_offer_three_retrieval_modes():
    page = homepage()
    modes = [
        attrs.get("data-retrieval-mode")
        for _, attrs in page.elements
        if attrs.get("data-retrieval-mode")
    ]

    assert modes.count("local") == 2
    assert modes.count("auto") == 2
    assert modes.count("force") == 2


def test_settings_use_visible_answer_detail_cards_and_no_web_checkbox():
    page = homepage()
    elements = {attrs["id"]: attrs for _, attrs in page.elements if "id" in attrs}
    detail_values = {
        attrs.get("value")
        for _, attrs in page.elements
        if attrs.get("name") == "answer-detail"
    }

    assert "setting-include-web" not in elements
    assert "setting-answer-detail" not in elements
    assert detail_values == {"concise", "standard", "detailed"}


def test_homepage_can_add_evidence_before_the_first_question():
    """防止上传入口只存在于聊天页，用户在首页描述案情时找不到证据上传。"""
    elements = {attrs["id"]: attrs for _, attrs in homepage().elements if "id" in attrs}
    upload = elements["home-material-upload"]
    assert upload["type"] == "button"
    assert upload["aria-label"] == "添加证据材料"
    assert "hidden" not in upload
    assert "disabled" not in upload
    assert elements["home-material-summary"]["aria-live"] == "polite"


def test_home_composer_groups_actions_and_guidance_into_two_clear_rows():
    elements = {attrs["id"]: attrs for _, attrs in homepage().elements if "id" in attrs}

    assert elements["home-composer-toolbar"]["role"] == "toolbar"
    assert elements["home-composer-toolbar"]["aria-label"] == "提问工具"
    assert elements["home-composer-meta"]["aria-label"] == "检索与隐私提示"
    assert elements["home-composer-privacy"]["role"] == "note"


def test_home_evidence_button_has_readable_contrast_on_the_light_composer():
    """Catch a light-on-light upload button that looks empty to users."""
    css = Path("frontend/src/brand.css").read_text(encoding="utf-8")
    variables = dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})", css))
    button_rule = re.search(
        r"\.zhitu-home-composer-actions \.home-material-upload\s*\{([^}]*)\}",
        css,
    )
    assert button_rule
    color_variable = re.search(r"\bcolor:\s*var\((--[\w-]+)\)", button_rule.group(1))
    assert color_variable

    def luminance(hex_color):
        channels = [int(hex_color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        channels = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4 for value in channels]
        return .2126 * channels[0] + .7152 * channels[1] + .0722 * channels[2]

    foreground = luminance(variables[color_variable.group(1)])
    background = luminance(variables["--warm"])
    contrast = (max(foreground, background) + .05) / (min(foreground, background) + .05)
    assert contrast >= 4.5, f"Evidence button contrast is only {contrast:.2f}:1"


def test_home_examples_appear_before_the_question_composer():
    elements = homepage().elements
    examples_position = next(
        index
        for index, (_, attrs) in enumerate(elements)
        if "zhitu-examples" in attrs.get("class", "").split()
    )
    composer_position = next(
        index
        for index, (_, attrs) in enumerate(elements)
        if attrs.get("id") == "home-question-form"
    )

    assert examples_position < composer_position


def test_login_is_initially_a_closed_dialog_not_homepage_content():
    elements = {attrs["id"]: attrs for _, attrs in homepage().elements if "id" in attrs}
    assert "hidden" in elements["auth-backdrop"]
    assert elements["auth-dialog"]["role"] == "dialog"
    assert elements["auth-dialog"]["aria-modal"] == "true"
    assert "auth-close-button" in elements


def test_sidebar_has_no_standalone_current_consultation_entry():
    entries = [
        attrs for _, attrs in homepage().elements
        if attrs.get("href") == "#ask" or attrs.get("data-route-target") == "ask"
    ]
    assert entries == [], "New and recent consultations already open the shared workspace"


def test_sidebar_keeps_home_new_consultation_and_recent_consultations():
    page = homepage()
    navigation = [
        attrs.get("data-route-target")
        for _, attrs in page.elements
        if attrs.get("class") == "route-link"
    ]
    elements = {attrs["id"]: attrs for _, attrs in page.elements if "id" in attrs}

    assert navigation == ["home"]
    assert "hidden" not in elements["new-chat-button"]
    assert elements["task-list"]["role"] == "list"
    assert "最近咨询" in elements["task-list"]["aria-label"]


def test_home_navigation_and_question_submission_have_distinct_accessible_names():
    page = homepage()
    navigation = {attrs["data-route-target"]: attrs for _, attrs in page.elements if attrs.get("class") == "route-link"}
    elements = {attrs["id"]: attrs for _, attrs in page.elements if "id" in attrs}
    assert navigation["home"].get("aria-label") == "咨询首页"
    assert elements["new-chat-button"].get("aria-label") == "新建法律咨询"
    assert elements["home-question-submit"].get("aria-label") == "发送问题"
    assert elements["home-question-submit"]["type"] == "submit"


def test_new_consultation_has_no_second_visible_creation_control():
    elements = {attrs["id"]: attrs for _, attrs in homepage().elements if "id" in attrs}
    assert "hidden" in elements["new-task-compact"]
    assert "hidden" not in elements["new-chat-button"]


def test_initial_login_does_not_require_registration_email():
    elements = {attrs["id"]: attrs for _, attrs in homepage().elements if "id" in attrs}
    assert "hidden" in elements["auth-email-field"]
    assert "disabled" in elements["auth-email"]
    assert "required" not in elements["auth-email"]


def test_guest_main_interface_has_login_guidance_and_collapsible_navigation():
    elements = {attrs["id"]: attrs for _, attrs in homepage().elements if "id" in attrs}
    assert "guest-history-hint" in elements
    assert elements["sidebar-toggle"]["aria-controls"] == "main-sidebar"
    assert elements["sidebar-toggle"]["aria-label"]
    assert "hidden" in elements["account-menu"]
    assert "account-menu-logout" in elements


def test_home_submission_shows_question_workspace_before_queueing():
    script = Path("frontend/src/app.js").read_text(encoding="utf-8")
    handler = script.split('homeForm?.addEventListener("submit",', 1)[1].split(
        'homeInput?.addEventListener', 1
    )[0].strip().removesuffix(");")
    check = """
    const assert = require('node:assert/strict');
    const homeInput = {value: '  借款如何处理？  '};
    const homeSubmit = {disabled: false};
    const calls = [];
    const navigate = route => calls.push(['navigate', route]);
    // Simulate the existing loading branch, which only queues a question.
    const submitQuestion = question => calls.push(['queue', question]);
    const event = {preventDefault() {}};
    const handler = HANDLER;
    handler(event);
    assert.deepEqual(calls, [['navigate', 'ask'], ['queue', '借款如何处理？']]);
    assert.equal(homeInput.value, '');
    assert.equal(homeSubmit.disabled, true);
    calls.length = 0;
    homeInput.value = '  ';
    handler(event);
    assert.deepEqual(calls, []);
    """.replace("HANDLER", handler)
    result = subprocess.run(["node", "-e", check], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr


def test_frontend_assets_are_served_and_original_form_contract_is_preserved():
    page = homepage()
    ids = [attrs["id"] for _, attrs in page.elements if "id" in attrs]
    assert len(ids) == len(set(ids)), "Duplicate IDs break existing form and route handlers"
    assert {
        "auth-form", "auth-username", "auth-password", "auth-submit",
        "question-form", "question-input", "send-button", "file-input",
        "material-drawer", "settings-backdrop", "setting-theme",
    }.issubset(ids)
    client = TestClient(app)
    for tag, attrs in page.elements:
        asset = attrs.get("src") if tag == "script" else attrs.get("href") if tag == "link" else None
        if asset and not urlsplit(asset).scheme:
            assert client.get("/" + asset.lstrip("/")).status_code == 200


def test_settings_dialog_has_separate_preferences_and_privacy_panels():
    elements = {attrs["id"]: attrs for _, attrs in homepage().elements if "id" in attrs}
    assert elements.get("settings-dialog", {}).get("role") == "dialog"
    for name in ("preferences", "account"):
        tab = elements.get(f"settings-tab-{name}", {})
        panel = elements.get(f"settings-panel-{name}", {})
        assert tab.get("role") == "tab"
        assert tab.get("aria-controls") == f"settings-panel-{name}"
        assert panel.get("role") == "tabpanel"
        assert panel.get("aria-labelledby") == f"settings-tab-{name}"
    assert "hidden" in elements["settings-panel-account"]
    assert elements["settings-status"].get("role") == "status"


def test_service_address_is_inside_initially_collapsed_advanced_settings():
    html = TestClient(app).get("/").text
    # Parse ancestry rather than matching CSS or source text.
    class SettingAncestry(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.ancestors = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if attrs.get("id") == "setting-backend-url":
                self.ancestors = list(self.stack)
            if tag not in {"input", "img", "link", "meta", "br", "hr"}:
                self.stack.append((tag, attrs))

        def handle_endtag(self, tag):
            if self.stack and self.stack[-1][0] == tag:
                self.stack.pop()

    parser = SettingAncestry()
    parser.feed(html)
    advanced = [attrs for tag, attrs in parser.ancestors if tag == "details"]
    assert advanced, "Service configuration must not crowd ordinary user preferences"
    assert "open" not in advanced[-1]
