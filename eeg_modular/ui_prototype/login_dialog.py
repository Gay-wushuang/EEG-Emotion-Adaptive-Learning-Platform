"""Startup identity and role selection dialog."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from services.identity_store import (
    IdentityStore,
    ROLE_LABELS,
    ROLE_RESEARCH,
    ROLE_STUDENT,
    ROLE_TEACHER,
)


ROLE_DESCRIPTIONS = {
    ROLE_STUDENT: "学习任务、当前建议与难度调整",
    ROLE_TEACHER: "班级状态趋势、异常提醒与过程记录",
    ROLE_RESEARCH: "设备、模型、数据质量与实验记录",
}


class LoginDialog(QDialog):
    """Two-step dialog: local identity first, product role second."""

    def __init__(
        self,
        store: Optional[IdentityStore] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.store = store or IdentityStore()
        self.selected_user_id = ""
        self.selected_user_name = ""
        self.selected_role = ROLE_STUDENT
        self._pending_role = ROLE_STUDENT

        self.setWindowTitle("登录 · 智学脑机助手")
        self.setModal(True)
        self.setMinimumSize(640, 500)
        self.resize(700, 540)
        self.setObjectName("LoginDialog")
        self.setStyleSheet("QDialog#LoginDialog { background-color: #111722; }")

        root = QVBoxLayout(self)
        root.setContentsMargins(42, 34, 42, 34)
        root.setSpacing(18)

        brand = QLabel("智学脑机助手")
        brand.setObjectName("PageTitle")
        brand.setAlignment(Qt.AlignCenter)
        root.addWidget(brand)

        subtitle = QLabel("EEG 学习状态辅助平台")
        subtitle.setObjectName("PageDescription")
        subtitle.setAlignment(Qt.AlignCenter)
        root.addWidget(subtitle)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_identity_page())
        self._stack.addWidget(self._build_role_page())
        root.addWidget(self._stack, 1)

        self._refresh_profiles()

    def _build_identity_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 12, 20, 10)
        layout.setSpacing(14)

        heading = QLabel("登录本地身份")
        heading.setStyleSheet("font-size: 20px; font-weight: 600; color: #F4F7FB;")
        layout.addWidget(heading)

        help_text = QLabel("选择已有 ID，或创建一个仅保存在本机的新 ID。")
        help_text.setObjectName("PageDescription")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        card = QFrame()
        card.setObjectName("Card")
        form = QFormLayout(card)
        form.setContentsMargins(24, 22, 24, 22)
        form.setSpacing(15)

        self._profile_combo = QComboBox()
        self._profile_combo.currentIndexChanged.connect(self._profile_changed)
        form.addRow("本地身份", self._profile_combo)

        self._input_user_id = QLineEdit()
        self._input_user_id.setPlaceholderText("例如：student_001")
        self._input_user_id.setMaxLength(32)
        form.addRow("用户 ID", self._input_user_id)

        self._input_user_name = QLineEdit()
        self._input_user_name.setPlaceholderText("请输入姓名或显示名称")
        self._input_user_name.setMaxLength(40)
        form.addRow("姓名", self._input_user_name)

        layout.addWidget(card)

        self._identity_error = QLabel("")
        self._identity_error.setStyleSheet("color: #F87171; font-size: 12px;")
        self._identity_error.setWordWrap(True)
        layout.addWidget(self._identity_error)
        layout.addStretch()

        actions = QHBoxLayout()
        self._delete_button = QPushButton("删除当前 ID")
        self._delete_button.setObjectName("DangerButton")
        self._delete_button.clicked.connect(self._delete_current_profile)
        actions.addWidget(self._delete_button)
        actions.addStretch()

        cancel = QPushButton("退出")
        cancel.clicked.connect(self.reject)
        actions.addWidget(cancel)

        next_button = QPushButton("下一步：选择使用端")
        next_button.setObjectName("PrimaryButton")
        next_button.clicked.connect(self._continue_to_roles)
        actions.addWidget(next_button)
        layout.addLayout(actions)
        return page

    def _build_role_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 12, 20, 10)
        layout.setSpacing(12)

        heading = QLabel("选择本次使用端")
        heading.setStyleSheet("font-size: 20px; font-weight: 600; color: #F4F7FB;")
        layout.addWidget(heading)

        self._role_identity = QLabel("")
        self._role_identity.setObjectName("PageDescription")
        layout.addWidget(self._role_identity)

        self._role_group = QButtonGroup(self)
        self._role_group.setExclusive(True)
        self._role_buttons = {}
        for role in (ROLE_STUDENT, ROLE_TEACHER, ROLE_RESEARCH):
            button = QPushButton(
                f"{ROLE_LABELS[role]}\n{ROLE_DESCRIPTIONS[role]}"
            )
            button.setCheckable(True)
            button.setMinimumHeight(64)
            button.setStyleSheet(
                "QPushButton { text-align: left; padding: 10px 18px; }"
                "QPushButton:checked { background-color: #1B2940; "
                "border: 2px solid #3B82F6; color: #FFFFFF; }"
            )
            button.clicked.connect(lambda checked, r=role: self._select_role(r))
            self._role_group.addButton(button)
            self._role_buttons[role] = button
            layout.addWidget(button)

        layout.addStretch()
        actions = QHBoxLayout()
        back = QPushButton("返回")
        back.clicked.connect(lambda: self._stack.setCurrentIndex(0))
        actions.addWidget(back)
        actions.addStretch()

        enter = QPushButton("进入系统")
        enter.setObjectName("PrimaryButton")
        enter.clicked.connect(self._finish_login)
        actions.addWidget(enter)
        layout.addLayout(actions)
        return page

    def _refresh_profiles(self, select_user_id: Optional[str] = None) -> None:
        profiles = self.store.list_profiles()
        preferred = select_user_id or self.store.last_user_id()

        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        self._profile_combo.addItem("＋ 创建新本地 ID", "")
        selected_index = 0
        for profile in profiles:
            self._profile_combo.addItem(
                f"{profile['name']}  ·  {profile['user_id']}",
                profile["user_id"],
            )
            if profile["user_id"] == preferred:
                selected_index = self._profile_combo.count() - 1
        self._profile_combo.setCurrentIndex(selected_index)
        self._profile_combo.blockSignals(False)
        self._profile_changed(selected_index)

    def _profile_changed(self, _index: int) -> None:
        user_id = str(self._profile_combo.currentData() or "")
        profile = self.store.get_profile(user_id) if user_id else None
        self._identity_error.clear()
        if profile:
            self._input_user_id.setText(profile["user_id"])
            self._input_user_name.setText(profile["name"])
            self._pending_role = profile["last_role"]
            self._delete_button.setEnabled(True)
        else:
            self._input_user_id.clear()
            self._input_user_name.clear()
            self._pending_role = ROLE_STUDENT
            self._delete_button.setEnabled(False)

    def _delete_current_profile(self) -> None:
        user_id = str(self._profile_combo.currentData() or "")
        if not user_id:
            return
        answer = QMessageBox.question(
            self,
            "删除本地 ID",
            f"确定删除本地 ID“{user_id}”吗？\n不会删除已有 EEG 会话数据。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.store.delete_profile(user_id)
            self._refresh_profiles()

    def _continue_to_roles(self) -> None:
        try:
            user_id, name = self.store.validate(
                self._input_user_id.text(),
                self._input_user_name.text(),
            )
        except ValueError as exc:
            self._identity_error.setText(str(exc))
            return

        self.selected_user_id = user_id
        self.selected_user_name = name
        self._role_identity.setText(f"当前身份：{name}（{user_id}）")
        if self._pending_role not in self._role_buttons:
            self._pending_role = ROLE_STUDENT
        self._role_buttons[self._pending_role].setChecked(True)
        self.selected_role = self._pending_role
        self._stack.setCurrentIndex(1)

    def _select_role(self, role: str) -> None:
        self.selected_role = role

    def _finish_login(self) -> None:
        if not self.selected_user_id or not self.selected_user_name:
            self._stack.setCurrentIndex(0)
            return
        self.store.save_profile(
            self.selected_user_id,
            self.selected_user_name,
            self.selected_role,
        )
        self.accept()
