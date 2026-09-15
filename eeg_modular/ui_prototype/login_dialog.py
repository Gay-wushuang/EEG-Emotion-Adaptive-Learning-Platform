"""Startup account selection and local identity creation dialogs."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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
    role_for_user_id,
)


ROLE_DESCRIPTIONS = {
    ROLE_STUDENT: "学习任务、当前建议与难度调整",
    ROLE_TEACHER: "班级状态趋势、异常提醒与过程记录",
    ROLE_RESEARCH: "设备、模型、数据质量与实验记录",
}

IDENTITY_ROLE_LABELS = {
    ROLE_STUDENT: "学生",
    ROLE_TEACHER: "教师",
    ROLE_RESEARCH: "管理员",
}


class LoginDialog(QDialog):
    """Local account picker; role is derived from the immutable ID prefix."""

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
        self.setMinimumSize(680, 520)
        self.resize(760, 580)
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
        root.addWidget(self._stack, 1)

        self._refresh_profiles()

    def _build_identity_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 12, 20, 10)
        layout.setSpacing(14)

        heading = QLabel("选择登录账号")
        heading.setStyleSheet("font-size: 20px; font-weight: 600; color: #F4F7FB;")
        layout.addWidget(heading)

        help_text = QLabel(
            "选择一个已有账号进入系统。账号角色在创建时确定，登录后不可修改。"
        )
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
        form.addRow("已有账号", self._profile_combo)

        self._input_user_id = QLineEdit()
        self._input_user_id.setReadOnly(True)
        self._input_user_id.setPlaceholderText("选择账号后自动显示")
        form.addRow("用户 ID", self._input_user_id)

        self._input_user_name = QLineEdit()
        self._input_user_name.setReadOnly(True)
        self._input_user_name.setPlaceholderText("选择账号后自动显示")
        form.addRow("姓名", self._input_user_name)

        self._role_display = QLineEdit()
        self._role_display.setReadOnly(True)
        self._role_display.setPlaceholderText("选择账号后自动显示")
        form.addRow("账号角色", self._role_display)

        layout.addWidget(card)

        self._identity_error = QLabel("")
        self._identity_error.setStyleSheet("color: #F87171; font-size: 12px;")
        self._identity_error.setWordWrap(True)
        layout.addWidget(self._identity_error)
        layout.addStretch()

        actions = QHBoxLayout()
        create_button = QPushButton("创建新用户")
        create_button.clicked.connect(self._open_create_dialog)
        actions.addWidget(create_button)

        self._delete_button = QPushButton("删除当前账号")
        self._delete_button.setObjectName("DangerButton")
        self._delete_button.clicked.connect(self._delete_current_profile)
        actions.addWidget(self._delete_button)
        actions.addStretch()

        cancel = QPushButton("退出")
        cancel.clicked.connect(self.reject)
        actions.addWidget(cancel)

        self._login_button = QPushButton("进入系统")
        self._login_button.setObjectName("PrimaryButton")
        self._login_button.clicked.connect(self._finish_login)
        actions.addWidget(self._login_button)
        layout.addLayout(actions)
        return page

    def _build_create_dialog(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle("创建新用户")
        dialog.setModal(True)
        dialog.setMinimumSize(520, 360)
        dialog.setObjectName("CreateIdentityDialog")
        dialog.setStyleSheet(
            "QDialog#CreateIdentityDialog{background:#111722;}"
            "QFrame#CreateIdentityCard{background:#1B2433;border:1px solid #2D394C;"
            "border-radius:12px;}"
        )
        root = QVBoxLayout(dialog)
        root.setContentsMargins(32, 28, 32, 26)
        root.setSpacing(16)
        title = QLabel("创建新用户")
        title.setStyleSheet("font-size:22px;font-weight:700;color:#F4F7FB;")
        root.addWidget(title)
        subtitle = QLabel("角色仅在创建时选择；系统会自动生成固定角色前缀的用户 ID。")
        subtitle.setObjectName("PageDescription")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        card = QFrame()
        card.setObjectName("CreateIdentityCard")
        form = QFormLayout(card)
        form.setContentsMargins(22, 20, 22, 20)
        form.setSpacing(15)
        self._create_role_combo = QComboBox()
        self._create_role_combo.addItem("学生", ROLE_STUDENT)
        self._create_role_combo.addItem("教师", ROLE_TEACHER)
        self._create_role_combo.addItem("管理员", ROLE_RESEARCH)
        form.addRow("用户角色", self._create_role_combo)
        self._create_name_input = QLineEdit()
        self._create_name_input.setMaxLength(40)
        self._create_name_input.setPlaceholderText("请输入姓名")
        form.addRow("姓名", self._create_name_input)
        generated = QLabel("用户 ID 将自动生成：st_xxx / teacher_xxx / admin_xxx")
        generated.setObjectName("PageDescription")
        generated.setWordWrap(True)
        form.addRow("", generated)
        root.addWidget(card)

        self._create_error = QLabel("")
        self._create_error.setStyleSheet("color:#F87171;font-size:12px;")
        self._create_error.setWordWrap(True)
        root.addWidget(self._create_error)
        root.addStretch()
        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("取消")
        cancel.clicked.connect(dialog.reject)
        buttons.addWidget(cancel)
        create = QPushButton("创建并选择")
        create.setObjectName("PrimaryButton")
        create.clicked.connect(lambda: self._create_new_user(dialog))
        buttons.addWidget(create)
        root.addLayout(buttons)
        return dialog

    def _open_create_dialog(self) -> None:
        self._create_dialog = self._build_create_dialog()
        self._create_dialog.exec()

    def _create_new_user(self, dialog: QDialog) -> None:
        try:
            profile = self.store.create_profile(
                str(self._create_role_combo.currentData() or ""),
                self._create_name_input.text(),
            )
        except ValueError as exc:
            self._create_error.setText(str(exc))
            return
        self._refresh_profiles(profile["user_id"])
        dialog.accept()

    def _refresh_profiles(self, select_user_id: Optional[str] = None) -> None:
        profiles = self.store.list_profiles()
        preferred = select_user_id or self.store.last_user_id()

        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        selected_index = -1
        for profile in profiles:
            self._profile_combo.addItem(
                f"{profile['name']}  ·  {profile['user_id']}",
                profile["user_id"],
            )
            if profile["user_id"] == preferred:
                selected_index = self._profile_combo.count() - 1
        if not profiles:
            self._profile_combo.addItem("暂无已有账号，请先创建用户", "")
            selected_index = 0
        elif selected_index < 0:
            selected_index = 0
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
            self._role_display.setText(IDENTITY_ROLE_LABELS[profile["last_role"]])
            self._delete_button.setEnabled(True)
            self._login_button.setEnabled(True)
        else:
            self._input_user_id.clear()
            self._input_user_name.clear()
            self._role_display.clear()
            self._pending_role = ROLE_STUDENT
            self._delete_button.setEnabled(False)
            self._login_button.setEnabled(False)

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
        # Compatibility hook for older callers; there is no role-selection step.
        self._finish_login()

    def _finish_login(self) -> None:
        selected_id = str(self._profile_combo.currentData() or "")
        try:
            user_id, name = self.store.validate(
                self._input_user_id.text(), self._input_user_name.text()
            )
            role = role_for_user_id(user_id)
        except ValueError as exc:
            self._identity_error.setText(str(exc))
            return
        if selected_id and user_id != selected_id:
            self._identity_error.setText("登录账号信息已变化，请重新选择账号。")
            return
        self.selected_user_id = user_id
        self.selected_user_name = name
        self.selected_role = role
        self.store.save_profile(user_id, name, role)
        self.accept()
