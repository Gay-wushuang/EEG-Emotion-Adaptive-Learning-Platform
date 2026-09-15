"""历史与报告页 1920×1080 真实验收截图（Windows 原生 Qt 主窗口 + 真实持久化数据）。

目标物理分辨率 1920×1080；按屏幕 devicePixelRatio 换算逻辑尺寸后 resize 主窗口，
使客户区（内容区）物理像素精确等于 1920×1080。

场景 A：进入历史页，未选择任何记录（产品化空态）
场景 B：选择一个含两个及以上任务的真实会话（多任务逐行展示）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ui_prototype"))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer

TARGET_PHYS_W, TARGET_PHYS_H = 1920, 1080
# 用于场景 B 的真实多任务会话（data/sessions 中 st_001 的 live 记录）
PREFERRED_SESSIONS = ["3021b05a4a17", "ab8308b05b53", "b7544094b8d4"]


def dump_geometry(window, tag):
    page = window._pages["history"]
    scr = window.screen()
    g, fg = window.geometry(), window.frameGeometry()
    cw = window.centralWidget()
    pg = page.geometry()
    dpr = scr.devicePixelRatio()
    print(f"===== geometry [{tag}] =====")
    print(f"screen: name={scr.name()} logical_geometry={scr.geometry().width()}x{scr.geometry().height()}"
          f" available={scr.availableGeometry().width()}x{scr.availableGeometry().height()}"
          f" dpr={dpr}")
    print(f"window.geometry (client, logical): {g.width()}x{g.height()} @({g.x()},{g.y()})"
          f" -> physical {round(g.width()*dpr)}x{round(g.height()*dpr)}")
    print(f"window.frameGeometry (含标题栏边框, logical): {fg.width()}x{fg.height()} @({fg.x()},{fg.y()})"
          f" -> physical {round(fg.width()*dpr)}x{round(fg.height()*dpr)}")
    print(f"frame 纵向额外占用 (logical): {fg.height() - g.height()} px")
    print(f"centralWidget.geometry (logical): {cw.width()}x{cw.height()}")
    print(f"history page geometry (logical): {pg.width()}x{pg.height()}")
    msh = page.minimumSizeHint()
    print(f"history page minimumSizeHint (logical): {msh.width()}x{msh.height()}"
          f" | fits_width={msh.width() <= pg.width()} fits_height={msh.height() <= pg.height()}")
    # 横向滚动检查：表格列宽总和 vs viewport 宽度
    tbl = page._table
    col_total = sum(tbl.columnWidth(i) for i in range(tbl.columnCount()))
    vp_w = tbl.viewport().width()
    hsb = tbl.horizontalScrollBar()
    print(f"table columns={tbl.columnCount()} col_total={col_total} viewport={vp_w}"
          f" hscrollbar_max={hsb.maximum()} (0 = 无需横向滚动)")
    return dpr


def main():
    # 与 main.py 启动流程 1:1 一致：高 DPI 策略 → QApplication → 中文字体 → theme.qss
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("智学脑机助手")

    from main import load_stylesheet
    from services.font_loader import ensure_chinese_font
    ensure_chinese_font()
    load_stylesheet(app)

    out_dir = ROOT / "screenshots"
    out_dir.mkdir(exist_ok=True)

    from main_window import MainWindow
    window = MainWindow(mode="mock", user_id="st_001", user_name="学生一", role="student")

    # 按物理 1920×1080 换算逻辑尺寸（DPR 1.25 屏 → 1536×864 逻辑）
    scr = window.screen() or app.primaryScreen()
    dpr = scr.devicePixelRatio()
    log_w, log_h = round(TARGET_PHYS_W / dpr), round(TARGET_PHYS_H / dpr)
    window.resize(log_w, log_h)
    window.show()
    window.activateWindow()
    window.raise_()

    state = {
        "dpr": dpr,
        "log_w": log_w,
        "log_h": log_h,
    }

    def grab(name):
        pm = window.grab()  # 客户区渲染，按 DPR 输出物理像素
        path = out_dir / name
        pm.save(str(path), "PNG")
        # pm.width() 为设备无关宽；物理宽 = round(width*dpr)
        print(f"saved {path} | pixmap logical={pm.width()}x{pm.height()}"
              f" dpr={pm.devicePixelRatio()}"
              f" physical={round(pm.width()*pm.devicePixelRatio())}x{round(pm.height()*pm.devicePixelRatio())}")

    def go_fullscreen(tag):
        """切换全屏（无标题栏）→ 客户区 = 整个物理屏幕 = 精确 1920×1080。"""
        window.showFullScreen()
        print(f"===== fullscreen [{tag}] =====")
        g = window.geometry()
        print(f"window.geometry (client, logical): {g.width()}x{g.height()}"
              f" -> physical {round(g.width()*dpr)}x{round(g.height()*dpr)}")

    def find_multitask_row(page):
        """在当前筛选下的表格里定位多任务会话行号。"""
        sessions = page._get_sorted_sessions()
        # 优先选择真实多任务会话
        for sid in PREFERRED_SESSIONS:
            for i, s in enumerate(sessions):
                if s.session_id == sid and len(getattr(s, "tasks", []) or []) >= 2:
                    return i, s
        # 兜底：任意含 ≥2 任务的会话
        for i, s in enumerate(sessions):
            if len(getattr(s, "tasks", []) or []) >= 2:
                return i, s
        return None, None

    def step_a():
        try:
            window._navigate_to("history")
        except Exception:
            window.stack.setCurrentWidget(window._pages["history"])
        page = window._pages["history"]
        # 切到"实时采集"（st_001 的真实 live 记录最多）
        page._combo_source.setCurrentIndex(1)
        page._refresh_table()
        page._clear_detail()  # 场景 A：未选择任何记录
        dump_geometry(window, "A 未选择记录（窗口化）")
        print(f"rows={page._table.rowCount()}")
        grab("history_windowed_A_unselected.png")
        # 全屏（无标题栏）→ 客户区精确 1920×1080 物理
        go_fullscreen("A")
        dump_geometry(window, "A 未选择记录（全屏）")
        grab("history_1920x1080_A_unselected.png")
        QTimer.singleShot(400, step_b)

    def step_b():
        page = window._pages["history"]
        idx, s = find_multitask_row(page)
        if s is None:
            print("!! 未找到多任务会话（真实数据）")
            finish()
            return
        page._table.selectRow(idx)
        dump_geometry(window, "B 选择多任务会话（全屏）")
        print(f"selected session_id={s.session_id} start={s.start_time}")
        print(f"tasks ({len(s.tasks)}):")
        for t in s.tasks:
            print(f"  - name={t.name} difficulty={t.difficulty}"
                  f" duration_seconds={t.duration_seconds} status={t.status}")
        print("tasks_label:")
        print(page._tasks_label.text())
        # 任务阶段明细卡的实际可见行数
        rows = [page._tasks_body_layout.itemAt(i).widget()
                for i in range(page._tasks_body_layout.count())]
        visible_task_rows = [w for w in rows if w is not None]
        print(f"任务阶段明细可见行数: {len(visible_task_rows)} (应为 {len(s.tasks)})")
        grab("history_1920x1080_B_multitask.png")
        QTimer.singleShot(300, finish)

    def finish():
        window.close()
        app.quit()

    QTimer.singleShot(800, step_a)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
