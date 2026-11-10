# -*- coding: utf-8 -*-
"""Generate the PACER multi-storey handoff report as a Word document."""

import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "PACER-多层建筑-A星路线-交接总结.docx")

CN = "Microsoft YaHei"
ACCENT = RGBColor(0x1F, 0x4E, 0x79)


def set_font(run, size=10.5, bold=False, color=None, mono=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    name = "Consolas" if mono else CN
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if color is not None:
        run.font.color.rgb = color


def para(doc, text="", size=10.5, bold=False, color=None, align=None,
         space_after=4, mono=False):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    if text:
        set_font(p.add_run(text), size, bold, color, mono)
    return p


def h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    set_font(p.add_run(text), 15, True, ACCENT)
    return p


def h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    set_font(p.add_run(text), 12, True, ACCENT)
    return p


def bullet(doc, text, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    set_font(p.add_run(text), size)
    return p


def code(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.space_after = Pt(6)
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        r = p.add_run(ln)
        set_font(r, 9, False, RGBColor(0x33, 0x33, 0x33), mono=True)
        if i < len(lines) - 1:
            r.add_break()
    return p


def table(doc, header, rows, widths=None, size=9.5):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, htxt in enumerate(header):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(p.add_run(htxt), size, True)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            set_font(p.add_run(str(val)), size)
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(doc, path, caption, width=15.5):
    if not os.path.exists(path):
        return
    doc.add_picture(path, width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run(caption), 9, False, RGBColor(0x55, 0x55, 0x55))


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(2.0)

    style = doc.styles["Normal"]
    style.font.name = CN
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), CN)

    # ---------------------------------------------------------------- title
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("PACER 多层建筑 + A* 路线"), 22, True, ACCENT)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("阶段性工作交接总结"), 16, True, ACCENT)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("分支 feat/multi-surface-heightquery"), 9.5, False,
             RGBColor(0x66, 0x66, 0x66))
    doc.add_paragraph()

    # ---------------------------------------------------------------- 0
    h1(doc, "零、一句话结论")
    para(doc,
         "多层高度图的感知层已经做对了（逐格查询、无块号、无状态机），"
         "下楼梯在层高修正后可以直立走完整层。"
         "目前唯一遗留的卡点是【拐角】：角色走到 90° 拐角时速度从 2.4 m/s 掉到 0.01 m/s，"
         "之后无法重新起步。"
         "最新证据指向的根本原因是【行走速度过快】，而不是转弯能力缺失。")

    # ---------------------------------------------------------------- 1
    h1(doc, "一、目标")
    para(doc, "让 PACER 人形（AMP 策略）在三层建筑中，从三楼起点出发，"
              "用 A* 规划路线，经楼梯下行到一楼指定地点停下。")

    # ---------------------------------------------------------------- 2
    h1(doc, "二、当前架构（已定型）")
    h2(doc, "2.1 高度查询：多层曲面栈 + 逐格查询")
    code(doc,
         "h(x, y) = max{ 曲面_s(x, y) : 曲面_s(x, y) <= foot_z + 0.35 }\n"
         "无曲面符合 -> PIT(-1.5)")
    bullet(doc, "没有块号 / 状态机 / 落点判据 / 重叠带 / 子路径切换（这些全部删除过，证明不是问题所在）")
    bullet(doc, "_foot_ref = 每步全身最低点 z，在 _compute_observations 中刷新")
    bullet(doc, "重生时在 _reset_ref_state_init 开头把 _foot_ref 置 None，回退到 _spawn_ref_z = start_z")

    h2(doc, "2.2 关键文件")
    table(doc,
          ["文件", "作用"],
          [["pacer/env/tasks/humanoid_pedestrain_terrain.py", "主体：MeshTerrain / sample_height_points / get_center_heights / get_heights / _compute_observations"],
           ["script/patch_multi_surface_query.py", "17 处编辑的补丁，apply / revert / status，备份 .multisurf_bak"],
           ["script/make_multi_floor_switchback.py", "三层场景生成器（NSTEP / LAND 控制连跑或折返），A* + 网格 + 验证"],
           ["script/make_2f_scene.py", "两层 / 单层场景（--single 只有二层楼板）"],
           ["script/make_2f_leg.py", "只走二层的路径生成"],
           ["script/eval_stage_mesh.sh", "评测脚本（已支持 START_YAW）"]],
          widths=[6.5, 9.5])

    # ---------------------------------------------------------------- 3
    h1(doc, "三、已修复的真 Bug")
    para(doc, "以下每一条都实际导致了错误结论或错误行为。", 9.5, False,
         RGBColor(0x55, 0x55, 0x55))
    table(doc,
          ["#", "Bug", "后果", "修法"],
          [["1", "heights1 未定义（UnboundLocalError）",
            "每次运行都崩在初始化，之前十几次实验分析的是旧 dump",
            "补丁锚点从 2 行扩到 3 行"],
           ["2", "重生沿用上一回合的脚参考",
            "在 3F 位置重生到 2F 高度，随后掉出世界",
            "_reset_ref_state_init 里清 _foot_ref"],
           ["3", "start_yaw 从未传入，且是相对旋转",
            "角色一直朝东出生；传 90° 实际得到 −179°",
            "评测脚本加 START_YAW；改为绝对朝向"],
           ["4", "层高 1.8 m < 身高 1.75 m",
            "净空仅 5 cm，角色蹲着 / 趴着下楼梯",
            "层高改为 2.4 m"],
           ["5", "楼梯井洞口在出口处被上层楼板盖住",
            "视野里出现低天花板（obs=−5.9 即高 1.2 m 的面），角色下蹲崩掉",
            "洞口在下行侧外扩 2 m"],
           ["6", "楼梯 B 洞口两头都外扩",
            "B 入口楼板被挖空，走不过去",
            "只在下行侧外扩"]],
          widths=[0.8, 4.4, 5.6, 3.6])

    para(doc, "注意：#3 的 start_yaw 绝对化改动与评测脚本的 START_YAW 支持是直接改在服务器文件上的，"
              "尚未写入 patch_multi_surface_query.py。若 revert 补丁会丢失，需补进补丁脚本。",
         9.5, False, RGBColor(0xB0, 0x30, 0x30))

    # ---------------------------------------------------------------- 4
    h1(doc, "四、当前场景参数（mfsb2）")
    code(doc,
         "层高 H = 12 级 x riser 0.20 = 2.40 m    (头顶余量 0.65 m)\n"
         "tread 0.36，连续 12 级（无中间平台）\n"
         "3F = 4.80   2F = 2.40   1F = 0.00\n\n"
         "U 型走廊  WEST x[0,3.6] y[0,14]   NORTH x[0,11] y[9.6,14]   EAST x[7.4,11] y[0,14]\n"
         "楼梯 A    x[0.3,3.3]  y 2.00->6.32 (+y)   z 4.80->2.40\n"
         "楼梯 B    x[7.7,10.7] y 8.00->3.68 (-y)   z 2.40->0.00\n"
         "洞口      3F 的 A 洞 y 2.0->8.32 ；2F 的 B 洞 y 1.68->8.0\n\n"
         "曲面      floor_3F, A, floor_2F, B, floor_1F  (5 张)\n"
         "A* 安全距 1.2 m，起点 (1.80,1.20)，终点 (9.20,1.00)，路线 22.22 m / 10 点")

    # ---------------------------------------------------------------- 5
    h1(doc, "五、已排除的假设（不必重复排查）")
    table(doc,
          ["假设", "结论"],
          [["判块状态机", "已删除，问题依旧"],
           ["重叠区 / 交界点", "已改，问题依旧"],
           ["A* 路径形状", "换纯直线，问题依旧"],
           ["路点指反", "数据：路点 0 恒定在前方 1.00 m，10 点跨度 1.69–1.72 m，随身体同步推进"],
           ["高度查询错", "打印结果：正确的楼梯剖面 3.60 → 3.30 → 3.15 → 3.00"],
           ["riser 太陡 / 级数太多", "层高修好后，12 级 x 0.20 能直立走完（最低躯干 0.72）"],
           ["感知没切到二层", "打印 foot_ref = 2.40、center = 2.40，正确"]],
          widths=[4.6, 10.9])

    # ---------------------------------------------------------------- 6
    h1(doc, "六、已确认的因果关系")
    h2(doc, "6.1 头顶有楼板会破坏二层行为")
    table(doc,
          ["场景", "头顶楼板", "二层直走", "拐角"],
          [["三层完整", "有", "往南偏、趴下", "失败"],
           ["两层（去 3F）", "无", "向北 2.8 m，直立", "停在拐角"],
           ["纯二层楼板", "无", "向北 3.8 m，直立", "不会转"],
           ["纯二层 + 圆弧", "无", "正常", "成功走完全程"]],
          widths=[3.6, 3.0, 5.2, 3.0])

    h2(doc, "6.2 速度统计（stage5，尖锐直角，700 帧）")
    table(doc,
          ["指标", "数值"],
          [["dt", "1/30 s（30 Hz），总时长 23.3 s"],
           ["平均速度", "0.455 m/s"],
           ["中位速度", "0.004 m/s"],
           ["最大速度", "3.127 m/s"],
           ["低于 0.10 m/s 的帧", "437 / 700 = 62.4%"],
           ["直行段平均速度", "2.10 m/s"],
           ["拐角处最低速度", "0.01 m/s"],
           ["最大角速度", "4.47 rad/s ≈ 244 deg/s"]],
          widths=[5.4, 9.4])
    figure(doc, os.path.join(ROOT, "experiment_result", "multi_floor", "s5_velocity.png"),
           "图 1  stage5 在二层路线上的线速度与角速度（灰色为两个 90° 拐角）")

    h2(doc, "6.3 拐角圆弧半径的影响（纯二层楼板，stage5）")
    table(doc,
          ["转弯方式", "距目标", "结果"],
          [["尖锐直角", "2.06 m", "卡在第二拐角"],
           ["1.0 m 圆弧（12 点）", "1.91 m", "卡在第二拐角"],
           ["1.5 m 圆弧（7 点）", "0.64 m（stage6 0.30 m）", "通过，走完全程"],
           ["2.0 m 圆弧", "4.36 m", "全程蠕行 0.18 m/s"],
           ["2.5 m 圆弧", "4.39 m", "全程蠕行 0.18 m/s"]],
          widths=[5.0, 5.4, 4.6])
    para(doc, "存在最优区间（约 1.5 m），并非越大越好。", 9.5, False,
         RGBColor(0x55, 0x55, 0x55))
    figure(doc, os.path.join(ROOT, "experiment_result", "multi_floor", "flat2_arc_sheet.png"),
           "图 2  1.5 m 半径圆弧：从楼梯 A 根部走到楼梯 B 口，两个 90° 转弯全部通过")

    # ---------------------------------------------------------------- 7
    h1(doc, "七、核心假设与下一步")
    h2(doc, "7.1 假设：根本矛盾是「走太快」，不是「不会转弯」")
    table(doc,
          ["场景", "路径长度", "实测平均速度", "90° 转弯"],
          [["s3route（原版能过）", "19.13 m", "0.50 m/s", "成功"],
           ["纯二层平板", "13.30 m", "2.7 m/s", "卡住"]],
          widths=[5.4, 3.4, 3.6, 2.8])
    para(doc, "转向率需求 ω = v / R：速度高 5 倍，每帧所需转角就高 5 倍。"
              "转弯并不是单独训练出来的技能，它是「跟随路点」涌现出来的行为——"
              "只要每帧所需转角足够小，就能走。")
    figure(doc, os.path.join(ROOT, "experiment_result", "multi_floor", "route_clearance.mp4".replace(".mp4", "_sheet.png"))
           if os.path.exists(os.path.join(ROOT, "experiment_result", "multi_floor", "route_clearance_sheet.png")) else "",
           "图 3  完整路线（A* 安全距 1.2 m）：三层下到二层，全程直立，最低躯干 0.81")

    h2(doc, "7.2 待验证实验")
    bullet(doc, "把隐含速度降到 0.5–1.0 m/s（拉长 traj_duration 或缩小 traj_sample_timestep），"
                "用尖锐直角路径重跑——若假设成立，直角也应该能过")
    bullet(doc, "通过后用 stage5 权重（转弯能力明显优于 stage6）跑完整三层路线")
    bullet(doc, "完整路线中 2F → 1F 段（楼梯 B）从未真正跑过，目前都停在 2F 楼梯口")

    # ---------------------------------------------------------------- 8
    h1(doc, "八、运行环境与命令")
    code(doc,
         "服务器 hoi = 211.81.55.164:41190   用户 ljx   GPU 7\n"
         "仓库 /data/ljx/code/pacer-main      环境 /data/ljx/envs/pacer\n"
         "source /data/ljx/scripts/pacer_env.sh\n\n"
         "权重：\n"
         "  stage5 = output/exp/pacer_stage5_1000ep   (转弯好)\n"
         "  stage6 = output/exp/pacer_stage6_500ep\n\n"
         "配置（只在服务器上，不在仓库）：\n"
         "  pacer/data/cfg/pacer_stage6_free.yaml   enableEarlyTermination: false, episodeLength 3000\n"
         "  pacer/data/cfg/pacer_stage6_hold.yaml   stair_goal_termination: false\n\n"
         "命令模板：\n"
         "WEIGHTS=... MESH=mfsb2 PATH_FILE=data/paths/mfsb2.json TAG=xxx \\\n"
         "CFG_ENV=pacer/data/cfg/pacer_stage6_free.yaml GPU=7 FRAMES=900 \\\n"
         "START_XY=1.80,1.20 START_YAW=1.5707963 bash script/eval_stage_mesh.sh")
    para(doc, "参考值：_fail_dist = 4.0，rise_tol = 0.35，view_min = 1.0，"
              "观测 1412 维（自身 368 + 20 路点 + 1024 高度图），use_center_height: true。")

    # ---------------------------------------------------------------- 9
    h1(doc, "九、产物与 Git 状态")
    para(doc, "本地 experiment_result/multi_floor/ 下保存了全部视频与截图：")
    bullet(doc, "route_clearance.mp4 —— 完整路线效果最好的一次（三层下到二层，最低躯干 0.81）")
    bullet(doc, "flat2_arc_SUCCESS.mp4 —— 圆弧方案成功走完二层同层路线")
    bullet(doc, "flat2_radius15 / 20 / 25.mp4 —— 三种半径对比")
    bullet(doc, "s5_velocity.png —— 线速度 / 角速度统计图")
    para(doc, "Git：分支 feat/multi-surface-heightquery，工作区有未提交的生成器与补丁改动，"
              "新对话开始后建议先 git status 确认再继续。")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc.save(OUT)
    print("saved:", OUT)
    print("size: %.1f KB" % (os.path.getsize(OUT) / 1024.0))


if __name__ == "__main__":
    main()
