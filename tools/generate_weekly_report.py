#!/usr/bin/env python3
"""Generate a newspaper-style weekly development report for the project.

The report is intentionally data-first: committed Git activity is separated
from the current working tree so that unfinished work is visible without
being mistaken for a released change.
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs" / "weekly_reports"
REPORT_ASSET = "assets/weekly-report-concept.png"
DIAGNOSTIC_ASSET = "../test1_diagnostic_contact_sheet.jpg"
EXCLUDED_STATUS_PREFIXES = ("docs/weekly_reports/",)


@dataclass(frozen=True)
class Commit:
    sha: str
    day: str
    subject: str
    author: str


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip()


def parse_start(value: str | None, now: datetime) -> datetime:
    if value:
        return datetime.strptime(value, "%Y-%m-%d").astimezone()
    return (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def git_window(start: datetime, end: datetime) -> tuple[list[Commit], dict[str, int], set[str]]:
    since = start.isoformat(timespec="seconds")
    until = end.isoformat(timespec="seconds")
    raw = run_git(
        "log",
        f"--since={since}",
        f"--until={until}",
        "--date=short",
        "--pretty=format:%H%x09%ad%x09%s%x09%an",
    )
    commits: list[Commit] = []
    for line in raw.splitlines():
        parts = line.split("\t", 3)
        if len(parts) == 4:
            commits.append(Commit(*parts))

    numstat = run_git(
        "log",
        f"--since={since}",
        f"--until={until}",
        "--format=",
        "--numstat",
    )
    additions = 0
    deletions = 0
    touched: set[str] = set()
    for line in numstat.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        added, deleted, path = parts
        if added.isdigit():
            additions += int(added)
        if deleted.isdigit():
            deletions += int(deleted)
        touched.add(path)
    stats = {"additions": additions, "deletions": deletions, "files": len(touched)}

    status_lines = run_git("status", "--porcelain=v1").splitlines()
    working_tree: set[str] = set()
    for line in status_lines:
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[-1]
        if not path.startswith(EXCLUDED_STATUS_PREFIXES):
            working_tree.add(path)
    return commits, stats, working_tree


def project_area(path: str) -> str:
    if path.startswith(("app/vision/", "app/pipeline/", "app/geometry/", "app/modes/")):
        return "视觉与实时管线"
    if path.startswith(("app/server/", "app/services/", "app/state/")):
        return "服务与状态"
    if path.startswith("web/"):
        return "前端与调试体验"
    if path.startswith(("tests/", "tools/")):
        return "测试与工具链"
    if path.startswith("docs/") or path in {"README.md", "SETUP_GUIDE.md"}:
        return "文档与验证"
    return "其他"


def count_test_functions() -> int:
    pattern = re.compile(r"^\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(", re.MULTILINE)
    return sum(len(pattern.findall(path.read_text(encoding="utf-8", errors="ignore"))) for path in (ROOT / "tests").glob("test_*.py"))


def collect_test_count() -> tuple[int, str]:
    fallback = count_test_functions()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=35,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return fallback, "静态测试清单"
    match = re.search(r"(\d+) tests? collected", result.stdout + result.stderr)
    if match:
        return int(match.group(1)), "pytest collect-only"
    return fallback, "静态测试清单"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def format_number(value: int) -> str:
    return f"{value:,}"


def daily_counts(commits: list[Commit], start: datetime, end: datetime) -> list[tuple[str, int]]:
    counts = Counter(commit.day for commit in commits)
    current = start.date()
    last = end.date()
    days: list[tuple[str, int]] = []
    while current <= last:
        days.append((current.strftime("%m/%d"), counts[current.isoformat()]))
        current += timedelta(days=1)
    return days


def bar_chart(days: list[tuple[str, int]]) -> str:
    max_count = max((count for _, count in days), default=1)
    width = 620
    bar_width = max(24, int(width / max(len(days), 1) * 0.56))
    gap = width / max(len(days), 1)
    items: list[str] = [f'<svg class="activity-chart" viewBox="0 0 {width} 190" role="img" aria-label="每日提交数量">']
    items.append('<line x1="0" y1="142" x2="620" y2="142" stroke="#191919" stroke-width="1"/>')
    for index, (label, count) in enumerate(days):
        x = index * gap + (gap - bar_width) / 2
        height = 0 if not count else max(10, int(105 * count / max_count))
        y = 142 - height
        color = "#2554c7" if index == len(days) - 1 else "#1a1a1a"
        items.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_width}" height="{height}" fill="{color}"/>')
        items.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 8}" text-anchor="middle" class="chart-value">{count}</text>')
        items.append(f'<text x="{x + bar_width / 2:.1f}" y="166" text-anchor="middle" class="chart-label">{esc(label)}</text>')
    items.append("</svg>")
    return "".join(items)


def pitch_diagram() -> str:
    return """<svg class="pitch-diagram" viewBox="0 0 220 310" role="img" aria-label="项目视觉管线示意图">
      <rect x="16" y="16" width="188" height="278" fill="none" stroke="#191919" stroke-width="1.4"/>
      <path d="M16 155h188M110 16v278M84 16v44h52V16M84 294v-44h52v44M110 128a27 27 0 1 0 0 54a27 27 0 1 0 0-54" fill="none" stroke="#191919" stroke-width="1.2"/>
      <path d="M42 82 L88 108 L137 88 L177 120 L143 153 L100 192 L66 225" fill="none" stroke="#2554c7" stroke-width="2"/>
      <path d="M178 213 L145 190 L110 226 L77 207 L43 232" fill="none" stroke="#787878" stroke-width="1.5" stroke-dasharray="4 4"/>
      <g fill="#2554c7" stroke="#fff" stroke-width="1.2">
        <circle cx="42" cy="82" r="6"/><circle cx="88" cy="108" r="6"/><circle cx="137" cy="88" r="6"/>
        <circle cx="177" cy="120" r="6"/><circle cx="143" cy="153" r="6"/><circle cx="100" cy="192" r="6"/><circle cx="66" cy="225" r="6"/>
      </g>
      <g fill="#8b8b8b" stroke="#fff" stroke-width="1.2"><circle cx="178" cy="213" r="5"/><circle cx="145" cy="190" r="5"/><circle cx="110" cy="226" r="5"/><circle cx="77" cy="207" r="5"/><circle cx="43" cy="232" r="5"/></g>
      <text x="16" y="307" class="svg-caption">PLAYER / BALL / FIELD-SPACE</text>
    </svg>"""


def section_copy(area: str, count: int, test_count: int) -> tuple[str, str]:
    if area == "视觉与实时管线":
        return ("视觉与实时管线", f"窗口内共有 {count} 个文件落在检测、跟踪、几何投影或运行模式相关路径。重点围绕帧级推理、角色语义与状态稳定性展开；报告只陈述 Git 可验证的文件范围，不替代模型精度评测。")
    if area == "服务与状态":
        return ("服务与状态", f"服务层与状态层涉及 {count} 个文件。该区域连接 FastAPI、WebSocket 与共享状态，是将视觉结果变成可观测系统的关键边界。")
    if area == "前端与调试体验":
        return ("前端与调试体验", f"前端路径涉及 {count} 个文件。变化集中在 dashboard 类型定义、渲染层或交互调试链路；实际 UI 体验应结合构建与浏览器验证继续观察。")
    if area == "测试与工具链":
        return ("测试与工具链", f"测试与工具链涉及 {count} 个文件。当前测试清单约 {test_count} 个测试函数，适合作为本周回归范围的基线，而不是通过率的替代品。")
    if area == "文档与验证":
        return ("文档与验证", f"文档或验证资产涉及 {count} 个文件。它们帮助把模型、运行模式与诊断证据固定下来，降低下一轮调试的上下文成本。")
    return ("其他变化", f"有 {count} 个文件未落入预设路径分类，建议在下一轮提交时补充更明确的模块边界。")


def write_report(start: datetime, end: datetime, output: Path | None = None) -> Path:
    commits, stats, working_tree = git_window(start, end)
    test_count, test_source = collect_test_count()

    commit_paths: set[str] = set()
    area_counts: Counter[str] = Counter()
    for commit in commits:
        show = run_git("show", "--format=", "--name-only", commit.sha)
        for path in show.splitlines():
            if path and not path.startswith(EXCLUDED_STATUS_PREFIXES):
                commit_paths.add(path)
                area_counts[project_area(path)] += 1
    for path in working_tree:
        area_counts[project_area(path)] += 1

    now_label = end.strftime("%Y.%m.%d")
    week_label = f"{start.strftime('%m.%d')} — {end.strftime('%m.%d, %Y')}"
    issue = end.isocalendar().week
    daily = daily_counts(commits, start, end)
    latest = commits[0].subject if commits else "观察窗口内没有新的提交，报告转向工作树与存量基线。"
    worktree_label = "、".join(sorted(working_tree)[:6]) if working_tree else "工作树干净"
    if len(working_tree) > 6:
        worktree_label += f" 等 {len(working_tree)} 项"

    areas = ["视觉与实时管线", "服务与状态", "前端与调试体验", "测试与工具链", "文档与验证"]
    area_articles: list[str] = []
    for area in areas:
        count = area_counts.get(area, 0)
        title, copy = section_copy(area, count, test_count)
        area_articles.append(f"""<article class="story">
          <div class="story-index">{areas.index(area) + 1} / {esc(title)}</div>
          <h3>{esc(title)}</h3>
          <p>{esc(copy)}</p>
        </article>""")

    if commits:
        commit_rows = "".join(
            f'<li><time>{esc(commit.day)}</time><span>{esc(commit.subject)}</span><small>{esc(commit.sha[:7])}</small></li>'
            for commit in commits[:6]
        )
    else:
        commit_rows = '<li class="empty-row"><span>本窗口暂无新的提交</span></li>'
    if working_tree:
        worktree_rows = "".join(f"<li><span>{esc(path)}</span><small>{esc(project_area(path))}</small></li>" for path in sorted(working_tree)[:10])
    else:
        worktree_rows = '<li class="empty-row"><span>工作树干净</span></li>'

    headline = "本周进展由提交与工作树共同构成" if commits else "本周没有新增提交，现状仍然可追踪"
    output = output or REPORT_DIR / f"report-{end.strftime('%Y-%m-%d')}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    latest_path = REPORT_DIR / "latest.html"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Soccer Analysis — Weekly Development Report {esc(week_label)}</title>
  <style>
    :root {{
      --ink: #141414;
      --muted: #5f5f5f;
      --rule: #1d1d1d;
      --paper: #fff;
      --blue: #2554c7;
      --soft-blue: #eef2fb;
      --serif: "Noto Serif CJK SC", "Songti SC", "STSong", Georgia, serif;
      --sans: Inter, -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #d9d9d9; color: var(--ink); font-family: var(--sans); }}
    .paper {{ width: min(1200px, 100%); margin: 24px auto; padding: 28px 52px 42px; background: var(--paper); box-shadow: 0 10px 38px rgba(0,0,0,.12); }}
    .masthead {{ display: grid; grid-template-columns: 1fr 1fr 1fr; align-items: center; gap: 18px; padding: 10px 0 9px; border-top: 3px solid var(--ink); border-bottom: 1px solid var(--ink); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }}
    .masthead div:nth-child(2) {{ text-align: center; }}
    .masthead div:last-child {{ text-align: right; }}
    .brandline {{ margin: 24px 0 18px; border-bottom: 4px solid var(--ink); font-family: var(--serif); font-size: clamp(54px, 9vw, 126px); line-height: .86; letter-spacing: -.065em; }}
    .kicker-row {{ display: grid; grid-template-columns: 1fr 280px; align-items: end; gap: 28px; padding: 13px 0 16px; border-bottom: 1px solid var(--ink); }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ padding-left: 18px; border-left: 10px solid var(--blue); font-family: var(--serif); font-size: clamp(32px, 5vw, 60px); line-height: .95; letter-spacing: -.04em; }}
    .dek {{ font-size: 12px; line-height: 1.45; text-transform: uppercase; letter-spacing: .07em; font-weight: 700; }}
    .hero-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 24px; margin-top: 18px; padding-bottom: 18px; border-bottom: 1px solid var(--ink); }}
    figure {{ margin: 0; }}
    .hero-image {{ display: block; width: 100%; aspect-ratio: 1.6; object-fit: cover; border: 1px solid var(--ink); }}
    figcaption {{ padding-top: 7px; color: var(--muted); font-size: 10px; line-height: 1.35; }}
    .brief {{ padding-left: 18px; border-left: 1px solid #777; }}
    .brief-title, .label {{ color: var(--blue); font-size: 11px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; }}
    .brief-title {{ margin-bottom: 4px; }}
    .pitch-diagram {{ display: block; width: 100%; margin-top: 5px; }}
    .svg-caption {{ font: 7px var(--sans); letter-spacing: .08em; fill: var(--muted); }}
    .metric-list {{ margin-top: 8px; border-top: 1px solid var(--rule); }}
    .metric {{ display: flex; justify-content: space-between; gap: 10px; padding: 8px 0; border-bottom: 1px solid #aaa; font-size: 11px; text-transform: uppercase; }}
    .metric strong {{ color: var(--blue); font-size: 17px; font-weight: 600; }}
    .stories {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; margin-top: 16px; border-bottom: 1px solid var(--ink); }}
    .story {{ min-height: 200px; padding: 0 18px 18px; border-right: 1px solid #aaa; }}
    .story:first-child {{ padding-left: 0; }}
    .story:nth-child(3) {{ border-right: 0; }}
    .story:nth-child(n+4) {{ padding-top: 18px; border-top: 1px solid #aaa; }}
    .story-index {{ margin-bottom: 8px; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
    .story h3 {{ margin-bottom: 8px; font-family: var(--serif); font-size: 25px; line-height: 1; letter-spacing: -.02em; }}
    .story p {{ font-size: 13px; line-height: 1.5; }}
    .section-heading {{ margin-top: 26px; padding-bottom: 8px; border-bottom: 2px solid var(--ink); font-family: var(--serif); font-size: 29px; letter-spacing: -.03em; }}
    .timeline-grid {{ display: grid; grid-template-columns: 1.25fr .75fr; gap: 28px; margin-top: 14px; }}
    .activity-chart {{ display: block; width: 100%; height: auto; padding: 10px 0 0; }}
    .chart-value {{ font: 12px var(--sans); fill: var(--ink); }}
    .chart-label {{ font: 11px var(--sans); fill: var(--muted); }}
    .timeline-note {{ padding: 12px 0 0 18px; border-left: 8px solid var(--blue); font-size: 14px; line-height: 1.5; }}
    .timeline-note strong {{ display: block; margin-bottom: 6px; font-family: var(--serif); font-size: 22px; line-height: 1.1; }}
    .evidence-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 28px; margin-top: 20px; padding-top: 18px; border-top: 1px solid var(--ink); }}
    .evidence-grid h2 {{ margin-bottom: 8px; font-family: var(--serif); font-size: 26px; letter-spacing: -.03em; }}
    .evidence-grid ul {{ list-style: none; padding: 0; margin: 0; border-top: 1px solid var(--ink); }}
    .evidence-grid li {{ display: grid; grid-template-columns: 82px 1fr auto; gap: 8px; padding: 8px 0; border-bottom: 1px solid #aaa; font-size: 12px; line-height: 1.35; }}
    .evidence-grid li time {{ color: var(--muted); }}
    .evidence-grid li small {{ color: var(--blue); font-size: 10px; font-weight: 700; }}
    .worktree li {{ grid-template-columns: 1fr auto; }}
    .empty-row {{ color: var(--muted); font-style: italic; }}
    .diagnostic {{ width: 100%; margin-top: 12px; border: 1px solid var(--ink); }}
    .footer {{ display: flex; justify-content: space-between; gap: 20px; margin-top: 30px; padding-top: 10px; border-top: 1px solid var(--ink); color: var(--muted); font-size: 10px; line-height: 1.4; }}
    @media (max-width: 760px) {{
      html, body {{ max-width: 100%; overflow-x: hidden; }}
      .paper {{ margin: 0; padding: 18px 18px 30px; box-shadow: none; }}
      .masthead {{ grid-template-columns: 1fr; gap: 5px; }}
      .masthead div, .masthead div:nth-child(2), .masthead div:last-child {{ text-align: left; }}
      .masthead div {{ overflow-wrap: anywhere; letter-spacing: .04em; }}
      .brandline {{ font-size: 48px; overflow-wrap: anywhere; }}
      .kicker-row, .hero-grid, .timeline-grid, .evidence-grid {{ grid-template-columns: 1fr; }}
      .brief {{ padding: 16px 0 0; border-left: 0; border-top: 1px solid #777; }}
      .stories {{ grid-template-columns: 1fr; }}
      .story, .story:first-child, .story:nth-child(3), .story:nth-child(n+4) {{ min-height: 0; padding: 16px 0; border-right: 0; border-top: 1px solid #aaa; }}
      .story:first-child {{ border-top: 0; }}
      .footer {{ display: block; }}
      .footer span {{ display: block; margin-top: 8px; }}
    }}
    @media print {{ body {{ background: white; }} .paper {{ width: 100%; margin: 0; padding: 10mm 12mm; box-shadow: none; }} }}
  </style>
</head>
<body>
  <main class="paper">
    <header class="masthead">
      <div>Weekly Development Report</div><div>The Signal / The System</div><div>{esc(week_label)} &nbsp;|&nbsp; Issue {issue}</div>
    </header>
    <div class="brandline">Soccer Analysis</div>
    <section class="kicker-row">
      <h1>This Week in Motion</h1>
      <p class="dek">关于实时足球计算机视觉系统的周度开发进展、优化与验证证据</p>
    </section>

    <section class="hero-grid">
      <figure>
        <img class="hero-image" src="{REPORT_ASSET}" alt="足球计算机视觉项目的编辑式视觉概念图" />
        <figcaption>Editorial reference image generated for this Artifact; project claims below are derived from the local repository window {esc(week_label)}.</figcaption>
      </figure>
      <aside class="brief">
        <div class="brief-title">System View</div>
        <div style="font-family:var(--serif);font-size:24px;line-height:1.05;">从帧到场地空间</div>
        {pitch_diagram()}
        <div class="metric-list">
          <div class="metric"><span>Committed changes</span><strong>{len(commits)}</strong></div>
          <div class="metric"><span>Files touched</span><strong>{stats['files']}</strong></div>
          <div class="metric"><span>Lines + / −</span><strong>{format_number(stats['additions'])} / {format_number(stats['deletions'])}</strong></div>
          <div class="metric"><span>Tests in inventory</span><strong>{test_count}</strong></div>
        </div>
      </aside>
    </section>

    <section class="stories">{''.join(area_articles[:3])}</section>
    <section class="stories">{''.join(area_articles[3:])}</section>

    <h2 class="section-heading">Development Timeline</h2>
    <section class="timeline-grid">
      <div>{bar_chart(daily)}</div>
      <div class="timeline-note"><strong>{esc(headline)}</strong><span>最新记录：{esc(latest)}。当前工作树：{esc(worktree_label)}。</span></div>
    </section>

    <section class="evidence-grid">
      <article>
        <div class="label">Committed Record</div>
        <h2>提交日志</h2>
        <ul>{commit_rows}</ul>
      </article>
      <article>
        <div class="label">Uncommitted Surface</div>
        <h2>工作树观察</h2>
        <ul class="worktree">{worktree_rows}</ul>
        <p style="margin-top:12px;font-size:11px;color:var(--muted);">测试基线：{esc(test_source)}；本页不把未执行的测试写成“通过”。</p>
      </article>
    </section>

    <section style="margin-top:24px;padding-top:16px;border-top:1px solid var(--ink);">
      <div class="label">Visual Evidence</div>
      <h2 class="section-heading" style="margin-top:6px;">诊断画面 / Field Evidence</h2>
      <figure>
        <img class="diagnostic" src="{DIAGNOSTIC_ASSET}" alt="项目诊断画面联系表，展示球员检测、投影与球追踪结果" />
        <figcaption>如果本地诊断联系表存在，它会作为项目证据图一并呈现；缺失时浏览器会保留空位，不影响报告其余内容。</figcaption>
      </figure>
    </section>

    <footer class="footer"><span>Generated from local Git history, working-tree status and test inventory.</span><span>Snapshot: {esc(now_label)} · HEAD {esc(run_git('rev-parse', '--short', 'HEAD') or 'unknown')}</span></footer>
  </main>
</body>
</html>
"""
    output.write_text(html_doc, encoding="utf-8")
    latest_path.write_text(html_doc, encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week-start", help="Monday date in YYYY-MM-DD; defaults to this week's Monday")
    parser.add_argument("--until", help="End timestamp in ISO format; defaults to now")
    parser.add_argument("--output", type=Path, help="Optional report path; latest.html is always updated")
    args = parser.parse_args()
    now = datetime.now().astimezone()
    end = datetime.fromisoformat(args.until).astimezone() if args.until else now
    start = parse_start(args.week_start, end)
    path = write_report(start, end, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
