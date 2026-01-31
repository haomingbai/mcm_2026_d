from __future__ import annotations

import re
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup


_BREF = "https://www.basketball-reference.com"


@dataclass(frozen=True)
class BRefFetchConfig:
    cache_dir: Path
    sleep_s: float = 1.0
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _get(url: str, cfg: BRefFetchConfig) -> str:
    headers = {"User-Agent": cfg.user_agent}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text


def fetch_year_html(*, year: int, table: str, cfg: BRefFetchConfig, force: bool = False) -> Path:
    """下载并缓存 BRef WNBA 年度统计页面 HTML。

    table 取值示例：
    - "per_game"
    - "advanced"

    返回缓存文件路径。
    """

    _ensure_dir(cfg.cache_dir)
    out = cfg.cache_dir / f"bref_wnba_{year}_{table}.html"
    if out.exists() and not force:
        return out

    url = f"{_BREF}/wnba/years/{year}_{table}.html"
    html = _get(url, cfg)
    out.write_text(html, encoding="utf-8")
    time.sleep(cfg.sleep_s)
    return out


def _read_table_from_html(html: str, table_id: str) -> pd.DataFrame:
    """解析 BRef HTML 里的指定 table（含被注释包裹的情况）。"""

    def _parse_table_by_datastat(table_tag) -> pd.DataFrame:
        """BRef 的 HTML 有时不够规整，pd.read_html 可能把整行粘成一个字符串。

        这里按每个单元格的 data-stat 顺序进行解析，更稳健。
        """

        thead = table_tag.find("thead")
        if thead is None:
            raise ValueError("table has no thead")
        header_rows = thead.find_all("tr")
        if not header_rows:
            raise ValueError("table has empty thead")

        # BRef 最后一行 th 通常是字段列表
        header_ths = header_rows[-1].find_all("th")
        stats = [th.get("data-stat") or "" for th in header_ths]

        # 去重列名（模仿 pandas 的 .1/.2 后缀）
        seen: dict[str, int] = {}
        cols: list[str] = []
        for s in stats:
            base = str(s)
            n = seen.get(base, 0)
            seen[base] = n + 1
            cols.append(base if n == 0 else f"{base}.{n}")

        tbody = table_tag.find("tbody")
        if tbody is None:
            raise ValueError("table has no tbody")

        rows: list[list[str | None]] = []
        for tr in tbody.find_all("tr"):
            classes = tr.get("class") or []
            if "thead" in classes:
                continue

            # BRef 的 HTML 有时会因为标签未闭合导致 <td> 被嵌在 <th data-stat="player"> 里。
            # 这里：player 用链接文本拿到名字，其它列只取“叶子节点”单元格，避免把整行粘进 player。
            player_th = tr.find("th", {"data-stat": "player"})
            if player_th is None:
                continue
            a = player_th.find("a")
            player_name = a.get_text(strip=True) if a is not None else player_th.get_text(" ", strip=True)

            cells = tr.find_all(["th", "td"], attrs={"data-stat": True})
            leaf_cells = [
                c
                for c in cells
                if c.get("data-stat") != "player" and c.find(["th", "td"], attrs={"data-stat": True}) is None
            ]
            other_vals = [c.get_text(" ", strip=True) for c in leaf_cells]
            vals = [player_name] + other_vals

            if len(vals) == 0:
                continue
            if len(vals) < len(cols):
                vals = vals + [None] * (len(cols) - len(vals))
            rows.append(vals[: len(cols)])

        df = pd.DataFrame(rows, columns=cols)

        # 把常用字段名映射到与其它代码一致的列名
        rename = {
            "player": "Player",
            "team": "Tm",
            "pos": "Pos",
            "g": "G",
            "mp": "MP",
            "per": "PER",
            "ts_pct": "TS%",
            "usg_pct": "USG%",
            "ast_pct": "AST%",
            "trb_pct": "TRB%",
            "stl_pct": "STL%",
            "blk_pct": "BLK%",
            "tov_pct": "TOV%",
            "ows": "OWS",
            "dws": "DWS",
            "ws": "WS",
            "ws_per_40": "WS/40",
            "bpm": "BPM",
        }
        df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
        return df

    def _looks_broken(df: pd.DataFrame) -> bool:
        if df.empty:
            return True
        # 正常情况下 minutes / games 不应几乎全空
        for c in ["MP", "G"]:
            if c in df.columns:
                try:
                    frac = float(pd.to_numeric(df[c], errors="coerce").notna().mean())
                except Exception:
                    frac = float(df[c].notna().mean())
                if frac < 0.2:
                    return True
        return False

    soup = BeautifulSoup(html, "lxml")

    # 1) 常规 table
    table = soup.find("table", {"id": table_id})
    if table is not None:
        # 注意：pd.read_html(str) 会把字符串当成“文件路径”处理，
        # 这里用 StringIO 强制按 HTML 内容解析。
        df = pd.read_html(StringIO(str(table)))[0]
        if _looks_broken(df):
            df = _parse_table_by_datastat(table)
        return df

    # 2) BRef 常把表放在 <!-- --> 注释里
    comments = soup.find_all(string=lambda s: isinstance(s, str) and "table" in s)
    for c in comments:
        if table_id in c:
            try:
                inner = BeautifulSoup(c, "lxml")
                table2 = inner.find("table", {"id": table_id})
                if table2 is not None:
                    df = pd.read_html(StringIO(str(table2)))[0]
                    if _looks_broken(df):
                        df = _parse_table_by_datastat(table2)
                    return df
            except Exception:
                continue

    raise ValueError(f"Could not find table id={table_id}")


def load_year_table(*, html_path: Path, table_id: str) -> pd.DataFrame:
    html = html_path.read_text(encoding="utf-8")
    df = _read_table_from_html(html, table_id=table_id)
    return df


def clean_players_df(df: pd.DataFrame) -> pd.DataFrame:
    """清洗 BRef 球员表：去掉表头重复行、TOT聚合行等。"""

    df = df.copy()

    # 统一列名
    df.columns = [str(c).strip() for c in df.columns]

    # 去掉重复表头行
    if "Player" in df.columns:
        df = df[df["Player"].astype(str) != "Player"]

    # 去 TOT（跨队合计）行，保留单队行更细
    if "Tm" in df.columns:
        df = df[df["Tm"].astype(str) != "TOT"]

    # 去掉空玩家
    if "Player" in df.columns:
        df = df[df["Player"].notna()]

    # Player 名字去掉星号（名人堂标记）
    df["Player"] = df["Player"].astype(str).str.replace("*", "", regex=False).str.strip()

    return df.reset_index(drop=True)


def _to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_feature_pool(
    *,
    advanced_df: pd.DataFrame,
    year: int,
    min_mp: float = 200.0,
) -> pd.DataFrame:
    """从 advanced 表构建一个“球员特征池”。

    输出列：
    - year
    - player
    - features...（用于能力向量）
    - mp（分钟）

    注：BRef 的 WNBA advanced 表字段可能随年份略有变化，这里用尽量稳健的字段集合。
    """

    df = clean_players_df(advanced_df)

    wanted = [
        "Player",
        "Tm",
        "MP",
        "G",
        "TS%",
        "USG%",
        "AST%",
        "TRB%",
        "STL%",
        "BLK%",
        "TOV%",
        "OWS",
        "DWS",
        "WS",
        "BPM",
    ]
    have = [c for c in wanted if c in df.columns]
    df = df[have].copy()

    numeric_cols = [c for c in have if c not in {"Player", "Tm"}]
    df = _to_numeric(df, numeric_cols)

    # 基础过滤：上场分钟太少的球员训练意义不大
    if "MP" in df.columns:
        df = df[df["MP"].fillna(0.0) >= min_mp]

    # 填补缺失：对数值列用列中位数
    for c in numeric_cols:
        med = float(df[c].median()) if df[c].notna().any() else 0.0
        df[c] = df[c].fillna(med)

    df.insert(0, "year", int(year))
    df.rename(columns={"Player": "player", "MP": "mp", "G": "g"}, inplace=True)

    # 去重：同名球员可能多队行，简单按 mp 最大保留一条
    df = df.sort_values("mp", ascending=False).drop_duplicates(subset=["year", "player"], keep="first")

    # 规范化百分比列：BRef 表里 TS% / USG% 有时是 0.xx 或 12.3，做启发式修正
    for pct in ["TS%", "USG%", "AST%", "TRB%", "STL%", "BLK%", "TOV%"]:
        if pct in df.columns:
            v = df[pct].astype(float)
            # 若中位数 > 2，认为是百分数形式（例如 23.1），转成 0.231
            if float(v.median()) > 2.0:
                df[pct] = v / 100.0

    return df.reset_index(drop=True)


def default_cache_dir() -> Path:
    return Path("data/raw/bref")
