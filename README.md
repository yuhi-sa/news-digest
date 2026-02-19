# News Digest / ニュースダイジェスト

エンジニア兼投資家向けの自動ニュース・論文ダイジェストシステム。40以上のRSSフィードから記事を収集し、Google Gemini AIで日本語要約・ブリーフィングを生成、さらにarXivから注目論文を選定して構造化サマリーを作成する。すべてGitHub ActionsでPRとして自動配信。

## Architecture

```
┌─ Collect (06:00 JST) ─────────────────────────────────┐
│  RSS Feeds (40+) → Parser → Dedup → Summarizer        │
│  → digests/YYYY-MM-DD.md + weekly_articles.json buffer │
└────────────────────────────────────────────────────────┘
         ↓
┌─ Digest PR (07:00 JST) ──────────────────────────────────────┐
│  Buffered articles → Gemini Stage 1 (select top articles)    │
│  → Full text fetch → Gemini Stage 2 (deep briefing)          │
│  → Gemini Stage 3 (quality refinement) → Post-processing     │
│  → PR with briefing body                                     │
└──────────────────────────────────────────────────────────────┘
         ↓
┌─ Paper Digest (07:30 JST) ───────────────────────────────────┐
│  arXiv RSS (primary) / API (fallback) → Paper selection      │
│  → Abstract enrichment → Gemini structured summary → PR      │
│  Categories: 分散システム / セキュリティ / AI / クラウド (日替わり) │
└──────────────────────────────────────────────────────────────┘
```

### 3-stage daily pipeline (GitHub Actions)

1. **Collect** (06:00 JST): 40以上のRSSフィードから並列取得 → URL正規化+タイトル類似度で重複排除 → Gemini日本語要約 → `digests/YYYY-MM-DD.md`生成 → `data/weekly_articles.json`にバッファ → mainにコミット
2. **Digest PR** (07:00 JST): バッファ記事を読み込み → Geminiで2段階分析（トップ記事選定 → フルテキスト取得 → 深掘りブリーフィング） → 決定論的後処理（リンク検証、市場数値チェック、禁止フレーズ監視） → PR作成
3. **Paper Digest** (07:30 JST): 4カテゴリ（分散システム、セキュリティ、AI、クラウド）をday_of_year % 4で日替わり選択 → arXiv RSSから論文取得（API fallback付き） → 未読論文を選定 → Geminiで構造化サマリー（前提知識、手法、貢献、実務応用） → PR作成

## Briefing Features

- **ハイライト**: 最重要ニュース3件を深掘り解説
- **テクノロジー**: エンジニア向け技術動向
- **データエンジニアリング**: データ基盤・パイプライン関連
- **セキュリティ**: CVE番号・深刻度・対応アクション付き（3-5件に厳選）
- **マーケット**: 具体的な指標数値（S&P500, NASDAQ, USD/JPY, 金価格等）
- **今後の注目**: 決算発表、経済指標発表の予定

## Paper Digest Features

- **前提知識**: 論文理解に必要な背景知識を平易に解説
- **手法**: アーキテクチャ図（Mermaid）付きのアプローチ説明
- **主要な貢献**: 論文の技術的貢献を箇条書きで整理
- **実務への応用可能性**: エンジニアが業務に活かせる観点

## Sources (40+ feeds)

| Category | Feeds |
|----------|-------|
| Engineering & Technology | ArXiv AI/ML, Hacker News, IEEE Spectrum AI, Ars Technica, MIT Technology Review, Google Developers Blog, Google AI Blog |
| Languages & Frameworks | Vercel, TypeScript, Go, Python Insider |
| Infrastructure | Kubernetes, CNCF, Kafka, MySQL, Redis, Cassandra |
| Data Engineering | dbt, Data Engineering Weekly, Databricks, AWS Big Data, Seattle Data Guy, Towards Data Science |
| Security | CISA, Krebs on Security, Schneier, The Hacker News, BleepingComputer, SANS ISC, Project Zero |
| Economy & Finance | Bloomberg Economics/Markets, Reuters, Investing.com |
| Investment & Markets | Seeking Alpha, Yahoo Finance, MarketWatch, CNBC Markets, Yahoo Finance Japan, Google News (日経平均, 米国株) |

## Usage

### Local (dry-run)

```bash
pip install -r requirements.txt

# 記事収集
python -m src.main collect --dry-run

# ブリーフィングPR生成
python -m src.main digest --dry-run

# 論文ダイジェストPR生成
python -m src.main paper --dry-run
```

### Run tests

```bash
pytest tests/ -v
```

### GitHub Actions

1. リポジトリのSecretに `GEMINI_API_KEY` を設定
2. 毎日3つのワークフローが自動実行（06:00 / 07:00 / 07:30 JST）
3. `workflow_dispatch` で手動実行も可能

## Configuration

- `config/feeds.yml` — フィードの追加・削除・カテゴリ設定
- `SUMMARIZER_API_KEY` — Google Gemini APIキー。未設定時はPassthroughSummarizerにフォールバック（LLM要約なし）

## Tech Stack

- Python 3.12+
- Google Gemini API (要約・ブリーフィング・論文サマリー)
- arXiv RSS + API (論文取得)
- feedparser (RSS解析)
- GitHub Actions (自動実行)
- `gh` CLI (PR作成)
