# 🧩 **Design Document: Topic–Article Relevance Assessment System**

**目的**
ある文章（Article）が、特定のトピック（Topic）に関連しているかどうかを、
**人間・自動判定（LLM / Rule）両方で評価し、
評価履歴・データセットを体系的に管理できるプラットフォームを構築する。**

本ドキュメントでは、コアドメインモデル・データ構造・ユースケース・制約・非機能要件を整理する。

---

# 1. Core Concepts Overview

本システムは以下の5つの概念を中心に構成される：

1. **Article / ArticleVersion**
   → 判定対象の文章。

2. **Topic / TopicVersion**
   → 関連性を評価したいトピック。

3. **Judge（Human / Automated）**
   → 判定主体（人間・LLM・ルールベースなど）。

4. **RelevanceAssessment**
   → 判定結果。

5. **Dataset / DatasetVersion / DatasetEntry**
   → ArticleVersion の集合（期間コーパスや学習セットなど）。

これらにより、判定・データ収集・学習・分析の全プロセスを統一的に扱える。

---

# 2. Domain Model

## 2.1 Article / ArticleVersion

文章（Article）は論理IDとして扱い、実際の内容はバージョンとして保持する。

### **Article**

| フィールド               | 型          | 説明                                                             | 制約                  |
| ------------------- | ---------- | -------------------------------------------------------------- | ------------------- |
| `article_id`        | UUID       | 論理的な文章ID                                                       | Primary Key         |
| `source_type`       | String(50) | `"twitter_community_note"`, `"csv_import"`, `"langfuse"`, etc. | NOT NULL            |
| `external_key`      | String     | 外部ID（note_id, trace_id など）                                     | Unique per source   |
| `latest_version_id` | UUID       | 最新ArticleVersion                                               | FK → ArticleVersion |
| `status`            | Enum       | `active`, `deleted`                                            | NOT NULL, default=active            |
| `created_at`        | Timestamp  | 初回作成時刻                                                         | NOT NULL            |
| `updated_at`        | Timestamp  | 最終更新時刻                                                         | NOT NULL            |

**制約**:
- `(source_type, external_key)` は UNIQUE
- 論理削除（`status=deleted`）を採用

---

### **ArticleVersion**

| フィールド                | 型         | 説明                   | 制約              |
| -------------------- | --------- | -------------------- | --------------- |
| `article_version_id` | UUID      | バージョンID              | Primary Key     |
| `article_id`         | UUID      | 親Article             | FK → Article    |
| `text`               | Text      | 本文                   | NOT NULL        |
| `context_json`       | JSONB     | メタ情報（型なしJSON）        | nullable        |
| `language`           | String(5) | 言語（ISO 639-1）        | nullable        |
| `content_hash`       | String(64)| SHA-256(text + context) | NOT NULL, Indexed |
| `created_at`         | Timestamp | 作成時刻                 | NOT NULL        |

**制約**:
- `content_hash` は UNIQUE（同一内容は同一バージョンとして再利用）
- INSERT 前に既存 `content_hash` を検索し、存在すれば再利用

**役割**
* 判定対象は常に ArticleVersion
* Article が更新された場合は新しいバージョンとして記録

---

## 2.2 Topic / TopicVersion

### Topic

| フィールド               | 型          | 説明                        | 制約                         |
| ------------------- | ---------- | ------------------------- | -------------------------- |
| `topic_id`          | UUID       | トピックID                    | Primary Key                |
| `name`              | String(200)| 表示名                       | NOT NULL, UNIQUE           |
| `status`            | Enum       | `draft`, `active`, `archived` | NOT NULL, default=draft    |
| `latest_version_id` | UUID       | 最新TopicVersion            | FK → TopicVersion          |
| `created_at`        | Timestamp  | 作成時刻                      | NOT NULL                   |
| `updated_at`        | Timestamp  | 最終更新時刻                    | NOT NULL                   |

### TopicVersion

| フィールド              | 型         | 説明               | 制約                |
| ------------------ | --------- | ---------------- | ----------------- |
| `topic_version_id` | UUID      | バージョンID          | Primary Key       |
| `topic_id`         | UUID      | 親Topic           | FK → Topic        |
| `definition`       | Text      | 説明文・判断ガイドライン     | NOT NULL          |
| `definition_hash`  | String(64)| SHA-256(definition) | NOT NULL, Indexed |
| `created_at`       | Timestamp | 作成時刻             | NOT NULL          |

**制約**:
- `(topic_id, definition_hash)` は UNIQUE（同一定義の重複防止）

**役割**
* トピック定義の変更に対応
* 判定結果は特定の TopicVersion に紐づく

---

## 2.3 Judge（判定主体）

### Judge（抽象）

| フィールド           | 型          | 説明                         | 制約           |
| --------------- | ---------- | -------------------------- | ------------ |
| `judge_id`      | UUID       | 判定主体ID                     | Primary Key  |
| `kind`          | Enum       | `"human"`, `"automated"`   | NOT NULL     |
| `display_name`  | String(100)| 名前                         | NOT NULL     |
| `status`        | Enum       | `active`, `inactive`       | NOT NULL, default=active     |
| `metadata_json` | JSONB      | 任意のメタ情報                    | nullable     |
| `created_at`    | Timestamp  | 作成時刻                       | NOT NULL     |
| `updated_at`    | Timestamp  | 最終更新時刻                     | NOT NULL     |

---

### HumanJudge

| フィールド      | 型          | 説明                                 | 制約                      |
| ---------- | ---------- | ---------------------------------- | ----------------------- |
| `judge_id` | UUID       | → Judge                            | FK → Judge, Primary Key |
| `user_id`  | String(100)| 外部ID（SSO / Langfuse userIdなど）      | NOT NULL, UNIQUE        |
| `role`     | Enum       | `annotator`, `reviewer`, `admin`   | NOT NULL                |

---

### AutomatedJudge

LLM / ルールベースなどを共通化した構造。

| フィールド             | 型          | 説明                                                  | 制約                      |
| ----------------- | ---------- | --------------------------------------------------- | ----------------------- |
| `judge_id`        | UUID       | → Judge                                             | FK → Judge, Primary Key |
| `automation_type` | Enum       | `"llm"`, `"rule"`, `"hybrid"`, `"external_service"` | NOT NULL                |
| `impl_key`        | String(100)| 実装を一意に識別するキー                                        | NOT NULL                |
| `version`         | String(50) | 実装バージョン                                             | NOT NULL                |
| `config_json`     | JSONB      | プロンプト・ルール設定など                                       | NOT NULL                |

**impl_key 命名規則**:
- LLM: `llm:{provider}:{model}` 例: `llm:openai:gpt-4-turbo-2024-04-09`
- Rule: `rule:{rule_name}` 例: `rule:keyword_matcher`
- Hybrid: `hybrid:{name}` 例: `hybrid:llm_with_keyword_filter`

**制約**:
- `(impl_key, version)` は UNIQUE（同一実装バージョンは1つのJudgeとして管理）

**拡張可能**
* 当面は LLM のみ → 後から rule-base を追加可能

---

## 2.4 RelevanceAssessment

「誰（judge）が、どのトピック（TopicVersion）に対し、
どの文章（ArticleVersion）をどう判断したか」を表す。

### RelevanceAssessment

| フィールド                   | 型         | 説明                                           | 制約              |
| ----------------------- | --------- | -------------------------------------------- | --------------- |
| `assessment_id`         | UUID      | ID                                           | Primary Key     |
| `judge_id`              | UUID      | → Judge                                      | FK → Judge      |
| `topic_version_id`      | UUID      | → TopicVersion                               | FK → TopicVersion |
| `article_version_id`    | UUID      | → ArticleVersion                             | FK → ArticleVersion |
| `label`                 | Enum      | `"relevant"`, `"not_relevant"`, `"unsure"`   | NOT NULL        |
| `confidence`            | Float     | 0–1（モデル判定など）                                 | nullable, CHECK (0 <= confidence <= 1) |
| `rationale`             | Text      | 任意説明（LLMの理由など）                               | nullable        |
| `supersedes_assessment_id` | UUID   | 再判定時の旧assessment_id                         | FK → RelevanceAssessment, nullable |
| `version`               | Integer   | 同一組み合わせの判定バージョン                              | NOT NULL, default=1 |
| `created_at`            | Timestamp | 判定時間                                         | NOT NULL        |

**制約**:
- `(judge_id, topic_version_id, article_version_id, version)` は UNIQUE
- 再判定時は `version` をインクリメント、`supersedes_assessment_id` に旧IDをセット
- Index: `(judge_id, topic_version_id, article_version_id, created_at DESC)` で最新取得を高速化

**confidence の扱い**:
- HumanJudge: 通常は `null`（確信度は記録しない）
- AutomatedJudge (LLM): モデルの確率値をセット
- `label=unsure` は「判定不能」、`confidence=0.5` は「確信度50%」（意味が異なる）

**特徴**
* 人間とモデルを同じ構造で扱える
* 複数 Judge の評価比較も容易
* 履歴が完全に残る（再判定も追跡可能）

---

## 2.5 Dataset / DatasetVersion / DatasetEntry

Dataset は **ArticleVersion の集合**
→ 学習用、評価用、期間コーパスなどを共通フォーマットで扱える。

### Dataset

| フィールド        | 型          | 説明                                        | 制約                 |
| ------------ | ---------- | ----------------------------------------- | ------------------ |
| `dataset_id` | UUID       | ID                                        | Primary Key        |
| `name`       | String(200)| 表示名（例: "community-notes-2025-q1"）         | NOT NULL, UNIQUE   |
| `purpose`    | Enum       | `corpus`, `analysis`, `training`, `evaluation` | NOT NULL           |
| `status`     | Enum       | `active`, `frozen`, `archived`            | NOT NULL, default=active |
| `created_at` | Timestamp  | 作成時刻                                      | NOT NULL           |
| `updated_at` | Timestamp  | 最終更新時刻                                    | NOT NULL           |

**status の意味**:
- `active`: エントリの追加・削除が可能
- `frozen`: 読み取り専用（学習・評価用に固定）
- `archived`: 非推奨・参照用

---

### DatasetVersion

| フィールド                       | 型         | 説明                     | 制約                         |
| --------------------------- | --------- | ---------------------- | -------------------------- |
| `dataset_version_id`        | UUID      | バージョンID                | Primary Key                |
| `dataset_id`                | UUID      | 親Dataset               | FK → Dataset               |
| `version_number`            | Integer   | バージョン番号（1, 2, 3...）    | NOT NULL                   |
| `created_at`                | Timestamp | 作成時刻                   | NOT NULL                   |
| `description`               | Text      | 説明                     | nullable                   |
| `kind`                      | Enum      | `"base"`, `"derived"`  | NOT NULL                   |
| `spec_json`                 | JSONB     | 抽出条件（期間・フィルタ条件など）      | NOT NULL                   |
| `parent_dataset_version_id` | UUID      | 派生元（サブセットの場合）          | FK → DatasetVersion, nullable |
| `entry_count`               | Integer   | エントリ数（非正規化キャッシュ）       | NOT NULL, default=0        |
| `frozen_at`                 | Timestamp | 固定化時刻                  | nullable                   |

**制約**:
- `(dataset_id, version_number)` は UNIQUE
- `kind=derived` の場合は `parent_dataset_version_id` が NOT NULL

**特徴**
* "base": 期間で切った生セット
* "derived": 特定トピック・ラベル条件などでサブセットを生成

---

### DatasetEntry

| フィールド                | 型         | 説明               | 制約                           |
| -------------------- | --------- | ---------------- | ---------------------------- |
| `dataset_version_id` | UUID      | → DatasetVersion | FK → DatasetVersion          |
| `article_version_id` | UUID      | → ArticleVersion | FK → ArticleVersion          |
| `entry_order`        | Integer   | エントリの順序（オプション）   | nullable                     |
| `added_at`           | Timestamp | 追加時刻             | NOT NULL                     |

**制約**:
- `(dataset_version_id, article_version_id)` は UNIQUE（重複エントリ防止）
- Primary Key: `(dataset_version_id, article_version_id)`

✦ ラベルは Dataset に含まれない（RelevanceAssessment に存在）
→ Dataset は純粋に「ArticleVersion の集合」

---

# 3. Constraints & Invariants

本セクションでは、データ整合性を保つための制約と不変条件を定義する。

## 3.1 データ整合性制約

### Article / ArticleVersion
- 同一 `(source_type, external_key)` の Article は1つのみ存在
- 同一 `content_hash` の ArticleVersion は1つのみ存在（重複防止）
- `Article.latest_version_id` は常に有効な ArticleVersion を参照
- `status=deleted` の Article の ArticleVersion も論理的には保持（履歴保存）

### Topic / TopicVersion
- 同一 `name` の Topic は1つのみ存在
- 同一 `(topic_id, definition_hash)` の TopicVersion は1つのみ存在
- `Topic.latest_version_id` は常に有効な TopicVersion を参照

### RelevanceAssessment
- 同一 `(judge_id, topic_version_id, article_version_id, version)` の組は1つのみ
- `supersedes_assessment_id` が non-null の場合、必ず `version > 1`
- `confidence` は HumanJudge では通常 null、AutomatedJudge では 0–1 の範囲

### Dataset / DatasetVersion / DatasetEntry
- 同一 `(dataset_id, version_number)` の DatasetVersion は1つのみ
- 同一 `(dataset_version_id, article_version_id)` の DatasetEntry は1つのみ
- `kind=derived` の DatasetVersion は `parent_dataset_version_id` が NOT NULL
- `status=frozen` の Dataset に属する DatasetVersion は変更不可

## 3.2 ビジネスルール

### Judge の有効性
- `Judge.status=active` のみが新規判定を実行可能
- `Judge.status=inactive` の Judge の過去判定は保持

### Dataset の固定化
- `DatasetVersion.frozen_at` が non-null の場合、DatasetEntry の追加・削除は禁止
- 学習・評価用データセットは frozen_at をセットして固定化

### 再判定のバージョニング
- 同じ Judge が同じ (topic_version, article_version) を再判定する場合：
  1. 最新の `version` を取得
  2. `version + 1` で新しい RelevanceAssessment を作成
  3. `supersedes_assessment_id` に旧 `assessment_id` をセット

---

# 4. JSON Schema Definitions

各 JSONB フィールドの構造を定義する。

## 4.1 ArticleVersion.context_json

文章のメタデータを格納。source_type により構造が異なる。

### Twitter Community Note の場合
```json
{
  "tweet_id": "string",
  "note_id": "string",
  "author_id": "string?",
  "created_at": "ISO8601 timestamp",
  "topics": ["string"],
  "classification": "string?"
}
```

### CSV Import の場合
```json
{
  "row_number": "integer",
  "import_batch_id": "string",
  "original_fields": {
    "任意のキー": "任意の値"
  }
}
```

### Langfuse の場合
```json
{
  "trace_id": "string",
  "observation_id": "string?",
  "user_id": "string?",
  "session_id": "string?",
  "metadata": {}
}
```

---

## 4.2 Judge.metadata_json

Judge 固有のメタデータ。

### HumanJudge
```json
{
  "email": "string?",
  "organization": "string?",
  "expertise_areas": ["string"]
}
```

### AutomatedJudge
```json
{
  "created_by": "user_id",
  "deployment_env": "production | staging | development",
  "cost_per_call": "number?"
}
```

---

## 4.3 AutomatedJudge.config_json

LLM / Rule の実行設定。

### LLM の場合
```json
{
  "provider": "openai | anthropic | azure | etc",
  "model": "gpt-4-turbo-2024-04-09",
  "temperature": 0.0,
  "max_tokens": 500,
  "system_prompt": "You are a relevance judge...",
  "user_prompt_template": "Topic: {topic_definition}\n\nArticle: {article_text}\n\nIs this article relevant?",
  "response_format": {
    "type": "json_object",
    "schema": {...}
  }
}
```

### Rule の場合
```json
{
  "rule_type": "keyword | regex | embedding_similarity",
  "keywords": ["keyword1", "keyword2"],
  "threshold": 0.7,
  "case_sensitive": false
}
```

---

## 4.4 DatasetVersion.spec_json

Dataset の抽出条件を定義。

### Base Dataset（時間範囲）
```json
{
  "type": "time_slice",
  "from": "2025-01-01T00:00:00Z",
  "to": "2025-03-31T23:59:59Z",
  "source_types": ["twitter_community_note"],
  "filters": {
    "language": ["ja", "en"]
  }
}
```

### Derived Dataset（フィルタ）
```json
{
  "type": "subset",
  "parent_dataset_version_id": "uuid",
  "filters": {
    "topic_version_id": "uuid",
    "labels": ["relevant", "not_relevant"],
    "min_confidence": 0.8,
    "judge_ids": ["uuid1", "uuid2"]
  },
  "sample": {
    "method": "random | stratified",
    "size": 1000
  }
}
```

---

# 5. Status Transition Rules

各エンティティのステータス遷移ルールを定義。

## 5.1 Article.status

```
[新規作成] → active
active → deleted (論理削除)
deleted → active (復元可能)
```

**ルール**:
- 論理削除を採用（物理削除は行わない）
- 復元は管理者のみ実行可能

---

## 5.2 Topic.status

```
[新規作成] → draft
draft → active (公開)
active → archived (廃止)
archived → active (再公開可能)
```

**ルール**:
- `draft`: 定義作成中、判定には使用不可
- `active`: 判定に使用可能
- `archived`: 新規判定には使用不可、過去判定は保持

---

## 5.3 Judge.status

```
[新規作成] → active
active → inactive (無効化)
inactive → active (再有効化)
```

**ルール**:
- `inactive` の Judge は新規判定を実行不可
- 過去の判定結果は保持

---

## 5.4 Dataset.status

```
[新規作成] → active
active → frozen (固定化)
frozen → archived (アーカイブ)
archived → frozen (再利用)
```

**ルール**:
- `active`: エントリの追加・削除が可能
- `frozen`: DatasetEntry の変更不可（学習・評価用）
- `archived`: 非推奨、参照のみ

---

# 6. Transaction Boundaries

各ユースケースにおけるトランザクション境界を定義。

## 6.1 Article 作成 + 判定

### トランザクション1: Article 永続化
```
BEGIN
  1. content_hash を計算
  2. 既存 ArticleVersion を検索
  3. 存在しない場合:
     - Article を作成
     - ArticleVersion を作成
     - Article.latest_version_id を更新
COMMIT
```

### トランザクション2: 判定実行
```
BEGIN
  1. AutomatedJudge を取得
  2. TopicVersion を取得
  3. LLM を呼び出し（外部API、トランザクション外）
  4. RelevanceAssessment を作成
COMMIT
```

**ロールバック戦略**:
- LLM 呼び出し失敗 → Assessment は作成しない（Article は残る）
- Assessment 作成失敗 → リトライ可能

---

## 6.2 CSV インポート

```
BEGIN
  FOR EACH row IN csv:
    1. Article + ArticleVersion を作成（既存は再利用）
    2. HumanJudge を作成（annotator が未登録の場合）
    3. RelevanceAssessment を作成
    4. Dataset が指定されている場合:
       - Dataset + DatasetVersion を作成（初回のみ）
       - DatasetEntry を作成
COMMIT (全行成功時) or ROLLBACK (1行でも失敗時)
```

**エラーハンドリング**:
- 1行でもエラーがあれば全体をロールバック
- エラー行を報告してユーザーに修正を促す

---

## 6.3 Dataset 作成

### Base Dataset
```
BEGIN
  1. Dataset を作成
  2. DatasetVersion を作成
  3. spec_json の条件で ArticleVersion を検索
  4. 該当する ArticleVersion の DatasetEntry を一括作成
  5. entry_count を更新
COMMIT
```

### Derived Dataset
```
BEGIN
  1. parent DatasetVersion のエントリを取得
  2. フィルタ条件（ラベル・Judge・confidence）で絞り込み
  3. 新しい DatasetVersion を作成
  4. DatasetEntry を一括作成
  5. entry_count を更新
COMMIT
```

---

## 6.4 Langfuse 連携

### 同期方式（推奨）
```
Langfuse webhook を受信:
  1. trace_id → Article.external_key で検索
  2. 存在しない場合は Article + ArticleVersion を作成
  3. feedback → RelevanceAssessment に変換
  4. user_id → HumanJudge にマッピング（未登録なら作成）
  5. Assessment を作成
```

**冪等性**:
- 同一 trace_id + feedback の重複受信を防ぐため、external_id ベースの重複チェック

---

# 7. Audit & Timestamp Strategy

## 7.1 タイムスタンプ

### 全テーブル共通
- `created_at`: 初回作成時刻（NOT NULL、自動セット）
- `updated_at`: 最終更新時刻（NOT NULL、更新時に自動更新）

### Immutable テーブル（更新されない）
以下は `updated_at` を持たない：
- `ArticleVersion`
- `TopicVersion`
- `RelevanceAssessment`
- `DatasetEntry`

---

## 7.2 監査ログ

### 変更履歴の記録

重要な操作は別途 `audit_log` テーブルに記録：

```sql
CREATE TABLE audit_log (
  log_id UUID PRIMARY KEY,
  entity_type VARCHAR(50) NOT NULL, -- 'Article', 'Topic', 'Judge', etc.
  entity_id UUID NOT NULL,
  action VARCHAR(20) NOT NULL, -- 'CREATE', 'UPDATE', 'DELETE', 'STATUS_CHANGE'
  actor_type VARCHAR(20) NOT NULL, -- 'user', 'system', 'api'
  actor_id VARCHAR(100), -- user_id or api_key_id
  changes_json JSONB, -- before/after の diff
  created_at TIMESTAMP NOT NULL
);
```

**記録対象**:
- Article/Topic の status 変更
- Judge の作成・無効化
- Dataset の frozen 化
- 管理者による RelevanceAssessment の削除

---

## 7.3 論理削除 vs 物理削除

### 論理削除（Soft Delete）
以下は `status` フィールドで論理削除：
- Article (`status=deleted`)
- Topic (`status=archived`)
- Judge (`status=inactive`)
- Dataset (`status=archived`)

### 物理削除なし
- RelevanceAssessment: 削除不可（監査のため完全保持）
- DatasetEntry: Dataset が `frozen` の場合のみ削除不可

---

# 8. Non-Functional Requirements

## 8.1 パフォーマンス

### レスポンスタイム
- **判定API（LLM）**: 目標 5秒以内、最大 30秒
- **判定API（Rule）**: 目標 500ms以内
- **Dataset 作成**: 10万エントリで 30秒以内
- **CSV インポート**: 1万行で 5分以内

### スループット
- **同時判定リクエスト**: 最大 100 req/sec（LLM のレート制限に依存）
- **Langfuse webhook**: 最大 1000 events/sec

---

## 8.2 スケーラビリティ

### データ規模見積もり
- **Article**: 1000万件（3年運用）
- **ArticleVersion**: 1200万件（平均1.2バージョン/Article）
- **RelevanceAssessment**: 5000万件（1 Article × 5 Judge × 平均）
- **DatasetEntry**: 1億件（複数Dataset × 平均10万エントリ）

### インデックス戦略
```sql
-- 高頻度クエリ用インデックス
CREATE INDEX idx_article_external ON Article(source_type, external_key);
CREATE INDEX idx_article_version_hash ON ArticleVersion(content_hash);
CREATE INDEX idx_assessment_latest ON RelevanceAssessment(judge_id, topic_version_id, article_version_id, created_at DESC);
CREATE INDEX idx_dataset_entry_article ON DatasetEntry(article_version_id);
```

---

## 8.3 可用性

- **稼働率**: 99.5%（年間ダウンタイム 43.8時間以内）
- **バックアップ**: 日次フルバックアップ + 継続的WALアーカイブ
- **リカバリ**: RPO 1時間、RTO 4時間

---

## 8.4 セキュリティ

### 認証・認可
- **API**: API Key または OAuth 2.0
- **Judge 識別**: HumanJudge.user_id を SSO と統合
- **Role-based Access**: annotator / reviewer / admin

### データ保護
- **PII のマスキング**: context_json 内の個人情報は暗号化またはハッシュ化
- **監査ログ**: 全ての重要操作を記録（GDPR 対応）

---

## 8.5 保守性

### データ保持期間
- **RelevanceAssessment**: 無期限保持（ML学習用）
- **ArticleVersion**: 無期限保持
- **audit_log**: 3年保持

### マイグレーション戦略
- **スキーマ変更**: Alembic / Flyway による管理
- **後方互換性**: API バージョニング（`/v1/relevance/judge`）

---

# 9. API Specifications

## 9.1 判定API

### POST /v1/relevance/judge

**リクエスト**:
```json
{
  "topic_id": "uuid",
  "article": {
    "text": "本文",
    "context": {
      "source": "twitter",
      "tweet_id": "123456"
    },
    "language": "ja"
  },
  "judge": {
    "type": "automated",
    "impl_key": "llm:openai:gpt-4-turbo-2024-04-09"
  },
  "options": {
    "persist_article": true,
    "return_rationale": true
  }
}
```

**レスポンス（成功）**:
```json
{
  "assessment_id": "uuid",
  "article_version_id": "uuid",
  "topic_version_id": "uuid",
  "judge_id": "uuid",
  "label": "relevant",
  "confidence": 0.92,
  "rationale": "この文章は AI の誤情報に関する具体的な事例を含んでいる...",
  "created_at": "2025-01-15T12:34:56Z"
}
```

**レスポンス（エラー）**:
```json
{
  "error": {
    "code": "JUDGE_NOT_FOUND",
    "message": "指定された Judge が見つかりません",
    "details": {
      "impl_key": "llm:openai:gpt-5"
    }
  }
}
```

---

### GET /v1/assessments

複数の判定結果を取得。

**クエリパラメータ**:
```
?article_version_id=uuid
&topic_id=uuid
&judge_id=uuid
&label=relevant,not_relevant
&from=2025-01-01
&to=2025-01-31
&limit=100
&offset=0
```

**レスポンス**:
```json
{
  "total": 523,
  "limit": 100,
  "offset": 0,
  "assessments": [
    {
      "assessment_id": "uuid",
      "judge": {
        "judge_id": "uuid",
        "display_name": "GPT-4 Turbo",
        "kind": "automated"
      },
      "topic": {
        "topic_id": "uuid",
        "name": "AI Disinformation"
      },
      "article_version_id": "uuid",
      "label": "relevant",
      "confidence": 0.92,
      "created_at": "2025-01-15T12:34:56Z"
    }
  ]
}
```

---

## 9.2 Dataset API

### POST /v1/datasets

**リクエスト（Base Dataset）**:
```json
{
  "name": "community-notes-2025-q1",
  "purpose": "corpus",
  "spec": {
    "type": "time_slice",
    "from": "2025-01-01T00:00:00Z",
    "to": "2025-03-31T23:59:59Z",
    "source_types": ["twitter_community_note"]
  }
}
```

**リクエスト（Derived Dataset）**:
```json
{
  "name": "ai-disinfo-labeled-v1",
  "purpose": "training",
  "spec": {
    "type": "subset",
    "parent_dataset_version_id": "uuid",
    "filters": {
      "topic_version_id": "uuid",
      "labels": ["relevant", "not_relevant"],
      "min_confidence": 0.8
    },
    "sample": {
      "method": "stratified",
      "size": 10000
    }
  }
}
```

**レスポンス**:
```json
{
  "dataset_id": "uuid",
  "dataset_version_id": "uuid",
  "version_number": 1,
  "entry_count": 15234,
  "created_at": "2025-01-15T12:00:00Z"
}
```

---

### POST /v1/datasets/{dataset_id}/freeze

Dataset を固定化（学習・評価用）。

**レスポンス**:
```json
{
  "dataset_id": "uuid",
  "status": "frozen",
  "frozen_at": "2025-01-15T12:00:00Z"
}
```

---

## 9.3 CSV インポート API

### POST /v1/import/csv

**リクエスト**（multipart/form-data）:
```
file: <CSV file>
mapping: {
  "text_column": "note_text",
  "label_column": "relevance",
  "annotator_column": "reviewer_id",
  "topic_name": "AI Disinformation"
}
```

**レスポンス**:
```json
{
  "import_id": "uuid",
  "status": "processing",
  "total_rows": 5000,
  "processed_rows": 0,
  "errors": []
}
```

### GET /v1/import/{import_id}

インポート状況を確認。

**レスポンス**:
```json
{
  "import_id": "uuid",
  "status": "completed",
  "total_rows": 5000,
  "processed_rows": 5000,
  "created_articles": 4523,
  "created_assessments": 5000,
  "errors": [
    {
      "row": 42,
      "error": "Invalid label value: 'maybe'"
    }
  ]
}
```

---

# 10. Entity Relationship Diagram

```
┌─────────────────┐         ┌──────────────────┐
│     Article     │1      n │ ArticleVersion   │
│─────────────────│◄────────┤──────────────────│
│ article_id (PK) │         │ article_ver_id(PK)│
│ source_type     │         │ article_id (FK)  │
│ external_key    │         │ text             │
│ latest_ver_id(FK)│        │ context_json     │
│ status          │         │ content_hash (UQ)│
│ created_at      │         │ created_at       │
│ updated_at      │         └──────────────────┘
└─────────────────┘                  │
                                     │n
                                     │
                         ┌───────────▼──────────┐
                         │   DatasetEntry       │
                         │──────────────────────│
                         │ dataset_ver_id (FK)  │
                         │ article_ver_id (FK)  │
                         │ added_at             │
                         └───────────┬──────────┘
                                     │n
                                     │
                         ┌───────────▼──────────┐
                         │  DatasetVersion      │
                         │──────────────────────│1
                         │ dataset_ver_id (PK)  │◄────┐
                         │ dataset_id (FK)      │     │
                         │ version_number       │     │n (self-ref)
                         │ kind                 │     │parent
                         │ spec_json            │     │
                         │ parent_ver_id (FK)   ├─────┘
                         │ entry_count          │
                         │ frozen_at            │
                         └───────────┬──────────┘
                                     │n
                                     │
                         ┌───────────▼──────────┐
                         │      Dataset         │
                         │──────────────────────│
                         │ dataset_id (PK)      │
                         │ name (UQ)            │
                         │ purpose              │
                         │ status               │
                         └──────────────────────┘


┌─────────────────┐         ┌──────────────────┐
│      Topic      │1      n │  TopicVersion    │
│─────────────────│◄────────┤──────────────────│
│ topic_id (PK)   │         │ topic_ver_id (PK)│
│ name (UQ)       │         │ topic_id (FK)    │
│ status          │         │ definition       │
│ latest_ver_id(FK)│        │ definition_hash  │
│ created_at      │         │ created_at       │
│ updated_at      │         └──────────────────┘
└─────────────────┘


┌─────────────────────────────────────────────────────┐
│                      Judge                          │
│─────────────────────────────────────────────────────│
│ judge_id (PK)                                       │
│ kind (human | automated)                            │
│ display_name                                        │
│ status                                              │
│ metadata_json                                       │
└────────────┬────────────────────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
┌─────▼──────┐ ┌───▼──────────────┐
│ HumanJudge │ │ AutomatedJudge   │
│────────────│ │──────────────────│
│judge_id(FK)│ │ judge_id (FK)    │
│ user_id(UQ)│ │ automation_type  │
│ role       │ │ impl_key         │
└────────────┘ │ version          │
               │ config_json      │
               └──────────────────┘


┌──────────────────────────────────────────┐
│       RelevanceAssessment                │
│──────────────────────────────────────────│
│ assessment_id (PK)                       │
│ judge_id (FK) ───────┐                   │
│ topic_version_id (FK)─┼──┐               │
│ article_version_id(FK)┼┐ │               │
│ label                 ││ │               │
│ confidence            ││ │               │
│ rationale             ││ │               │
│ supersedes_id (FK)    ││ │               │
│ version               ││ │               │
│ created_at            ││ │               │
└───────────────────────┼┼─┼───────────────┘
                        │└─┼───► TopicVersion
                        │  └───► ArticleVersion
                        └──────► Judge
```

---

# 11. 今後の拡張余地

* **RuleJudge の本格実装**
  - キーワードマッチング
  - 正規表現ベース判定
  - Embedding 類似度ベース判定

* **LLM の self-training pipeline**
  - 人間判定結果を学習データとして自動再訓練
  - ModelVersion テーブルで管理（別 Bounded Context）

* **Active Learning**
  - 不確実性ベースのサンプリング
  - Human-in-the-loop による効率的なアノテーション

* **DatasetVersion の高度な運用**
  - lock / freeze の自動化
  - バージョン間の差分表示
  - 派生データセットの依存関係可視化

* **Langfuse の deeper integration**
  - trace → ArticleVersion 自動生成
  - Langfuse の prompt version と TopicVersion の同期

* **評価メトリクス自動計算**
  - Judge 間の一致率（Cohen's Kappa）
  - AutomatedJudge vs HumanJudge のF1スコア
  - Dataset ごとの品質指標ダッシュボード

---

# Appendix A: Implementation Checklist

実装時の確認事項：

- [ ] 全テーブルに `created_at`、必要に応じて `updated_at` を追加
- [ ] UNIQUE 制約をすべて設定
- [ ] 外部キー制約をすべて設定（CASCADE / RESTRICT を適切に選択）
- [ ] インデックスを高頻度クエリに対して設定
- [ ] JSONB フィールドのバリデーション関数を実装
- [ ] ステータス遷移のビジネスロジックをアプリケーション層で実装
- [ ] トランザクション境界を明確にしたサービス層を実装
- [ ] audit_log への記録ロジックを実装（トリガーまたはアプリ層）
- [ ] API のエラーハンドリングとバリデーションを実装
- [ ] CSV インポートの非同期処理（Celery / BullMQ など）を実装
- [ ] Langfuse webhook のエンドポイントと冪等性を実装
- [ ] Dataset の frozen 化後の変更を防ぐチェックを実装
- [ ] RelevanceAssessment の再判定時のバージョニングロジックを実装
- [ ] パフォーマンステストで目標値を達成

---

# Appendix B: FAQ

### Q1: 同じ Article を複数の Topic で判定できるか？
**A**: はい。RelevanceAssessment は `topic_version_id` を持つため、1つの ArticleVersion を複数の Topic で判定可能。

### Q2: Judge を削除できるか？
**A**: いいえ。Judge は `status=inactive` で無効化するのみ。過去の判定結果との整合性を保つため、物理削除しない。

### Q3: Dataset の entry_count はリアルタイムに更新されるか？
**A**: `status=active` の場合はリアルタイム更新。`status=frozen` の場合は変更されない。

### Q4: ArticleVersion の content_hash 衝突時はどうなるか？
**A**: SHA-256 の衝突確率は事実上ゼロだが、衝突時は新しい ArticleVersion を作成せず、既存のものを再利用する。

### Q5: Langfuse の trace が更新された場合は？
**A**: 新しい ArticleVersion を作成し、Article.latest_version_id を更新。旧バージョンは保持。

### Q6: Dataset の派生関係はどこまで深くできるか？
**A**: 技術的には無制限だが、運用上は3階層程度を推奨（base → filtered → sampled）。
