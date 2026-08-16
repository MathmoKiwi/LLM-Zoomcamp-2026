# Monitoring dashboard

Grafana runs at http://localhost:3000 (admin / value of GRAFANA_ADMIN_PASSWORD).
The Postgres datasource is provisioned automatically as `TechnotePG`.

The rubric wants user feedback collection plus a dashboard with at least
five charts. Feedback collection is already wired (thumbs in the UI write
to the `feedback` table). Build the dashboard by creating a panel per query
below, then export the dashboard JSON into this folder and commit it so the
repo carries the finished dashboard.

TODO(dashboard): build the panels, screenshot the result for the README,
export JSON to monitoring/grafana/dashboard.json.

## Panel 1: questions over time (time series)

```sql
SELECT date_trunc('hour', ts) AS time, count(*) AS questions
FROM conversations
GROUP BY 1 ORDER BY 1;
```

## Panel 2: feedback ratio (gauge or pie)

```sql
SELECT
  count(*) FILTER (WHERE value = 1)  AS thumbs_up,
  count(*) FILTER (WHERE value = -1) AS thumbs_down
FROM feedback;
```

## Panel 3: p95 response time by retrieval mode (bar chart)

```sql
SELECT mode, percentile_cont(0.95) WITHIN GROUP (ORDER BY response_ms) AS p95_ms
FROM conversations
GROUP BY mode;
```

## Panel 4: abstention rate over time (time series)

The share of answers where the system said NOT_FOUND. A sudden change in
either direction usually means retrieval broke or the prompt regressed.

```sql
SELECT date_trunc('day', ts) AS time,
       avg(CASE WHEN abstained THEN 1.0 ELSE 0.0 END) AS abstention_rate
FROM conversations
GROUP BY 1 ORDER BY 1;
```

## Panel 5: token spend per day (time series)

```sql
SELECT date_trunc('day', ts) AS time,
       sum(coalesce(prompt_tokens, 0))     AS prompt_tokens,
       sum(coalesce(completion_tokens, 0)) AS completion_tokens
FROM conversations
GROUP BY 1 ORDER BY 1;
```

## Panel 6: most retrieved documents (table)

Useful for spotting one technote dominating results, which usually points
at a chunking or scoring problem.

```sql
SELECT src AS filename, count(*) AS times_retrieved
FROM conversations, jsonb_array_elements_text(sources) AS src
GROUP BY 1 ORDER BY 2 DESC LIMIT 15;
```
