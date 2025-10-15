
from flask import Flask, render_template_string
from datetime import datetime, timedelta
from pymongo import MongoClient


app = Flask(__name__)
developer_ip = '192.168.2.163'

client = MongoClient(f"mongodb://admin:tP_kc8mn@{developer_ip}:27017/?authSource=admin")
Health_collection = client.health.status

def humanize_time_diff(delta: timedelta) -> str:
    s = int(delta.total_seconds())
    if s < 10:
        return "just now"
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        m, r = divmod(s, 60)
        return f"{m}m{f' {r}s' if r else ''} ago"
    h, r = divmod(s, 3600)
    m = r // 60
    return f"{h}h{f' {m}m' if m else ''} ago"

@app.route("/")
def index():
    now = datetime.now()
    processed = []
    for entry in Health_collection.find({}):
        last = datetime.strptime(entry["last_heartbeat"], "%Y-%m-%d %H:%M:%S")
        delta = now - last
        s = int(delta.total_seconds())

        if s < 60:
            status = "alive"; label = "Alive"; severity = 0
        elif s < 120:
            status = "warn"; label = "Needs Attention"; severity = 1
        else:
            status = "dead"; label = "DEAD"; severity = 2

        processed.append({
            "feed_name": entry["feed_name"],
            "ip": entry["worker_ip"],
            "last_heartbeat": entry["last_heartbeat"],
            "ago": humanize_time_diff(delta),
            "status": status,
            "label": label,
            "severity": severity,
        })

    grouped = {}
    for row in processed:
        grouped.setdefault(row["feed_name"], []).append(row)
    for k in grouped:
        grouped[k] = sorted(grouped[k], key=lambda r: (r["severity"]))

    updated_at = now.strftime("%Y-%m-%d %H:%M:%S")

    html = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Feed Status</title>
<style>
  body {
    margin:0;
    background:#111;
    color:#eee;
    font-family: monospace;
    font-size:14px;
  }
  .wrap { max-width:900px; margin:20px auto; padding:0 10px; }
  h1 { font-size:16px; margin-bottom:10px; }
  table { width:100%; border-collapse:collapse; }
  th,td { text-align:left; padding:4px 6px; }
  th { font-weight:600; }
  .status.alive { color:#0f0; }
  .status.warn { color:#ff0; }
  .status.dead { color:#f00; }
  .feed { margin-bottom:16px; }
  .feed-title { font-weight:600; margin-bottom:4px;font-size: 1.25rem; }
</style>
</head>
<body>
<div class="wrap">
<h1>Feed Status</h1>
<div>Updated {{ updated_at }}</div>

{% for feed, rows in grouped.items() %}
  <div class="feed">
    <div class="feed-title">{{ feed }}</div>
    <table>
      <thead>
        <tr><th>Worker IP</th><th>Last</th><th>When</th><th>Status</th></tr>
      </thead>
      <tbody>
        {% for r in rows %}
        <tr>
          <td>{{ r.ip }}</td>
          <td>{{ r.last_heartbeat }}</td>
          <td>{{ r.ago }}</td>
          <td class="status {{ r.status }}">{{ r.label }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
{% endfor %}

</div>
</body>
</html>
    """
    return render_template_string(html, grouped=grouped, updated_at=updated_at)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port="5800")