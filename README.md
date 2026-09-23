# College football player props

Prices six player prop markets for every FBS-vs-FBS game each week: anytime TD, rushing TDs,
passing TDs, rushing yards, receiving yards, and passing yards. Every market is trained on one
season and tested blind on the next.

Data: cfbfastR play-by-play and ESPN rosters from the sportsdataverse releases, plus the week's
spreads and totals from ESPN's public scoreboard feed. No API keys needed.

## Setup (same as the NFL site)

1. Create a new public GitHub repository and upload everything in this folder, including `.github`.
2. Settings → Pages → Source: **GitHub Actions**.
3. Actions tab → **Rebuild CFB Props** → **Run workflow**. The first run takes about 10–15 minutes.

It rebuilds every morning at 9 AM Eastern, every two hours on Saturdays, and on Thursday and Friday
afternoons. When the upcoming week's lines aren't posted yet, the site shows the most recent completed
week as a preview.

## Run it locally

```
pip install -r requirements.txt
python cfb_build.py
```
Then open `site/index.html`.
