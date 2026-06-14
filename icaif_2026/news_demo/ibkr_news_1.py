from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 4001, clientId=123)  # 4002 = paper, 4001 = live


providers = ib.reqNewsProviders()
for p in providers:
    print(p.code, p.name)


def get_historical_news(conId, provider_codes, start, end):
    """
    conId: contract ID (e.g., AAPL)
    provider_codes: list like ['BRFG', 'DJNL']
    start/end: 'YYYY-MM-DD HH:MM:SS'
    """
    headlines = ib.reqHistoricalNews(
        conId=conId,
        providerCodes=','.join(provider_codes),
        startDateTime=start,
        endDateTime=end,
        totalResults=100
    )

    results = []
    for h in headlines:
        results.append({
            "time": h.time,
            "provider": h.providerCode,
            "articleId": h.articleId,
            "headline": h.headline
        })

    return results


contract = Stock('MRVL', 'SMART', 'USD')
ib.qualifyContracts(contract)

news = get_historical_news(
    conId=contract.conId,
    provider_codes=['BRFG'],
    start='20240101 00:00:00',
    end='20240110 23:59:59'
)

for n in news:
    print(n)