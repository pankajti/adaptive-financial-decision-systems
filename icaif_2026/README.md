# Research Ideas 

Research ideas for https://icaif2026.org/

## Narrative-Aware Trading System

### Overview
This project extends narrative-driven financial modeling from macro policy signals to 
real-time trading systems.

### Motivation
- ECAI-25: Policy → Treasury prediction
- ICAIF-26: News → Intraday trading decisions

### Architecture
- Narrative Signal Engine
- ORB Trading System
- Event-driven execution (IBKR)

### Research Questions
- Does narrative improve intraday trading?
- When should trading systems ignore news?

### Status
🚧 Work in progress (ICAIF 2026 submission)

## Adaptive ORB Candidate Selection

### Overview 

Can daily OHLCV + instrument metadata improve the selection of stocks likely to 
produce profitable ORB trades the next day?

Core contribution:

Daily price/volume data
+ sector / industry metadata
→ candidate ranking model
→ ORB backtest
→ compare against naive selection

Baselines to compare:

1. Top volume stocks
2. Top ATR stocks
3. Random sector-balanced selection
4. Your adaptive ranking model

Good target label:

profitable_orb_next_day = 1 if ORB trade on next day has positive return

Useful features:

- 1d / 5d / 20d return
- 5d / 20d volatility
- ATR
- volume spike ratio
- gap %
- close position in daily range
- sector
- industry
- country

Paper-style title:

Adaptive Candidate Selection for Event-Driven Intraday Breakout Systems Using Daily Market Features

### Motivation 

My current work on ORB and MCI based trading systems 

### data selection caveat 

using current list of snp500 stocks. might result in potential survivorship bias. 

