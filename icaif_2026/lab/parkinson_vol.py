import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    symbol = 'NVDA'
    yf_ticker = yf.Ticker(symbol)
    market_history = yf_ticker.history(period="1y")
    num_days = 252
    T = 1
    high = market_history.High
    low = market_history.Low
    close = market_history.Close

    sigma_p = np.sqrt((num_days/(4*len(high)*np.log(2))) * np.sum((np.log(high/low) )**2))
    sigma_p_daily = sigma_p/(num_days**.5)

    S0 = close.iloc[-1] # initial price
    mu = 0 # $\mu$
    tp = S0*(1+0.1) # take profit
    sl = S0*(1-0.05) # stop loss
    th=30 # time horizon
    n_sims=10000


    Z= np.random.randn(th, n_sims)

    daily_returns = np.exp( (mu+0.5*sigma_p_daily**2)*T - sigma_p_daily*np.sqrt(T)*Z)
    prices = np.zeros([th+1, n_sims])
    prices[0]=S0

    for i in range(1, th+1):
        prices[i] = prices[i-1]*daily_returns[i-1]

    hit_tp = prices> tp
    hit_sl = prices<sl

    tp_days = np.argmax(hit_tp,axis =0)
    sl_days = np.argmax(hit_sl,axis =0)

    tp_days = np.where(hit_tp.any(axis=0), tp_days,999)
    sl_days = np.where(hit_sl.any(axis=0), sl_days,999)

    wins = np.sum(tp_days<sl_days)
    losses = np.sum(sl_days<tp_days)
    holds = np.sum((tp_days==999) &(sl_days==999))
    ties = np.sum((tp_days == sl_days) & (tp_days != 999))
    losses += ties

    total = wins + losses + holds

    print(f"--- 30-Day Trade Simulation for {symbol} ---")
    print(f"Initial Price: ${S0:.2f} | TP: ${tp:.2f} | SL: ${sl:.2f}")
    print(f"Parkinson Volatility (Annual): {sigma_p:.2%}")
    print("-" * 40)
    print(f"Win Probability:  {wins/total:.2%}")
    print(f"Loss Probability: {losses/total:.2%}")
    print(f"Hold Probability: {holds/total:.2%}")

    # ==========================================
    # 7. Visualization f
    # ==========================================
    # Create a side-by-side figure (Paths vs. Outcomes)
    visualize_results(S0, holds, losses, n_sims, prices, sl, symbol, total, tp, wins)


def visualize_results(S0, holds, losses, n_sims, prices, sl, symbol, total, tp, wins):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2.5, 1]})
    # --- Plot 1: Monte Carlo Price Paths (Subset for visual clarity) ---
    sample_size = 200  # We only plot 200 out of 10,000 so the chart isn't a solid block of color
    ax1.plot(prices[:, :sample_size], alpha=0.15, color='royalblue')
    # Highlight the key threshold lines
    ax1.axhline(S0, color='black', linestyle='--', linewidth=1.5, label=f'Initial Price (${S0:.2f})')
    ax1.axhline(tp, color='green', linestyle='-', linewidth=2, label=f'Take Profit (${tp:.2f})')
    ax1.axhline(sl, color='red', linestyle='-', linewidth=2, label=f'Stop Loss (${sl:.2f})')
    ax1.set_title(f"Monte Carlo Simulation: {symbol} 30-Day Forecast\nShowing {sample_size} Sample Paths", fontsize=14,
                  fontweight='bold')
    ax1.set_xlabel("Trading Days from Today", fontsize=12)
    ax1.set_ylabel("Simulated Price ($)", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    # --- Plot 2: Bar Chart of Probabilities ---
    labels = ['Wins\n(Hit TP First)', 'Losses\n(Hit SL First)', 'Holds\n(Hit Neither)']
    counts = [wins, losses, holds]
    colors = ['mediumseagreen', 'tomato', 'lightslategray']
    bars = ax2.bar(labels, counts, color=colors, edgecolor='black', alpha=0.8)
    # Add percentage labels to the top of each bar
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height + (n_sims * 0.01),
                 f"{count / total:.1%}", ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax2.set_title("Probability Distribution\nof Trade Outcomes", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Number of Simulations", fontsize=12)
    ax2.set_ylim(0, n_sims * 1.1)  # Give room for the text labels above the bars
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()