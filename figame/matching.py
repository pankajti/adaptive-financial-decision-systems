import heapq
import time
import random
import copy


# ==========================================
# PHASE 1: The Structural Physics (Exchange)
# ==========================================

class Order:
    def __init__(self, agent_id, side, price, qty):
        self.agent_id = agent_id
        self.side = side
        self.price = float(price)
        self.qty = int(qty)
        self.timestamp = time.time()


class MatchingEngine:
    def __init__(self, initial_price=100.0):
        self.bids = []
        self.asks = []
        self.trade_history = []
        # The universal anchor price for the market
        self.last_traded_price = initial_price

    def submit_order(self, order):
        if order.side == 'BUY':
            heapq.heappush(self.bids, (-order.price, order.timestamp, order))
        else:
            heapq.heappush(self.asks, (order.price, order.timestamp, order))
        self._match_orders()

    def _match_orders(self):
        while self.bids and self.asks:
            highest_bid_price = -self.bids[0][0]
            lowest_ask_price = self.asks[0][0]

            if highest_bid_price >= lowest_ask_price:
                best_bid = self.bids[0][2]
                best_ask = self.asks[0][2]

                # Trade executes at the resting order's price (Time Priority)
                trade_price = best_bid.price if best_bid.timestamp < best_ask.timestamp else best_ask.price
                trade_qty = min(best_bid.qty, best_ask.qty)

                # ==========================================
                # THE CRITICAL UPDATE: Market Price Shifts
                # ==========================================
                self.last_traded_price = trade_price

                self.trade_history.append({
                    'buyer': best_bid.agent_id,
                    'seller': best_ask.agent_id,
                    'price': trade_price,
                    'qty': trade_qty
                })
                print(
                    f"TRADE: {trade_qty} shares @ ${trade_price:.2f} | Ticker updated to: ${self.last_traded_price:.2f}")

                # Deduct filled quantities
                best_bid.qty -= trade_qty
                best_ask.qty -= trade_qty

                if best_bid.qty == 0: heapq.heappop(self.bids)
                if best_ask.qty == 0: heapq.heappop(self.asks)
            else:
                break

# ==========================================
# PHASE 2: Biological Evolution (Agents)
# ==========================================

class MarketAwareAgent:
    def __init__(self, agent_id, exchange, buy_margin, sell_margin):
        self.agent_id = agent_id
        self.exchange = exchange
        self.cash = 10000.0
        self.inventory = 100

        # Genetic parameters
        self.buy_margin = buy_margin
        self.sell_margin = sell_margin

    def act(self):
        # 1. Agent queries the exchange for the absolute latest price
        current_ticker_price = self.exchange.last_traded_price

        # 2. Agent formulates a subjective opinion (heterogeneous belief)
        # They believe the true value is somewhere within +/- 5% of the ticker
        perceived_value = current_ticker_price * random.uniform(0.95, 1.05)

        side = random.choice(['BUY', 'SELL'])
        qty = random.randint(1, 5)

        # 3. Agent applies their genetic traits to their subjective valuation
        if side == 'BUY' and self.cash > 0:
            bid_price = perceived_value * (1.0 - self.buy_margin)
            order = Order(self.agent_id, side, round(bid_price, 2), qty)
            self.exchange.submit_order(order)

        elif side == 'SELL' and self.inventory >= qty:
            ask_price = perceived_value * (1.0 + self.sell_margin)
            order = Order(self.agent_id, side, round(ask_price, 2), qty)
            self.exchange.submit_order(order)

    def get_fitness(self):
        # FIX: Added the fitness calculation so the generation loop can rank agents
        return self.cash + (self.inventory * self.exchange.last_traded_price)

# ==========================================
# PHASE 3: The Natural Selection Loop
# ==========================================

def breed(parent1, parent2, new_id, exchange):
    # Crossover: Average the parents' genes
    child_buy_margin = (parent1.buy_margin + parent2.buy_margin) / 2.0
    child_sell_margin = (parent1.sell_margin + parent2.sell_margin) / 2.0

    # Mutation: 10% chance to heavily mutate a gene
    if random.random() < 0.10:
        child_buy_margin += random.uniform(-0.02, 0.02)
        child_sell_margin += random.uniform(-0.02, 0.02)

    # Ensure margins stay logical (greater than 0)
    child_buy_margin = max(0.001, child_buy_margin)
    child_sell_margin = max(0.001, child_sell_margin)

    # FIX: Ensure we are instantiating the correct class
    return MarketAwareAgent(new_id, exchange, child_buy_margin, child_sell_margin)


if __name__ == "__main__":
    exchange = MatchingEngine()

    GENERATIONS = 25
    CYCLES_PER_GEN = 100
    survivor_count=5
    NUM_AGENTS=10

    # Generate Primordial Soup (10 agents with random genes)
    agents = []
    for i in range(NUM_AGENTS):
        b_margin = random.uniform(0.01, 0.10)
        s_margin = random.uniform(0.01, 0.10)
        agents.append(MarketAwareAgent(f"Agent_{i}", exchange, b_margin, s_margin))


    print("--- Starting Evolutionary Market Simulation ---\n")

    for gen in range(GENERATIONS):
        print(f"--- GENERATION {gen + 1} ---")

        # 1. Market Phase: Agents trade
        for _ in range(CYCLES_PER_GEN):
            active_agent = random.choice(agents)
            active_agent.act()

        # 2. Evaluation Phase: Calculate Fitness
        agents.sort(key=lambda a: a.get_fitness(), reverse=True)

        print(
            f"Top Survivor Fitness: ${agents[0].get_fitness():.2f} | Genes: [Buy Margin: {agents[0].buy_margin:.3f}, Sell Margin: {agents[0].sell_margin:.3f}]")
        print(
            f"Bottom Loser Fitness: ${agents[-1].get_fitness():.2f} | Genes: [Buy Margin: {agents[-1].buy_margin:.3f}, Sell Margin: {agents[-1].sell_margin:.3f}]")

        # 3. Selection & Reproduction Phase (Top 50% live, Bottom 50% die)
        survivors = agents[:survivor_count]
        next_generation = copy.deepcopy(survivors)  # Survivors move to next gen

        # Repopulate the dead 50% by breeding survivors
        for i in range(survivor_count):
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            child = breed(p1, p2, f"Gen{gen + 1}_Child_{i}", exchange)
            next_generation.append(child)

        agents = next_generation
        print(f"Current Market Price: ${exchange.last_traded_price:.2f}\n")