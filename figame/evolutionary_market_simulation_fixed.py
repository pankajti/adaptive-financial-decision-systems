from __future__ import annotations

import argparse
import heapq
import itertools
import math
import random
import statistics
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


EPSILON = 1e-9


# ============================================================
# PHASE 1: Exchange mechanics and economic settlement
# ============================================================


@dataclass
class Order:
    order_id: int
    agent_id: str
    side: str
    price: float
    original_qty: int
    remaining_qty: int
    sequence: int
    created_step: int
    status: str = "OPEN"
    cancel_reason: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.status == "OPEN" and self.remaining_qty > 0


@dataclass
class Trade:
    trade_id: int
    step: int
    buyer_id: str
    seller_id: str
    price: float
    qty: int
    maker_id: str
    taker_id: str
    buy_order_id: int
    sell_order_id: int
    buyer_fee: float
    seller_fee: float


class MatchingEngine:
    """Price-time-priority limit-order-book matching engine.

    Important implementation choices:
    - Orders reserve cash or inventory before entering the book.
    - Trades settle cash, inventory, and fees immediately.
    - Self-trades are prevented by cancelling the newer crossing order.
    - Heap priority uses deterministic integer sequence numbers.
    - Cancelled/filled orders are removed lazily from the heaps.
    """

    VALID_SIDES = {"BUY", "SELL"}

    def __init__(
        self,
        initial_price: float = 100.0,
        fee_rate: float = 0.0001,
        max_order_age_steps: int = 50,
        invariant_checks: bool = True,
        verbose_trades: bool = False,
    ) -> None:
        if not math.isfinite(initial_price) or initial_price <= 0:
            raise ValueError("initial_price must be a finite positive number")
        if fee_rate < 0:
            raise ValueError("fee_rate cannot be negative")
        if max_order_age_steps <= 0:
            raise ValueError("max_order_age_steps must be positive")

        self.initial_price = float(initial_price)
        self.last_traded_price = float(initial_price)
        # Exogenous fundamental value prevents agents from improving fitness
        # merely by moving a tiny final trade. It evolves as a mild random walk.
        self.fundamental_price = float(initial_price)
        self.fee_rate = float(fee_rate)
        self.max_order_age_steps = int(max_order_age_steps)
        self.invariant_checks = invariant_checks
        self.verbose_trades = verbose_trades

        self.bids: List[Tuple[float, int, int]] = []
        self.asks: List[Tuple[float, int, int]] = []
        self.orders: Dict[int, Order] = {}
        self.agents: Dict[str, MarketAwareAgent] = {}
        self.trade_history: List[Trade] = []

        self._sequence_counter = itertools.count(1)
        self._order_id_counter = itertools.count(1)
        self._trade_id_counter = itertools.count(1)
        self.current_step = 0

        self.fee_pool = 0.0
        self._initial_total_cash: Optional[float] = None
        self._initial_total_inventory: Optional[int] = None

    # ---------------------------
    # Registration and invariants
    # ---------------------------

    def register_agent(self, agent: "MarketAwareAgent") -> None:
        if agent.agent_id in self.agents:
            raise ValueError(f"Duplicate agent_id: {agent.agent_id}")
        self.agents[agent.agent_id] = agent

    def snapshot_initial_totals(self) -> None:
        self._initial_total_cash = sum(a.cash for a in self.agents.values())
        self._initial_total_inventory = sum(a.inventory for a in self.agents.values())
        self._assert_invariants()

    def _assert_invariants(self) -> None:
        if not self.invariant_checks:
            return

        for agent in self.agents.values():
            if agent.cash < -1e-7:
                raise AssertionError(f"Negative cash for {agent.agent_id}: {agent.cash}")
            if agent.inventory < 0:
                raise AssertionError(
                    f"Negative inventory for {agent.agent_id}: {agent.inventory}"
                )
            if agent.reserved_cash < -1e-7:
                raise AssertionError(
                    f"Negative reserved cash for {agent.agent_id}: {agent.reserved_cash}"
                )
            if agent.reserved_inventory < 0:
                raise AssertionError(
                    f"Negative reserved inventory for {agent.agent_id}: "
                    f"{agent.reserved_inventory}"
                )
            if agent.reserved_cash > agent.cash + 1e-7:
                raise AssertionError(
                    f"Reserved cash exceeds cash for {agent.agent_id}: "
                    f"{agent.reserved_cash} > {agent.cash}"
                )
            if agent.reserved_inventory > agent.inventory:
                raise AssertionError(
                    f"Reserved inventory exceeds inventory for {agent.agent_id}: "
                    f"{agent.reserved_inventory} > {agent.inventory}"
                )

        if self._initial_total_cash is not None:
            total_cash_plus_fees = (
                sum(a.cash for a in self.agents.values()) + self.fee_pool
            )
            if not math.isclose(
                total_cash_plus_fees,
                self._initial_total_cash,
                rel_tol=1e-10,
                abs_tol=1e-6,
            ):
                raise AssertionError(
                    "Cash conservation failed: "
                    f"agents + fees={total_cash_plus_fees:.10f}, "
                    f"initial={self._initial_total_cash:.10f}"
                )

        if self._initial_total_inventory is not None:
            total_inventory = sum(a.inventory for a in self.agents.values())
            if total_inventory != self._initial_total_inventory:
                raise AssertionError(
                    "Inventory conservation failed: "
                    f"current={total_inventory}, "
                    f"initial={self._initial_total_inventory}"
                )

    # ---------------------------
    # Order-book helpers
    # ---------------------------

    def set_step(self, step: int) -> None:
        self.current_step = int(step)

    def _clean_heap(self, heap: List[Tuple[float, int, int]]) -> None:
        while heap:
            order_id = heap[0][2]
            order = self.orders[order_id]
            if order.is_open:
                break
            heapq.heappop(heap)

    def _best_bid_order(self) -> Optional[Order]:
        self._clean_heap(self.bids)
        if not self.bids:
            return None
        return self.orders[self.bids[0][2]]

    def _best_ask_order(self) -> Optional[Order]:
        self._clean_heap(self.asks)
        if not self.asks:
            return None
        return self.orders[self.asks[0][2]]

    @property
    def best_bid(self) -> Optional[float]:
        order = self._best_bid_order()
        return order.price if order else None

    @property
    def best_ask(self) -> Optional[float]:
        order = self._best_ask_order()
        return order.price if order else None

    def update_fundamental(
        self, rng: random.Random, volatility_per_round: float = 0.002
    ) -> float:
        """Advance an exogenous fundamental value by one simulation round."""
        shock = rng.normalvariate(0.0, volatility_per_round)
        self.fundamental_price *= math.exp(
            shock - 0.5 * volatility_per_round * volatility_per_round
        )
        # Defensive bounds prevent numerical runaway in unusually long runs.
        lower = 0.25 * self.initial_price
        upper = 4.00 * self.initial_price
        self.fundamental_price = clamp(self.fundamental_price, lower, upper)
        return self.fundamental_price

    def robust_market_price(self, recent_trade_count: int = 25) -> float:
        """Median/midpoint market estimate for diagnostics and agent beliefs."""
        prices = [t.price for t in self.trade_history[-recent_trade_count:]]
        robust_trade_price = (
            statistics.median(prices) if prices else self.last_traded_price
        )
        bid = self.best_bid
        ask = self.best_ask
        if bid is not None and ask is not None:
            midpoint = (bid + ask) / 2.0
            return (robust_trade_price + midpoint) / 2.0
        return robust_trade_price

    def reference_price(self) -> float:
        """Blend observable market information with exogenous fair value."""
        market_anchor = self.robust_market_price()
        return 0.65 * market_anchor + 0.35 * self.fundamental_price

    def mark_price(self) -> float:
        """Manipulation-resistant accounting mark used for fitness."""
        return self.fundamental_price

    def recent_return(self, lookback_trades: int = 5) -> float:
        if len(self.trade_history) < 2:
            return 0.0
        end_price = self.trade_history[-1].price
        start_index = max(0, len(self.trade_history) - 1 - lookback_trades)
        start_price = self.trade_history[start_index].price
        if start_price <= 0:
            return 0.0
        return (end_price / start_price) - 1.0

    # ---------------------------
    # Submission, cancellation, expiry
    # ---------------------------

    def submit_order(
        self,
        agent_id: str,
        side: str,
        price: float,
        qty: int,
        step: Optional[int] = None,
    ) -> Optional[int]:
        side = side.upper().strip()
        if side not in self.VALID_SIDES:
            raise ValueError(f"Invalid side {side!r}; expected BUY or SELL")
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent_id: {agent_id}")
        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
            raise ValueError("qty must be a positive integer")
        if not math.isfinite(price) or price <= 0:
            raise ValueError("price must be a finite positive number")

        rounded_price = round(float(price), 2)
        if rounded_price <= 0:
            raise ValueError("price rounds to a non-positive tick")

        agent = self.agents[agent_id]
        required_buy_cash = rounded_price * qty * (1.0 + self.fee_rate)

        if side == "BUY":
            if agent.available_cash + EPSILON < required_buy_cash:
                return None
            agent.reserved_cash += required_buy_cash
        else:
            if agent.available_inventory < qty:
                return None
            agent.reserved_inventory += qty

        sequence = next(self._sequence_counter)
        order_id = next(self._order_id_counter)
        order = Order(
            order_id=order_id,
            agent_id=agent_id,
            side=side,
            price=rounded_price,
            original_qty=qty,
            remaining_qty=qty,
            sequence=sequence,
            created_step=self.current_step if step is None else int(step),
        )
        self.orders[order_id] = order

        if side == "BUY":
            heapq.heappush(self.bids, (-rounded_price, sequence, order_id))
        else:
            heapq.heappush(self.asks, (rounded_price, sequence, order_id))

        self._match_orders()
        self._assert_invariants()
        return order_id

    def cancel_order(
        self,
        order_id: int,
        requester_agent_id: Optional[str] = None,
        reason: str = "USER_CANCELLED",
    ) -> bool:
        order = self.orders.get(order_id)
        if order is None or not order.is_open:
            return False
        if requester_agent_id is not None and order.agent_id != requester_agent_id:
            raise PermissionError("An agent may only cancel its own order")
        self._cancel_order_internal(order, reason)
        self._assert_invariants()
        return True

    def _cancel_order_internal(self, order: Order, reason: str) -> None:
        if not order.is_open:
            return
        agent = self.agents[order.agent_id]
        remaining_qty = order.remaining_qty

        if order.side == "BUY":
            release = order.price * remaining_qty * (1.0 + self.fee_rate)
            agent.reserved_cash = max(0.0, agent.reserved_cash - release)
        else:
            agent.reserved_inventory -= remaining_qty

        order.remaining_qty = 0
        order.status = "CANCELLED"
        order.cancel_reason = reason

    def expire_orders(self, current_step: Optional[int] = None) -> int:
        step = self.current_step if current_step is None else int(current_step)
        expired = 0
        for order in self.orders.values():
            if order.is_open and step - order.created_step >= self.max_order_age_steps:
                self._cancel_order_internal(order, "EXPIRED")
                expired += 1
        if expired:
            self._assert_invariants()
        return expired

    def cancel_all_orders(self, agent_id: Optional[str] = None) -> int:
        cancelled = 0
        for order in self.orders.values():
            if not order.is_open:
                continue
            if agent_id is not None and order.agent_id != agent_id:
                continue
            self._cancel_order_internal(order, "END_OF_EPISODE")
            cancelled += 1
        self._clean_heap(self.bids)
        self._clean_heap(self.asks)
        self._assert_invariants()
        return cancelled

    # ---------------------------
    # Matching and settlement
    # ---------------------------

    def _match_orders(self) -> None:
        while True:
            best_bid = self._best_bid_order()
            best_ask = self._best_ask_order()

            if best_bid is None or best_ask is None:
                return
            if best_bid.price + EPSILON < best_ask.price:
                return

            # Prevent wash trades. Cancelling the newer order preserves the
            # older resting quote and guarantees the matching loop advances.
            if best_bid.agent_id == best_ask.agent_id:
                newer = (
                    best_bid if best_bid.sequence > best_ask.sequence else best_ask
                )
                self._cancel_order_internal(newer, "SELF_TRADE_PREVENTION")
                continue

            trade_qty = min(best_bid.remaining_qty, best_ask.remaining_qty)
            resting_order = (
                best_bid if best_bid.sequence < best_ask.sequence else best_ask
            )
            incoming_order = best_ask if resting_order is best_bid else best_bid
            trade_price = resting_order.price

            buyer = self.agents[best_bid.agent_id]
            seller = self.agents[best_ask.agent_id]

            buyer_fee = trade_price * trade_qty * self.fee_rate
            seller_fee = trade_price * trade_qty * self.fee_rate
            buyer_total_cost = trade_price * trade_qty + buyer_fee
            seller_net_proceeds = trade_price * trade_qty - seller_fee

            # Release the reservation at the BUY limit price. Since execution
            # cannot exceed that limit, the buyer always has enough cash.
            buyer_reservation_release = (
                best_bid.price * trade_qty * (1.0 + self.fee_rate)
            )
            buyer.reserved_cash = max(
                0.0, buyer.reserved_cash - buyer_reservation_release
            )
            seller.reserved_inventory -= trade_qty

            buyer.cash -= buyer_total_cost
            buyer.inventory += trade_qty
            seller.cash += seller_net_proceeds
            seller.inventory -= trade_qty
            buyer.fees_paid += buyer_fee
            seller.fees_paid += seller_fee
            self.fee_pool += buyer_fee + seller_fee

            buyer.trade_count += 1
            seller.trade_count += 1
            buyer.traded_volume += trade_qty
            seller.traded_volume += trade_qty

            best_bid.remaining_qty -= trade_qty
            best_ask.remaining_qty -= trade_qty

            if best_bid.remaining_qty == 0:
                best_bid.status = "FILLED"
            if best_ask.remaining_qty == 0:
                best_ask.status = "FILLED"

            self.last_traded_price = trade_price
            trade = Trade(
                trade_id=next(self._trade_id_counter),
                step=self.current_step,
                buyer_id=buyer.agent_id,
                seller_id=seller.agent_id,
                price=trade_price,
                qty=trade_qty,
                maker_id=resting_order.agent_id,
                taker_id=incoming_order.agent_id,
                buy_order_id=best_bid.order_id,
                sell_order_id=best_ask.order_id,
                buyer_fee=buyer_fee,
                seller_fee=seller_fee,
            )
            self.trade_history.append(trade)

            if self.verbose_trades:
                print(
                    f"TRADE {trade.trade_id}: {trade_qty} @ {trade_price:.2f} | "
                    f"buyer={buyer.agent_id} seller={seller.agent_id} | "
                    f"maker={trade.maker_id}"
                )

            self._assert_invariants()


# ============================================================
# PHASE 2: Agent genomes and market behaviour
# ============================================================


@dataclass(frozen=True)
class Genome:
    genome_id: str
    buy_margin: float
    sell_margin: float
    valuation_noise: float
    inventory_aversion: float
    buy_probability: float
    max_order_size: int


@dataclass
class MarketAwareAgent:
    agent_id: str
    genome: Genome
    exchange: MatchingEngine
    initial_cash: float = 10_000.0
    initial_inventory: int = 100

    cash: float = field(init=False)
    inventory: int = field(init=False)
    reserved_cash: float = field(default=0.0, init=False)
    reserved_inventory: int = field(default=0, init=False)
    fees_paid: float = field(default=0.0, init=False)
    trade_count: int = field(default=0, init=False)
    traded_volume: int = field(default=0, init=False)
    wealth_history: List[float] = field(default_factory=list, init=False)
    relative_equity_history: List[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_cash)
        self.inventory = int(self.initial_inventory)
        self.exchange.register_agent(self)
        initial_wealth = self.cash + self.inventory * self.exchange.initial_price
        self.wealth_history.append(initial_wealth)
        self.relative_equity_history.append(initial_wealth)

    @property
    def available_cash(self) -> float:
        return max(0.0, self.cash - self.reserved_cash)

    @property
    def available_inventory(self) -> int:
        return max(0, self.inventory - self.reserved_inventory)

    def wealth(self, mark_price: float) -> float:
        return self.cash + self.inventory * mark_price

    def record_wealth(self, mark_price: float) -> None:
        actual_wealth = self.wealth(mark_price)
        initial_wealth = (
            self.initial_cash + self.initial_inventory * self.exchange.initial_price
        )
        passive_benchmark = self.initial_cash + self.initial_inventory * mark_price
        # Relative equity starts at initial wealth and tracks value added over
        # simply retaining the initial cash and inventory endowment.
        relative_equity = initial_wealth + actual_wealth - passive_benchmark
        self.wealth_history.append(actual_wealth)
        self.relative_equity_history.append(relative_equity)

    def act(self, rng: random.Random, step: int) -> Optional[int]:
        current_price = self.exchange.reference_price()
        recent_return = self.exchange.recent_return()

        # Persistent agent traits now influence valuation, inventory control,
        # side choice, and order size. This gives evolution a stronger signal
        # than margins alone while preserving the original market-aware idea.
        noisy_value = current_price * math.exp(
            rng.normalvariate(0.0, self.genome.valuation_noise)
        )
        trend_adjustment = 1.0 + 0.20 * recent_return

        inventory_target = max(1, self.initial_inventory)
        inventory_gap = (self.inventory - inventory_target) / inventory_target
        reservation_value = noisy_value * trend_adjustment * (
            1.0 - self.genome.inventory_aversion * inventory_gap
        )
        reservation_value = max(0.01, reservation_value)

        adjusted_buy_probability = self.genome.buy_probability - 0.30 * inventory_gap
        adjusted_buy_probability = min(max(adjusted_buy_probability, 0.05), 0.95)
        side = "BUY" if rng.random() < adjusted_buy_probability else "SELL"

        requested_qty = rng.randint(1, self.genome.max_order_size)

        if side == "BUY":
            price = round(
                reservation_value * (1.0 - self.genome.buy_margin), 2
            )
            if price <= 0:
                return None
            max_affordable = int(
                self.available_cash
                // (price * (1.0 + self.exchange.fee_rate))
            )
            qty = min(requested_qty, max_affordable)
            if qty <= 0:
                return None
        else:
            price = round(
                reservation_value * (1.0 + self.genome.sell_margin), 2
            )
            qty = min(requested_qty, self.available_inventory)
            if qty <= 0:
                return None

        return self.exchange.submit_order(
            agent_id=self.agent_id,
            side=side,
            price=price,
            qty=qty,
            step=step,
        )


# ============================================================
# PHASE 3: Fair episodic evaluation and natural selection
# ============================================================


@dataclass
class EpisodeResult:
    score: float
    terminal_wealth: float
    absolute_return_pct: float
    return_pct: float
    max_drawdown: float
    inventory_deviation: float
    trade_count: int
    traded_volume: int
    fees_paid: float


@dataclass
class AggregateResult:
    genome: Genome
    mean_score: float
    mean_terminal_wealth: float
    mean_absolute_return_pct: float
    mean_return_pct: float
    mean_max_drawdown: float
    mean_inventory_deviation: float
    mean_trade_count: float
    mean_traded_volume: float
    mean_fees_paid: float


def max_drawdown(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


def run_episode(
    genomes: Sequence[Genome],
    seed: int,
    rounds: int,
    initial_price: float = 100.0,
    initial_cash: float = 10_000.0,
    initial_inventory: int = 100,
    fee_rate: float = 0.0001,
    max_order_age_steps: int = 50,
    verbose_trades: bool = False,
) -> Tuple[Dict[str, EpisodeResult], MatchingEngine]:
    rng = random.Random(seed)
    exchange = MatchingEngine(
        initial_price=initial_price,
        fee_rate=fee_rate,
        max_order_age_steps=max_order_age_steps,
        invariant_checks=True,
        verbose_trades=verbose_trades,
    )

    agents: List[MarketAwareAgent] = []
    for genome in genomes:
        agents.append(
            MarketAwareAgent(
                agent_id=genome.genome_id,
                genome=genome,
                exchange=exchange,
                initial_cash=initial_cash,
                initial_inventory=initial_inventory,
            )
        )

    exchange.snapshot_initial_totals()

    action_step = 0
    for _ in range(rounds):
        exchange.update_fundamental(rng)
        # Every agent receives exactly one action per round.
        action_order = agents.copy()
        rng.shuffle(action_order)

        for agent in action_order:
            action_step += 1
            exchange.set_step(action_step)
            exchange.expire_orders(action_step)
            agent.act(rng, action_step)

        mark = exchange.mark_price()
        for agent in agents:
            agent.record_wealth(mark)

    terminal_mark = exchange.mark_price()
    results: Dict[str, EpisodeResult] = {}

    for agent in agents:
        initial_wealth = agent.initial_cash + agent.initial_inventory * initial_price
        terminal_wealth = agent.wealth(terminal_mark)
        passive_benchmark = (
            agent.initial_cash + agent.initial_inventory * terminal_mark
        )
        absolute_return_pct = (terminal_wealth - initial_wealth) / initial_wealth
        # Fitness return is excess performance over passively retaining the
        # initial endowment, removing the common effect of fundamental moves.
        return_pct = (terminal_wealth - passive_benchmark) / initial_wealth
        drawdown = max_drawdown(agent.relative_equity_history)
        inventory_deviation = (
            abs(agent.inventory - agent.initial_inventory)
            / max(1, agent.initial_inventory)
        )

        # Risk-adjusted fitness. Fees are already reflected in cash. A small
        # inactivity penalty prevents permanently non-crossing genomes from
        # winning solely by avoiding all participation.
        inactivity_penalty = 0.0005 if agent.trade_count == 0 else 0.0
        score = (
            return_pct
            - 0.30 * drawdown
            - 0.01 * inventory_deviation
            - inactivity_penalty
        )

        results[agent.genome.genome_id] = EpisodeResult(
            score=score,
            terminal_wealth=terminal_wealth,
            absolute_return_pct=absolute_return_pct,
            return_pct=return_pct,
            max_drawdown=drawdown,
            inventory_deviation=inventory_deviation,
            trade_count=agent.trade_count,
            traded_volume=agent.traded_volume,
            fees_paid=agent.fees_paid,
        )

    # Release all outstanding reservations and verify conservation.
    exchange.cancel_all_orders()
    exchange._assert_invariants()
    return results, exchange


def evaluate_population(
    genomes: Sequence[Genome],
    episodes: int,
    rounds: int,
    base_seed: int,
    verbose_trades: bool = False,
) -> List[AggregateResult]:
    episode_results: Dict[str, List[EpisodeResult]] = {
        genome.genome_id: [] for genome in genomes
    }

    # Common random numbers: every generation is tested against the same seed
    # set, reducing noise when comparing successive populations.
    for episode in range(episodes):
        seed = base_seed + episode * 10_003
        results, _ = run_episode(
            genomes=genomes,
            seed=seed,
            rounds=rounds,
            verbose_trades=verbose_trades,
        )
        for genome_id, result in results.items():
            episode_results[genome_id].append(result)

    aggregates: List[AggregateResult] = []
    genome_by_id = {genome.genome_id: genome for genome in genomes}

    for genome_id, results in episode_results.items():
        aggregates.append(
            AggregateResult(
                genome=genome_by_id[genome_id],
                mean_score=statistics.fmean(r.score for r in results),
                mean_terminal_wealth=statistics.fmean(
                    r.terminal_wealth for r in results
                ),
                mean_absolute_return_pct=statistics.fmean(
                    r.absolute_return_pct for r in results
                ),
                mean_return_pct=statistics.fmean(r.return_pct for r in results),
                mean_max_drawdown=statistics.fmean(
                    r.max_drawdown for r in results
                ),
                mean_inventory_deviation=statistics.fmean(
                    r.inventory_deviation for r in results
                ),
                mean_trade_count=statistics.fmean(
                    r.trade_count for r in results
                ),
                mean_traded_volume=statistics.fmean(
                    r.traded_volume for r in results
                ),
                mean_fees_paid=statistics.fmean(r.fees_paid for r in results),
            )
        )

    aggregates.sort(key=lambda result: result.mean_score, reverse=True)
    return aggregates


def random_genome(genome_id: str, rng: random.Random) -> Genome:
    return Genome(
        genome_id=genome_id,
        buy_margin=rng.uniform(0.002, 0.050),
        sell_margin=rng.uniform(0.002, 0.050),
        valuation_noise=rng.uniform(0.005, 0.050),
        inventory_aversion=rng.uniform(0.000, 0.120),
        buy_probability=rng.uniform(0.35, 0.65),
        max_order_size=rng.randint(1, 6),
    )


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def breed(
    parent1: Genome,
    parent2: Genome,
    child_id: str,
    rng: random.Random,
    mutation_probability: float = 0.15,
) -> Genome:
    buy_margin = (parent1.buy_margin + parent2.buy_margin) / 2.0
    sell_margin = (parent1.sell_margin + parent2.sell_margin) / 2.0
    valuation_noise = (
        parent1.valuation_noise + parent2.valuation_noise
    ) / 2.0
    inventory_aversion = (
        parent1.inventory_aversion + parent2.inventory_aversion
    ) / 2.0
    buy_probability = (
        parent1.buy_probability + parent2.buy_probability
    ) / 2.0
    max_order_size = round(
        (parent1.max_order_size + parent2.max_order_size) / 2.0
    )

    if rng.random() < mutation_probability:
        buy_margin += rng.uniform(-0.010, 0.010)
    if rng.random() < mutation_probability:
        sell_margin += rng.uniform(-0.010, 0.010)
    if rng.random() < mutation_probability:
        valuation_noise += rng.uniform(-0.010, 0.010)
    if rng.random() < mutation_probability:
        inventory_aversion += rng.uniform(-0.025, 0.025)
    if rng.random() < mutation_probability:
        buy_probability += rng.uniform(-0.10, 0.10)
    if rng.random() < mutation_probability:
        max_order_size += rng.choice([-2, -1, 1, 2])

    return Genome(
        genome_id=child_id,
        buy_margin=clamp(buy_margin, 0.001, 0.100),
        sell_margin=clamp(sell_margin, 0.001, 0.100),
        valuation_noise=clamp(valuation_noise, 0.001, 0.100),
        inventory_aversion=clamp(inventory_aversion, 0.000, 0.250),
        buy_probability=clamp(buy_probability, 0.10, 0.90),
        max_order_size=int(clamp(float(max_order_size), 1.0, 10.0)),
    )


def format_genome(genome: Genome) -> str:
    return (
        f"buy={genome.buy_margin:.4f}, "
        f"sell={genome.sell_margin:.4f}, "
        f"noise={genome.valuation_noise:.4f}, "
        f"inv_aversion={genome.inventory_aversion:.4f}, "
        f"p_buy={genome.buy_probability:.3f}, "
        f"max_qty={genome.max_order_size}"
    )


def evolve_market(
    generations: int = 25,
    population_size: int = 10,
    survivor_count: int = 5,
    rounds_per_episode: int = 30,
    evaluation_episodes: int = 5,
    base_seed: int = 42,
    verbose_trades: bool = False,
) -> List[AggregateResult]:
    if generations <= 0:
        raise ValueError("generations must be positive")
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if not 1 <= survivor_count < population_size:
        raise ValueError(
            "survivor_count must be between 1 and population_size - 1"
        )
    if rounds_per_episode <= 0 or evaluation_episodes <= 0:
        raise ValueError("rounds and episodes must be positive")

    rng = random.Random(base_seed)
    population = [
        random_genome(f"G0_Agent_{i}", rng) for i in range(population_size)
    ]

    print("--- Starting Corrected Evolutionary Market Simulation ---")
    print(
        f"Population={population_size}, survivors={survivor_count}, "
        f"episodes/gen={evaluation_episodes}, rounds/episode={rounds_per_episode}\n"
    )

    ranked: List[AggregateResult] = []

    for generation in range(generations):
        ranked = evaluate_population(
            genomes=population,
            episodes=evaluation_episodes,
            rounds=rounds_per_episode,
            base_seed=base_seed,
            verbose_trades=verbose_trades,
        )

        best = ranked[0]
        worst = ranked[-1]
        print(f"--- GENERATION {generation + 1} ---")
        print(
            f"Top: {best.genome.genome_id} | score={best.mean_score:.6f} | "
            f"excess={best.mean_return_pct:.3%} | "
            f"absolute={best.mean_absolute_return_pct:.3%} | "
            f"maxDD={best.mean_max_drawdown:.3%} | "
            f"trades={best.mean_trade_count:.1f}"
        )
        print(f"     genes: {format_genome(best.genome)}")
        print(
            f"Bottom: {worst.genome.genome_id} | score={worst.mean_score:.6f} | "
            f"excess={worst.mean_return_pct:.3%} | "
            f"absolute={worst.mean_absolute_return_pct:.3%} | "
            f"maxDD={worst.mean_max_drawdown:.3%} | "
            f"trades={worst.mean_trade_count:.1f}\n"
        )

        if generation == generations - 1:
            break

        survivor_genomes = [result.genome for result in ranked[:survivor_count]]

        # Elitism copies only immutable genomes, never agents or exchanges.
        next_population: List[Genome] = [
            replace(genome) for genome in survivor_genomes
        ]

        child_index = 0
        while len(next_population) < population_size:
            parent1 = rng.choice(survivor_genomes)
            parent2 = rng.choice(survivor_genomes)
            child = breed(
                parent1,
                parent2,
                child_id=f"G{generation + 1}_Child_{child_index}",
                rng=rng,
            )
            next_population.append(child)
            child_index += 1

        population = next_population

    print("--- FINAL RANKING ---")
    for rank, result in enumerate(ranked, start=1):
        print(
            f"{rank:2d}. {result.genome.genome_id:20s} "
            f"score={result.mean_score:+.6f} "
            f"excess={result.mean_return_pct:+.3%} "
            f"absolute={result.mean_absolute_return_pct:+.3%} "
            f"DD={result.mean_max_drawdown:.3%} "
            f"trades={result.mean_trade_count:.1f}"
        )

    return ranked


# ============================================================
# Embedded smoke tests
# ============================================================


def _test_genome(genome_id: str) -> Genome:
    return Genome(
        genome_id=genome_id,
        buy_margin=0.01,
        sell_margin=0.01,
        valuation_noise=0.01,
        inventory_aversion=0.05,
        buy_probability=0.50,
        max_order_size=5,
    )


def run_smoke_tests() -> None:
    # Resting-order price, settlement, and conservation.
    exchange = MatchingEngine(initial_price=100.0, fee_rate=0.0)
    seller = MarketAwareAgent("seller", _test_genome("seller_g"), exchange)
    buyer = MarketAwareAgent("buyer", _test_genome("buyer_g"), exchange)
    exchange.snapshot_initial_totals()

    sell_order_id = exchange.submit_order("seller", "SELL", 100.0, 3, step=1)
    buy_order_id = exchange.submit_order("buyer", "BUY", 101.0, 2, step=2)
    assert sell_order_id is not None and buy_order_id is not None
    assert len(exchange.trade_history) == 1
    trade = exchange.trade_history[0]
    assert trade.price == 100.0
    assert trade.qty == 2
    assert seller.inventory == 98
    assert buyer.inventory == 102
    assert math.isclose(seller.cash, 10_200.0)
    assert math.isclose(buyer.cash, 9_800.0)
    assert exchange.orders[sell_order_id].remaining_qty == 1

    # Reservation release through cancellation.
    assert seller.reserved_inventory == 1
    assert exchange.cancel_order(sell_order_id, requester_agent_id="seller")
    assert seller.reserved_inventory == 0

    # Self-trade prevention must cancel the newer crossing order.
    exchange2 = MatchingEngine(initial_price=100.0, fee_rate=0.0)
    self_agent = MarketAwareAgent("self", _test_genome("self_g"), exchange2)
    exchange2.snapshot_initial_totals()
    ask_id = exchange2.submit_order("self", "SELL", 100.0, 1, step=1)
    bid_id = exchange2.submit_order("self", "BUY", 101.0, 1, step=2)
    assert ask_id is not None and bid_id is not None
    assert len(exchange2.trade_history) == 0
    assert exchange2.orders[bid_id].status == "CANCELLED"
    assert exchange2.orders[bid_id].cancel_reason == "SELF_TRADE_PREVENTION"
    exchange2.cancel_all_orders()

    # Invalid and unaffordable orders.
    exchange3 = MatchingEngine(initial_price=100.0, fee_rate=0.0)
    poor = MarketAwareAgent(
        "poor",
        _test_genome("poor_g"),
        exchange3,
        initial_cash=10.0,
        initial_inventory=0,
    )
    exchange3.snapshot_initial_totals()
    assert exchange3.submit_order("poor", "BUY", 100.0, 1) is None
    assert exchange3.submit_order("poor", "SELL", 100.0, 1) is None

    try:
        exchange3.submit_order("poor", "HOLD", 100.0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid side should raise ValueError")

    exchange3._assert_invariants()
    print("Smoke tests passed.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrected evolutionary limit-order-book market simulation"
    )
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--survivors", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument(
        "--verbose-trades",
        action="store_true",
        help="Print every trade; useful only for small runs",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip embedded smoke tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_tests:
        run_smoke_tests()

    evolve_market(
        generations=args.generations,
        population_size=args.agents,
        survivor_count=args.survivors,
        rounds_per_episode=args.rounds,
        evaluation_episodes=args.episodes,
        base_seed=args.seed,
        verbose_trades=args.verbose_trades,
    )


if __name__ == "__main__":
    main()
