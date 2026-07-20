"""Starfall Defender — a single-file pygame-ce demo game.

Install:
    python -m pip install pygame-ce

Run:
    python starfall_defender_pygame_ce.py

Controls:
    Move        WASD or arrow keys
    Fire        Space
    Pause       P or Escape
    Restart     R (after game over)
    Start       Enter or Space

The game uses only procedural graphics, so no external assets are required.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from enum import Enum, auto

# The dummy driver lets the built-in smoke test run on headless machines.
if "--smoke-test" in sys.argv:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame


WIDTH = 960
HEIGHT = 600
FPS = 120
TITLE = "Starfall Defender — pygame-ce Demo"

BLACK = (5, 8, 18)
WHITE = (235, 243, 255)
CYAN = (80, 220, 255)
BLUE = (70, 120, 255)
GREEN = (85, 235, 150)
YELLOW = (255, 220, 95)
ORANGE = (255, 145, 70)
RED = (255, 80, 100)
PURPLE = (180, 105, 255)
GRAY = (115, 130, 155)
DARK_PANEL = (15, 23, 43)

Vec2 = pygame.Vector2


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def circle_hit(a: Vec2, a_radius: float, b: Vec2, b_radius: float) -> bool:
    radius = a_radius + b_radius
    return a.distance_squared_to(b) <= radius * radius


class GameState(Enum):
    TITLE = auto()
    PLAYING = auto()
    PAUSED = auto()
    GAME_OVER = auto()


@dataclass(slots=True)
class Particle:
    position: Vec2
    velocity: Vec2
    color: tuple[int, int, int]
    radius: float
    lifetime: float
    max_lifetime: float

    def update(self, dt: float) -> bool:
        self.position += self.velocity * dt
        self.velocity *= 0.985 ** (dt * 60.0)
        self.lifetime -= dt
        return self.lifetime > 0.0

    def draw(self, surface: pygame.Surface) -> None:
        life_ratio = clamp(self.lifetime / self.max_lifetime, 0.0, 1.0)
        radius = max(1, int(self.radius * life_ratio))
        alpha = int(255 * life_ratio)
        sprite = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        pygame.draw.circle(
            sprite,
            (*self.color, alpha),
            (radius * 2, radius * 2),
            radius,
        )
        surface.blit(sprite, self.position - Vec2(radius * 2, radius * 2))


@dataclass(slots=True)
class Star:
    position: Vec2
    speed: float
    size: int
    brightness: int

    def update(self, dt: float) -> None:
        self.position.y += self.speed * dt
        if self.position.y > HEIGHT + 4:
            self.position.y = -4
            self.position.x = random.uniform(0, WIDTH)

    def draw(self, surface: pygame.Surface) -> None:
        shade = clamp(self.brightness, 60, 255)
        color = (int(shade * 0.72), int(shade * 0.84), int(shade))
        pygame.draw.circle(surface, color, self.position, self.size)


class Bullet:
    SPEED = 760.0

    def __init__(self, position: Vec2) -> None:
        self.position = position.copy()
        self.radius = 4.0
        self.alive = True

    def update(self, dt: float) -> None:
        self.position.y -= self.SPEED * dt
        if self.position.y < -20:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        tail = self.position + Vec2(0, 15)
        pygame.draw.line(surface, BLUE, tail, self.position, 5)
        pygame.draw.circle(surface, WHITE, self.position, 3)


class Meteor:
    def __init__(self, level: int) -> None:
        self.radius = random.randint(15, 34)
        self.position = Vec2(
            random.uniform(self.radius, WIDTH - self.radius),
            -self.radius - random.uniform(0, 100),
        )
        self.speed = random.uniform(105, 185) + level * 8
        self.drift = random.uniform(-50, 50)
        self.rotation = random.uniform(0, math.tau)
        self.angular_speed = random.uniform(-2.4, 2.4)
        self.hp = 1 if self.radius < 25 else 2
        self.max_hp = self.hp
        self.alive = True
        self.vertices = self._make_vertices()

    def _make_vertices(self) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        count = random.randint(8, 11)
        for index in range(count):
            angle = math.tau * index / count
            scale = random.uniform(0.72, 1.0)
            points.append((angle, scale))
        return points

    def update(self, dt: float) -> None:
        self.position += Vec2(self.drift, self.speed) * dt
        self.rotation += self.angular_speed * dt
        if self.position.y - self.radius > HEIGHT + 40:
            self.alive = False

    def hit(self) -> bool:
        self.hp -= 1
        if self.hp <= 0:
            self.alive = False
            return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        points: list[Vec2] = []
        for angle, scale in self.vertices:
            rotated = angle + self.rotation
            points.append(
                self.position
                + Vec2(math.cos(rotated), math.sin(rotated))
                * self.radius
                * scale
            )

        outer = (115, 96, 105) if self.hp == self.max_hp else (175, 105, 85)
        pygame.draw.polygon(surface, outer, points)
        pygame.draw.polygon(surface, (55, 48, 65), points, 3)

        crater_offset = Vec2(self.radius * 0.25, -self.radius * 0.18).rotate_rad(
            -self.rotation
        )
        pygame.draw.circle(
            surface,
            (68, 58, 70),
            self.position + crater_offset,
            max(3, self.radius // 5),
        )


class PowerUp:
    KINDS = ("shield", "rapid", "heal")

    def __init__(self, position: Vec2) -> None:
        self.kind = random.choices(self.KINDS, weights=(0.42, 0.42, 0.16), k=1)[0]
        self.position = position.copy()
        self.speed = 125.0
        self.radius = 13.0
        self.angle = 0.0
        self.alive = True

    @property
    def color(self) -> tuple[int, int, int]:
        return {"shield": CYAN, "rapid": YELLOW, "heal": GREEN}[self.kind]

    def update(self, dt: float) -> None:
        self.position.y += self.speed * dt
        self.angle += dt * 3.0
        if self.position.y - self.radius > HEIGHT:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        pulse = 1.0 + 0.12 * math.sin(self.angle * 3.0)
        radius = int(self.radius * pulse)
        pygame.draw.circle(surface, self.color, self.position, radius, 3)
        pygame.draw.circle(surface, (*self.color,), self.position, 4)

        if self.kind == "shield":
            pygame.draw.arc(
                surface,
                self.color,
                pygame.Rect(
                    self.position.x - 8,
                    self.position.y - 8,
                    16,
                    16,
                ),
                math.pi,
                math.tau,
                2,
            )
        elif self.kind == "rapid":
            pygame.draw.line(
                surface,
                self.color,
                self.position + Vec2(-4, 6),
                self.position + Vec2(4, -6),
                3,
            )
        else:
            pygame.draw.line(
                surface,
                self.color,
                self.position + Vec2(-6, 0),
                self.position + Vec2(6, 0),
                3,
            )
            pygame.draw.line(
                surface,
                self.color,
                self.position + Vec2(0, -6),
                self.position + Vec2(0, 6),
                3,
            )


class Player:
    def __init__(self) -> None:
        self.position = Vec2(WIDTH / 2, HEIGHT - 75)
        self.radius = 18.0
        self.speed = 390.0
        self.lives = 3
        self.fire_timer = 0.0
        self.invulnerable_timer = 0.0
        self.shield_timer = 0.0
        self.rapid_timer = 0.0

    @property
    def fire_delay(self) -> float:
        return 0.09 if self.rapid_timer > 0 else 0.20

    def reset(self) -> None:
        self.__init__()

    def update(self, dt: float, keys: pygame.key.ScancodeWrapper) -> None:
        direction = Vec2(
            float(keys[pygame.K_d] or keys[pygame.K_RIGHT])
            - float(keys[pygame.K_a] or keys[pygame.K_LEFT]),
            float(keys[pygame.K_s] or keys[pygame.K_DOWN])
            - float(keys[pygame.K_w] or keys[pygame.K_UP]),
        )
        if direction.length_squared() > 0:
            direction = direction.normalize()
            self.position += direction * self.speed * dt

        self.position.x = clamp(self.position.x, 28, WIDTH - 28)
        self.position.y = clamp(self.position.y, HEIGHT * 0.38, HEIGHT - 32)

        self.fire_timer = max(0.0, self.fire_timer - dt)
        self.invulnerable_timer = max(0.0, self.invulnerable_timer - dt)
        self.shield_timer = max(0.0, self.shield_timer - dt)
        self.rapid_timer = max(0.0, self.rapid_timer - dt)

    def try_fire(self) -> Bullet | None:
        if self.fire_timer > 0:
            return None
        self.fire_timer = self.fire_delay
        return Bullet(self.position + Vec2(0, -24))

    def take_hit(self) -> bool:
        if self.invulnerable_timer > 0:
            return False
        if self.shield_timer > 0:
            self.shield_timer = 0.0
            self.invulnerable_timer = 0.55
            return False
        self.lives -= 1
        self.invulnerable_timer = 1.4
        return True

    def apply_powerup(self, kind: str) -> None:
        if kind == "shield":
            self.shield_timer = max(self.shield_timer, 6.0)
        elif kind == "rapid":
            self.rapid_timer = max(self.rapid_timer, 7.0)
        elif kind == "heal":
            self.lives = min(5, self.lives + 1)

    def draw(self, surface: pygame.Surface, elapsed: float) -> None:
        if self.invulnerable_timer > 0 and int(elapsed * 12) % 2 == 0:
            return

        nose = self.position + Vec2(0, -24)
        left = self.position + Vec2(-17, 17)
        right = self.position + Vec2(17, 17)
        inner_left = self.position + Vec2(-7, 8)
        inner_right = self.position + Vec2(7, 8)

        pygame.draw.polygon(surface, CYAN, (nose, left, inner_left, inner_right, right))
        pygame.draw.polygon(surface, WHITE, (nose, left, right), 2)
        pygame.draw.circle(surface, BLUE, self.position + Vec2(0, 2), 5)

        flame_length = 14 + 5 * math.sin(elapsed * 20)
        pygame.draw.polygon(
            surface,
            ORANGE,
            (
                self.position + Vec2(-6, 15),
                self.position + Vec2(0, flame_length + 16),
                self.position + Vec2(6, 15),
            ),
        )

        if self.shield_timer > 0:
            shield_radius = 29 + int(2 * math.sin(elapsed * 6))
            pygame.draw.circle(surface, CYAN, self.position, shield_radius, 2)


class Game:
    def __init__(self, smoke_test: bool = False) -> None:
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.smoke_test = smoke_test

        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 38)
        self.font_large = pygame.font.Font(None, 74)
        self.font_huge = pygame.font.Font(None, 102)

        self.background = self._make_background()
        self.stars = [self._new_star() for _ in range(100)]
        self.player = Player()

        self.state = GameState.TITLE
        self.running = True
        self.elapsed = 0.0
        self.high_score = 0
        self.score = 0
        self.level = 1
        self.spawn_timer = 0.0
        self.screen_shake = 0.0

        self.bullets: list[Bullet] = []
        self.meteors: list[Meteor] = []
        self.powerups: list[PowerUp] = []
        self.particles: list[Particle] = []

    def _make_background(self) -> pygame.Surface:
        surface = pygame.Surface((WIDTH, HEIGHT))
        top = (5, 8, 25)
        bottom = (20, 12, 48)
        for y in range(HEIGHT):
            t = y / max(1, HEIGHT - 1)
            color = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
            pygame.draw.line(surface, color, (0, y), (WIDTH, y))
        return surface

    @staticmethod
    def _new_star() -> Star:
        speed = random.uniform(15, 90)
        return Star(
            position=Vec2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)),
            speed=speed,
            size=1 if speed < 55 else 2,
            brightness=random.randint(100, 245),
        )

    @property
    def spawn_interval(self) -> float:
        return max(0.22, 0.78 - (self.level - 1) * 0.045)

    def start_game(self) -> None:
        self.player.reset()
        self.score = 0
        self.level = 1
        self.spawn_timer = 0.25
        self.screen_shake = 0.0
        self.bullets.clear()
        self.meteors.clear()
        self.powerups.clear()
        self.particles.clear()
        self.state = GameState.PLAYING

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                elif self.state == GameState.TITLE:
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self.start_game()
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
                elif self.state == GameState.PLAYING:
                    if event.key in (pygame.K_p, pygame.K_ESCAPE):
                        self.state = GameState.PAUSED
                elif self.state == GameState.PAUSED:
                    if event.key in (pygame.K_p, pygame.K_ESCAPE):
                        self.state = GameState.PLAYING
                    elif event.key == pygame.K_q:
                        self.state = GameState.TITLE
                elif self.state == GameState.GAME_OVER:
                    if event.key in (pygame.K_r, pygame.K_RETURN, pygame.K_SPACE):
                        self.start_game()
                    elif event.key == pygame.K_ESCAPE:
                        self.state = GameState.TITLE

    def update(self, dt: float) -> None:
        self.elapsed += dt
        for star in self.stars:
            star.update(dt)

        self.particles = [particle for particle in self.particles if particle.update(dt)]
        self.screen_shake = max(0.0, self.screen_shake - dt * 24.0)

        if self.state != GameState.PLAYING:
            return

        keys = pygame.key.get_pressed()
        self.player.update(dt, keys)

        if keys[pygame.K_SPACE]:
            bullet = self.player.try_fire()
            if bullet is not None:
                self.bullets.append(bullet)
                self._burst(bullet.position, BLUE, 3, speed_range=(35, 80))

        self.level = 1 + self.score // 500
        self.spawn_timer -= dt
        if self.spawn_timer <= 0:
            self.meteors.append(Meteor(self.level))
            self.spawn_timer = self.spawn_interval * random.uniform(0.70, 1.25)

        for bullet in self.bullets:
            bullet.update(dt)
        for meteor in self.meteors:
            meteor.update(dt)
        for powerup in self.powerups:
            powerup.update(dt)

        self._handle_collisions()

        self.bullets = [item for item in self.bullets if item.alive]
        self.meteors = [item for item in self.meteors if item.alive]
        self.powerups = [item for item in self.powerups if item.alive]

        if self.player.lives <= 0:
            self.high_score = max(self.high_score, self.score)
            self.state = GameState.GAME_OVER

    def _handle_collisions(self) -> None:
        for bullet in self.bullets:
            if not bullet.alive:
                continue
            for meteor in self.meteors:
                if not meteor.alive:
                    continue
                if circle_hit(bullet.position, bullet.radius, meteor.position, meteor.radius):
                    bullet.alive = False
                    destroyed = meteor.hit()
                    self.score += 8
                    self._burst(bullet.position, ORANGE, 6, speed_range=(70, 160))
                    if destroyed:
                        self.score += 30 + meteor.radius
                        self.screen_shake = max(self.screen_shake, 4.0)
                        self._burst(
                            meteor.position,
                            ORANGE,
                            18 + meteor.radius // 2,
                            speed_range=(80, 240),
                        )
                        if random.random() < 0.10:
                            self.powerups.append(PowerUp(meteor.position))
                    break

        for meteor in self.meteors:
            if not meteor.alive:
                continue
            if circle_hit(
                self.player.position,
                self.player.radius,
                meteor.position,
                meteor.radius * 0.82,
            ):
                meteor.alive = False
                lost_life = self.player.take_hit()
                self.screen_shake = 11.0
                self._burst(
                    self.player.position,
                    RED if lost_life else CYAN,
                    30,
                    speed_range=(90, 300),
                )

        for powerup in self.powerups:
            if not powerup.alive:
                continue
            if circle_hit(
                self.player.position,
                self.player.radius,
                powerup.position,
                powerup.radius,
            ):
                powerup.alive = False
                self.player.apply_powerup(powerup.kind)
                self.score += 50
                self._burst(powerup.position, powerup.color, 20, speed_range=(70, 210))

    def _burst(
        self,
        position: Vec2,
        color: tuple[int, int, int],
        count: int,
        speed_range: tuple[float, float] = (60.0, 180.0),
    ) -> None:
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(*speed_range)
            lifetime = random.uniform(0.25, 0.75)
            self.particles.append(
                Particle(
                    position=position.copy(),
                    velocity=Vec2(math.cos(angle), math.sin(angle)) * speed,
                    color=color,
                    radius=random.uniform(2.0, 5.0),
                    lifetime=lifetime,
                    max_lifetime=lifetime,
                )
            )

    def draw(self) -> None:
        world = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        world.blit(self.background, (0, 0))

        for star in self.stars:
            star.draw(world)
        for particle in self.particles:
            particle.draw(world)
        for powerup in self.powerups:
            powerup.draw(world)
        for bullet in self.bullets:
            bullet.draw(world)
        for meteor in self.meteors:
            meteor.draw(world)

        if self.state in (GameState.PLAYING, GameState.PAUSED, GameState.GAME_OVER):
            self.player.draw(world, self.elapsed)
            self._draw_hud(world)

        shake = Vec2()
        if self.screen_shake > 0:
            shake.xy = (
                random.uniform(-self.screen_shake, self.screen_shake),
                random.uniform(-self.screen_shake, self.screen_shake),
            )

        self.screen.fill(BLACK)
        self.screen.blit(world, shake)

        if self.state == GameState.TITLE:
            self._draw_title()
        elif self.state == GameState.PAUSED:
            self._draw_overlay("PAUSED", "P / Esc to resume   •   Q for title")
        elif self.state == GameState.GAME_OVER:
            self._draw_game_over()

        pygame.display.flip()

    def _draw_hud(self, surface: pygame.Surface) -> None:
        panel = pygame.Surface((WIDTH, 58), pygame.SRCALPHA)
        panel.fill((*DARK_PANEL, 205))
        surface.blit(panel, (0, 0))

        score_text = self.font_medium.render(f"Score  {self.score:06d}", True, WHITE)
        level_text = self.font_small.render(f"Wave {self.level}", True, CYAN)
        lives_text = self.font_small.render(f"Lives  {'◆' * self.player.lives}", True, RED)
        surface.blit(score_text, (22, 10))
        surface.blit(level_text, (WIDTH // 2 - level_text.get_width() // 2, 19))
        surface.blit(lives_text, (WIDTH - lives_text.get_width() - 22, 19))

        status_x = 22
        status_y = HEIGHT - 30
        if self.player.shield_timer > 0:
            text = self.font_small.render(
                f"SHIELD {self.player.shield_timer:0.1f}s", True, CYAN
            )
            surface.blit(text, (status_x, status_y))
            status_x += text.get_width() + 24
        if self.player.rapid_timer > 0:
            text = self.font_small.render(
                f"RAPID FIRE {self.player.rapid_timer:0.1f}s", True, YELLOW
            )
            surface.blit(text, (status_x, status_y))

    def _draw_title(self) -> None:
        title = self.font_huge.render("STARFALL", True, WHITE)
        subtitle = self.font_large.render("DEFENDER", True, CYAN)
        prompt = self.font_medium.render("Press Enter or Space to launch", True, WHITE)
        controls = self.font_small.render(
            "Move: WASD / Arrows    Fire: Space    Pause: P / Esc    Fullscreen: F11",
            True,
            GRAY,
        )
        high_score = self.font_small.render(
            f"Session high score: {self.high_score}", True, YELLOW
        )

        self.screen.blit(title, (WIDTH / 2 - title.get_width() / 2, 120))
        self.screen.blit(subtitle, (WIDTH / 2 - subtitle.get_width() / 2, 205))
        self.screen.blit(prompt, (WIDTH / 2 - prompt.get_width() / 2, 350))
        self.screen.blit(controls, (WIDTH / 2 - controls.get_width() / 2, 430))
        self.screen.blit(high_score, (WIDTH / 2 - high_score.get_width() / 2, 475))

    def _draw_overlay(self, heading: str, subheading: str) -> None:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.screen.blit(overlay, (0, 0))
        heading_image = self.font_large.render(heading, True, WHITE)
        subtitle_image = self.font_small.render(subheading, True, CYAN)
        self.screen.blit(
            heading_image,
            (WIDTH / 2 - heading_image.get_width() / 2, HEIGHT / 2 - 75),
        )
        self.screen.blit(
            subtitle_image,
            (WIDTH / 2 - subtitle_image.get_width() / 2, HEIGHT / 2 + 10),
        )

    def _draw_game_over(self) -> None:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((8, 0, 18, 205))
        self.screen.blit(overlay, (0, 0))

        heading = self.font_large.render("MISSION OVER", True, RED)
        score = self.font_medium.render(f"Final score: {self.score}", True, WHITE)
        best = self.font_small.render(f"Session best: {self.high_score}", True, YELLOW)
        prompt = self.font_small.render(
            "R / Enter / Space to restart   •   Esc for title", True, CYAN
        )

        y = 185
        for image in (heading, score, best, prompt):
            self.screen.blit(image, (WIDTH / 2 - image.get_width() / 2, y))
            y += image.get_height() + 28

    def run(self) -> None:
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.033)
            self.handle_events()
            self.update(dt)
            self.draw()
        pygame.quit()

    def run_smoke_test(self, frames: int = 20) -> None:
        self.start_game()
        for _ in range(frames):
            self.handle_events()
            self.update(1.0 / 60.0)
            self.draw()
        pygame.quit()
        print(f"Smoke test passed: rendered {frames} frames with pygame {pygame.version.ver}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Starfall Defender demo.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Render a few frames using SDL's dummy video driver, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    game = Game(smoke_test=args.smoke_test)
    if args.smoke_test:
        game.run_smoke_test()
    else:
        game.run()


if __name__ == "__main__":
    main()
