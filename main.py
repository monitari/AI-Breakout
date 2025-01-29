import pygame
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import math
import pygame.mixer
pygame.mixer.init()
beep_sound = pygame.mixer.Sound("assets/8-bit-explosion-95847.mp3")
pad_sound = pygame.mixer.Sound("assets/retro-jump-3-236683.mp3")
death_sound = pygame.mixer.Sound("assets/retro-explode-1-236678.mp3")
wall_sound = pygame.mixer.Sound("assets/8-bit-laser-151672.mp3")

class Config:
    # 화면 설정
    SCREEN_SIZE = (1000, 600)
    COLORS = {
        'WHITE': (255, 255, 255),
        'BLUE': (0, 0, 255),
        'RED': (255, 0, 0),
        'YELLOW': (255, 255, 0),
        'BLACK': (0, 0, 0)
    }
    
    # 게임 요소 설정
    GAME = {
        'PADDLE_WIDTH': 120,
        'BALL_SIZE': 10,
        'BRICK_WIDTH': 50,
        'BRICK_HEIGHT': 20,
        'BRICK_GAP': 2,
        'BRICK_START_Y': 30,
        'BASE_SPEED': 15,
        'PADDLE_SPEED': 30
    }
    
    # 강화학습 설정
    RL = {
        'STATE_SIZE': 7,
        'ACTION_SIZE': 3,
        'BATCH_SIZE': 128,
        'MEMORY_SIZE': 100000,
        'GAMMA': 0.99,
        'EPSILON_START': 1.0,
        'EPSILON_MIN': 0.01,
        'EPSILON_DECAY': 0.995,
        'LEARNING_RATE': 0.0005,
        'TAU': 0.01,
        'MAX_EPISODES': 10000,
        'MAX_STEPS': 10000,
        'REWARD_SCALE': 0.5
    }
    
    # 렌더링 설정
    RENDER = {
        'ALWAYS_RENDER': True,
        'RENDER_FPS': 30,
        'TRAINING_FPS': 1000
    }

class GameObjects:
    class Paddle(pygame.sprite.Sprite):
        def __init__(self, width, human_mode=False):
            super().__init__()
            self.width = Config.GAME['PADDLE_WIDTH']  # Config 값 사용
            self.image = pygame.Surface((self.width, 20))
            self.image.fill(Config.COLORS['BLUE'])
            self.rect = self.image.get_rect(center=(Config.SCREEN_SIZE[0]//2, Config.SCREEN_SIZE[1]-50))
            self.speed = 0
            self.max_speed = Config.GAME['PADDLE_SPEED']

        def update(self, action=None):
            if action is not None:
                self.speed = self.max_speed if action == 2 else -self.max_speed if action == 0 else 0
            else:
                keys = pygame.key.get_pressed()
                self.speed = -self.max_speed if keys[pygame.K_LEFT] else self.max_speed if keys[pygame.K_RIGHT] else 0

            self.rect.x += self.speed
            self.rect.left = max(self.rect.left, 0)
            self.rect.right = min(self.rect.right, Config.SCREEN_SIZE[0])

    class Ball(pygame.sprite.Sprite):
        def __init__(self, base_speed=5, paddle=None, human_mode=False):
            super().__init__()
            self.paddle = paddle 
            self.image = pygame.Surface((Config.GAME['BALL_SIZE'], Config.GAME['BALL_SIZE']), pygame.SRCALPHA)  # 투명 배경 사용
            pygame.draw.ellipse(self.image, Config.COLORS['RED'], [0, 0, Config.GAME['BALL_SIZE'], Config.GAME['BALL_SIZE']])  # 동그라미 그리기
            self.rect = self.image.get_rect(center=Config.SCREEN_SIZE)
            self.prev_rect = self.rect.copy()  # 이전 위치 저장 추가
            self.human_mode = human_mode  # human_mode 속성 추가
            self.base_speed = Config.GAME['BASE_SPEED']
            self.speed_multiplier = 1.0  # 속도 증가율 초기화
            self.beep_sound = beep_sound
            self.pad_sound = pad_sound
            self.reset()

        def reset(self):
            angle = random.uniform(math.pi/6, math.pi/3)  # 30° ~ 60° 사이 각도
            speed_factor = 0.75
            self.speed_x = self.base_speed * math.cos(angle) * random.choice([1, -1]) * speed_factor
            self.speed_y = self.base_speed * math.sin(angle) * speed_factor
            self.rect.center = (Config.SCREEN_SIZE[0]//2, Config.SCREEN_SIZE[1]//2)

        def update(self):
            self.prev_rect = self.rect.copy()  # 이동 전 위치 저장
            steps = int(math.ceil(max(abs(self.speed_x), abs(self.speed_y)) / 5)) + 1  # steps 계산 조정
            for _ in range(steps):
                self.rect.x += self.speed_x / steps
                self.rect.y += self.speed_y / steps
                self._check_collision()  # 단계별 충돌 검사

        def _check_collision(self):
            # 정밀 충돌 검사 메서드
            if self.rect.left <= 0 or self.rect.right >= Config.SCREEN_SIZE[0]:
                self.speed_x *= -1
                wall_sound.play()  # 벽 충돌 사운드 추가
            if self.rect.top <= 0:
                self.speed_y *= -1
                wall_sound.play()  # 벽 충돌 사운드 추가

        def reflect(self, surface_type, collision_point=None):
            if surface_type == "paddle":
                # 패들 충돌: 위치에 따라 각도 변경
                offset = (collision_point - self.paddle.rect.centerx) / (self.paddle.rect.width/2)
                angle = offset * math.pi/3  # -60°~+60° 범위
                self.speed_x = self.base_speed * math.sin(angle) * self.speed_multiplier
                self.speed_y = -abs(self.base_speed * math.cos(angle)) * self.speed_multiplier
                self.pad_sound.play()
                
            elif surface_type == "brick":
                # 벽돌 충돌: 충돌 면 판별
                if collision_point == "top_bottom":
                    self.speed_y *= -1
                else:
                    self.speed_x *= -1
                #self.speed_multiplier *= 1.01 # 속도 증가율 증가 (선택 사항)
                self.beep_sound.play()

        def _handle_brick_collision(self, brick):
            """벽돌 충돌 처리 최적화 버전"""
            prev_center = pygame.Vector2(self.prev_rect.center)
            curr_center = pygame.Vector2(self.rect.center)
            move_vec = curr_center - prev_center
            
            # overlap_x = min(self.rect.right, brick.rect.right) - max(self.rect.left, brick.rect.left)
            # overlap_y = min(self.rect.bottom, brick.rect.bottom) - max(self.rect.top, brick.rect.top)
            
            if abs(move_vec.x) > abs(move_vec.y):
                self.speed_x *= -1
            else:
                self.speed_y *= -1
                
    class Brick(pygame.sprite.Sprite):
        def __init__(self, x, y):
            super().__init__()
            self.image = pygame.Surface((Config.GAME['BRICK_WIDTH'], Config.GAME['BRICK_HEIGHT']))  # Config 값 사용
            self.image.fill(Config.COLORS['YELLOW'])
            self.rect = self.image.get_rect(topleft=(x, y))

class BreakoutGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(Config.SCREEN_SIZE)
        pygame.display.set_caption("Breakout AI v2.0")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 20, bold=True)  # 폰트 크기 줄이기
        self.level = 1
        self.lives = 3
        self.game_over = False
        self.game_over_sound = death_sound
        self.episode = 0

    def reset(self, human_mode=False):
        self.human_mode = human_mode  # human_mode 속성 추가
        self.paddle = GameObjects.Paddle(Config.GAME['PADDLE_WIDTH'], human_mode=self.human_mode)
        self.ball = GameObjects.Ball(5 + self.level//2, self.paddle, human_mode=self.human_mode)
        self._init_bricks(8 + self.level)  # 벽돌 층 수 늘리기
        self.score = 0
        self.lives = 3 
        self.steps = 0
        self.episode += 1
        return self.get_normalized_state()

    def _init_bricks(self, rows):
        self.bricks = pygame.sprite.Group()
        
        # 화면 너비 기반으로 열 개수 자동 계산
        max_columns = (Config.SCREEN_SIZE[0] - Config.GAME['BRICK_GAP']) // (Config.GAME['BRICK_WIDTH'] + Config.GAME['BRICK_GAP'])
        total_width = max_columns * (Config.GAME['BRICK_WIDTH'] + Config.GAME['BRICK_GAP']) - Config.GAME['BRICK_GAP']
        
        # 벽돌을 화면 중앙에 정렬하기 위한 시작 위치 계산
        start_x = (Config.SCREEN_SIZE[0] - total_width) // 2
        
        # 벽돌 생성
        for row in range(rows):
            for col in range(max_columns):
                x = start_x + col * (Config.GAME['BRICK_WIDTH'] + Config.GAME['BRICK_GAP'])
                y = row * (Config.GAME['BRICK_HEIGHT'] + Config.GAME['BRICK_GAP']) + Config.GAME['BRICK_START_Y']
                self.bricks.add(GameObjects.Brick(x, y))

    def get_normalized_state(self):
        """정규화된 상태 관찰값 계산"""
        paddle_x = self.paddle.rect.centerx / Config.SCREEN_SIZE[0]
        ball_rel_x = (self.ball.rect.centerx - self.paddle.rect.centerx) / Config.SCREEN_SIZE[0]
        ball_y = self.ball.rect.centery / Config.SCREEN_SIZE[1]
        speed_x = self.ball.speed_x / 10.0
        speed_y = self.ball.speed_y / 10.0
        relative_motion = (self.ball.rect.centerx - self.paddle.rect.centerx) * self.ball.speed_x
        paddle_speed = self.paddle.speed / Config.GAME['PADDLE_SPEED']
        
        return np.array([
            paddle_x,
            ball_rel_x,
            ball_y,
            speed_x,
            speed_y,
            relative_motion,
            paddle_speed
        ], dtype=np.float32)

    def step(self, action):
        reward = 0.01  # 기본 생존 보상
        done = False
        self.steps += 1

        self.paddle.update(action=action)
        self.ball.update()

        # 패들 충돌 처리 (개선판)
        if self.ball.rect.colliderect(self.paddle.rect):
            # 충돌 위치 계산
            collision_x = self.ball.rect.centerx
            self.ball.paddle = self.paddle  # 패들 정보 전달
            self.ball.reflect("paddle", collision_x)  # 패들 충돌 위치 전달
            self.ball.rect.bottom = self.paddle.rect.top

        # 벽돌 충돌 처리 (변경된 부분)
        brick_collisions = pygame.sprite.spritecollide(self.ball, self.bricks, True)
        if brick_collisions:
            # 충돌 면 판별
            for brick in brick_collisions:
                if self.ball.rect.centerx < brick.rect.left or self.ball.rect.centerx > brick.rect.right:
                    self.ball.reflect("brick", "side")
                else:
                    self.ball.reflect("brick", "top_bottom")
            
            self.score += len(brick_collisions) * 10  # 벽돌 파괴 시 점수 추가

        # 종료 조건
        if self.ball.rect.top > Config.SCREEN_SIZE[1]:
            self.game_over_sound.play() 
            reward -= 1.0
            self.lives -= 1
            if self.lives <= 0:
                done = True
                self.game_over = True
            else:
                # 패들 및 공 리셋
                self.ball.reset()
                self.paddle.rect.centerx = Config.SCREEN_SIZE[0]//2
                done = False

        if not self.bricks:
            done = True
            self.level += 1  # 레벨 업

        if self.steps >= Config.RL['MAX_STEPS']:
            done = True

        reward *= Config.RL['REWARD_SCALE']  # 보상 스케일링 적용
        return reward, done, self.get_normalized_state()  # 클리핑 제거

    def render(self, training_mode=False):
        self.screen.fill(Config.COLORS['BLACK'])
        self.screen.blit(self.paddle.image, self.paddle.rect)
        self.screen.blit(self.ball.image, self.ball.rect)
        self.bricks.draw(self.screen)
        self._draw_info_panel()
        pygame.display.flip()
        self.clock.tick(Config.RENDER['RENDER_FPS'])

    def _draw_info_panel(self):
        info = [
            f"Episode: {self.episode}",
            f"Level: {self.level}",
            f"Lives: {self.lives}",
            f"Score: {self.score}",
            f"Steps: {self.steps}",
            f"FPS: {self.clock.get_fps():.2f}"
        ]
        
        # 텍스트 표면 생성
        text_surface = self.font.render(" | ".join(info), True, Config.COLORS['WHITE'])
        
        # 배경 박스 렌더링 (반투명)
        bg_rect = pygame.Rect(5, 5, text_surface.get_width()+10, text_surface.get_height()+5)
        s = pygame.Surface((bg_rect.width, bg_rect.height))  # 크기 설정
        s.set_alpha(128)  # 투명도 설정 (0-255)
        s.fill(Config.COLORS['BLACK'])  # 색상 설정
        self.screen.blit(s, (bg_rect.x, bg_rect.y))  # 화면에 반투명 박스 그리기
        pygame.draw.rect(self.screen, Config.COLORS['WHITE'], bg_rect, 1)  # 테두리 그리기
        
        # 텍스트 렌더링
        self.screen.blit(text_surface, (10, 8))

class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.RL['LEARNING_RATE'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=100)
        self.memory = deque(maxlen=Config.RL['MEMORY_SIZE'])
        self.epsilon = Config.RL['EPSILON_START']
        self.loss_fn = nn.SmoothL1Loss()
        self.update_target()

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(Config.RL['STATE_SIZE'], 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, Config.RL['ACTION_SIZE'])
        )

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def soft_update(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(Config.RL['TAU']*param.data + (1-Config.RL['TAU'])*target_param.data)

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return random.randrange(Config.RL['ACTION_SIZE'])
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(state).argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < Config.RL['BATCH_SIZE']:
            return None

        batch = random.sample(self.memory, Config.RL['BATCH_SIZE'])
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target = rewards + (1 - dones) * Config.RL['GAMMA'] * next_q

        loss = self.loss_fn(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.soft_update()
        
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(Config.RL['EPSILON_MIN'], self.epsilon * Config.RL['EPSILON_DECAY'])

def train():
    env = BreakoutGame()
    agent = DQNAgent()
    best_score = -np.inf
    stats = {'rewards': [], 'losses': [], 'epsilons': []}

    for episode in range(Config.RL['MAX_EPISODES']):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0
        done = False

        while not done and steps < Config.RL['MAX_STEPS']:
            # 렌더링 처리
            if Config.RENDER['ALWAYS_RENDER']:
                env.render(training_mode=True)
                pygame.display.update()

            # 이벤트 처리 (창 닫기 방지)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # 에이전트 동작 및 학습
            action = agent.get_action(state)
            reward, done, next_state = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1

            loss = agent.replay()
            if loss is not None:
                total_loss += loss

            # 학습 속도 조절
            pygame.time.Clock().tick(Config.RENDER['TRAINING_FPS'])

        agent.decay_epsilon()
        avg_loss = total_loss / steps if steps > 0 else 0
        stats['rewards'].append(total_reward)
        stats['losses'].append(avg_loss)
        stats['epsilons'].append(agent.epsilon)

        # 학습률 동적 조정
        agent.scheduler.step(avg_loss)

        # 모니터링 출력
        print(f"Ep {env.episode:4d} | Reward {total_reward:7.2f} | Loss {avg_loss:7.4f} "
              f"| Eps {agent.epsilon:.3f} | LR {agent.optimizer.param_groups[0]['lr']:.2e}")

        # 최고 모델 저장
        if total_reward > best_score:
            best_score = total_reward
            torch.save(agent.model.state_dict(), f'best_model.pth')

    # 학습 종료 후 통계 시각화 (추가 구현 필요)
    pygame.quit()

def play(use_ai=True, model_path='best_model.pth'):
    env = BreakoutGame()
    agent = DQNAgent()
    
    if use_ai:
        agent.model.load_state_dict(torch.load(model_path))
        agent.model.eval()
        print("AI 플레이어 준비 완료!")
    else:
        env.human_mode = True  # 인간 모드 활성화

    running = True
    while running:
        state = env.reset(human_mode=not use_ai)  # human_mode 파라미터 전달
        total_reward = 0
        done = False
        
        while not done:
            env.render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
            
            if use_ai:
                with torch.no_grad():
                    action = agent.model(torch.FloatTensor(state).to(agent.device)).argmax().item()
            else:
                keys = pygame.key.get_pressed()
                action = 0 if keys[pygame.K_LEFT] else 2 if keys[pygame.K_RIGHT] else 1

            reward, done, next_state = env.step(action)
            state = next_state
            total_reward += reward

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False
                done = True

        print(f"Total reward: {total_reward:.2f}")
        env.game_over = False

    pygame.quit()

class GameManager:
    """게임 모드 관리 클래스"""
    @staticmethod
    def select_mode():
        """모드 선택 메뉴 개선"""
        while True:
            print("\n" + "="*40)
            print("Breakout Game Modes:")
            print("1. AI Training Mode")
            print("2. AI Play Mode")
            print("3. Human Play Mode")
            print("4. Exit")
            
            choice = input("Select mode (1-4): ").strip()
            
            if choice == '1':
                GameManager.run_train_mode()
            elif choice == '2':
                GameManager.run_ai_mode()
            elif choice == '3':
                GameManager.run_human_mode()
            elif choice == '4':
                print("Exiting game...")
                pygame.quit()
                sys.exit()
            else:
                print("Invalid input! Please enter 1-4.")

    @staticmethod
    def run_train_mode():
        """학습 모드 실행"""
        train()

    @staticmethod
    def run_ai_mode():
        """AI 플레이 모드 실행"""
        play(use_ai=True, model_path='best_model.pth')

    @staticmethod
    def run_human_mode():
        """인간 플레이 모드 실행"""
        play(use_ai=False)

if __name__ == "__main__":
    try:
        GameManager.select_mode()
    except KeyboardInterrupt:
        print("\nGame terminated by user!")
    finally:
        pygame.quit()
        sys.exit()