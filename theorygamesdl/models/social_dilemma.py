"""
Класс для моделирования социальных дилемм
"""

import numpy as np


class SocialDilemma:
    """
    Класс для моделирования различных типов социальных дилемм.
    
    Поддерживаемые типы дилемм:
    - pd: Дилемма заключенного (Prisoner's Dilemma)
    - sh: Охота на оленя (Stag Hunt)
    - ch: Игра в курицу (Chicken Game)
    - bs: Битва полов (Battle of Sexes)
    - hs: Hawk-Dove
    - sp: Snowdrift Problem
    - wmp: War of Attrition
    """
    
    def __init__(self, pd=[3, 1, 0, 4], sh=[5, 2, 0, 4], ch=[3, 0, 1, 4], bs=[2, 0, 0, 1],
                 dilemma_type="pd", steps_number=1, noise_prob=0):
        """
        Инициализация социальной дилеммы.
        
        Args:
            pd (list): Выплаты для дилеммы заключенного [CC, DD, DC, CD]
            sh (list): Выплаты для охоты на оленя [CC, DD, DC, CD]
            ch (list): Выплаты для игры в курицу [CC, DD, DC, CD]
            bs (list): Выплаты для битвы полов [CC, DD, DC, CD]
            dilemma_type (str): Тип дилеммы ('pd', 'sh', 'ch', 'bs', 'hs', 'sp', 'wmp')
            steps_number (int): Количество шагов в игре
            noise_prob (float): Вероятность шума
        """
        self.steps_number = steps_number
        self.states = np.arange(steps_number)
        self.actions = [0, 1]  # 0 - defect, 1 - cooperate
        self.curr_state = 0
        self.dilemma_type = dilemma_type
        self.pd = pd
        self.sh = sh
        self.ch = ch
        self.bs = bs
        
        # Настройка основной функции вознаграждения в зависимости от типа дилеммы
        if self.dilemma_type == "pd":
            self.main_rew_f = self.pd_rew
            self.other_rew_f1 = self.sh_rew
            self.other_rew_f2 = self.ch_rew
            self.other_rew_f3 = self.bs_rew
        if self.dilemma_type == "sh":
            self.main_rew_f = self.sh_rew
            self.other_rew_f1 = self.pd_rew
            self.other_rew_f2 = self.ch_rew
            self.other_rew_f3 = self.bs_rew
        if self.dilemma_type == "ch":
            self.main_rew_f = self.ch_rew
            self.other_rew_f1 = self.pd_rew
            self.other_rew_f2 = self.sh_rew
            self.other_rew_f3 = self.bs_rew
        if self.dilemma_type == "bs":
            self.main_rew_f = self.bs_rew
            self.other_rew_f1 = self.pd_rew
            self.other_rew_f2 = self.sh_rew
            self.other_rew_f3 = self.ch_rew
        if self.dilemma_type == "hs":
            self.main_rew_f = self.hs_rew
        if self.dilemma_type == "sp":
            self.main_rew_f = self.sp_rew
        if self.dilemma_type == "wmp":
            self.main_rew_f = self.wmp_rew
            
        self.noise_prob = noise_prob
        
    def action(self):
        """Выполнить действие и перейти к следующему состоянию"""
        self.curr_state += 1
        return self.curr_state
    
    def reset(self):
        """Сбросить игру в начальное состояние"""
        self.curr_state = 0
        
    def pd_rew(self, your_act, opponent_act, player):
        """
        Функция вознаграждения для дилеммы заключенного
        
        Args:
            your_act (int): Ваше действие (0 - предать, 1 - сотрудничать)
            opponent_act (int): Действие оппонента
            player (int): Номер игрока (0 или 1)
            
        Returns:
            float: Вознаграждение
        """
        if (your_act == 1 and opponent_act == 1):  # CC
            reward = self.pd[0]
        if (your_act == 0 and opponent_act == 0):  # DD
            reward = self.pd[1]
        if (your_act == 1 and opponent_act == 0):  # CD
            reward = self.pd[2]
        if (your_act == 0 and opponent_act == 1):  # DC
            reward = self.pd[3]
        return reward
    
    def sh_rew(self, your_act, opponent_act, player):
        """
        Функция вознаграждения для охоты на оленя
        
        Args:
            your_act (int): Ваше действие
            opponent_act (int): Действие оппонента
            player (int): Номер игрока (0 или 1)
            
        Returns:
            float: Вознаграждение
        """
        if (your_act == 1 and opponent_act == 1):  # CC
            reward = self.sh[0]
        if (your_act == 0 and opponent_act == 0):  # DD
            reward = self.sh[1]
        if (your_act == 1 and opponent_act == 0):  # CD
            reward = self.sh[2]
        if (your_act == 0 and opponent_act == 1):  # DC
            reward = self.sh[3]
        return reward
    
    def ch_rew(self, your_act, opponent_act, player):
        """
        Функция вознаграждения для игры в курицу
        
        Args:
            your_act (int): Ваше действие
            opponent_act (int): Действие оппонента
            player (int): Номер игрока (0 или 1)
            
        Returns:
            float: Вознаграждение
        """
        if (your_act == 1 and opponent_act == 1):  # CC
            reward = self.ch[0]
        if (your_act == 0 and opponent_act == 0):  # DD
            reward = self.ch[1]
        if (your_act == 1 and opponent_act == 0):  # CD
            reward = self.ch[2]
        if (your_act == 0 and opponent_act == 1):  # DC
            reward = self.ch[3]
        return reward
    
    def bs_rew(self, your_act, opponent_act, player):
        """
        Функция вознаграждения для битвы полов
        
        Args:
            your_act (int): Ваше действие
            opponent_act (int): Действие оппонента
            player (int): Номер игрока (0 или 1)
            
        Returns:
            float: Вознаграждение
        """
        if (player == 0):
            if (your_act == 1 and opponent_act == 1):
                reward = self.bs[0]
            if (your_act == 0 and opponent_act == 1):
                reward = self.bs[1]
            if (your_act == 1 and opponent_act == 0):
                reward = self.bs[2]
            if (your_act == 0 and opponent_act == 0):
                reward = self.bs[3]
        if (player == 1):
            if (your_act == 1 and opponent_act == 1):
                reward = self.bs[3]
            if (your_act == 0 and opponent_act == 1):
                reward = self.bs[2]
            if (your_act == 1 and opponent_act == 0):
                reward = self.bs[1]
            if (your_act == 0 and opponent_act == 0):
                reward = self.bs[0]
        return reward
    
    def hs_rew(self, your_act, opponent_act, player):
        """
        Функция вознаграждения для Hawk-Dove
        
        Args:
            your_act (int): Ваше действие
            opponent_act (int): Действие оппонента
            player (int): Номер игрока (0 или 1)
            
        Returns:
            float: Вознаграждение
        """
        if (player == 0):
            if (your_act == 1 and opponent_act == 1):
                reward = 0
            if (your_act == 0 and opponent_act == 1):
                reward = 1
            if (your_act == 1 and opponent_act == 0):
                reward = 1
            if (your_act == 0 and opponent_act == 0):
                reward = 0
        if (player == 1):
            if (your_act == 1 and opponent_act == 1):
                reward = 1
            if (your_act == 0 and opponent_act == 1):
                reward = 0
            if (your_act == 1 and opponent_act == 0):
                reward = 0
            if (your_act == 0 and opponent_act == 0):
                reward = 1
        return reward
    
    def sp_rew(self, your_act, opponent_act, player):
        """
        Функция вознаграждения для Snowdrift Problem
        
        Args:
            your_act (int): Ваше действие
            opponent_act (int): Действие оппонента
            player (int): Номер игрока (0 или 1)
            
        Returns:
            float: Вознаграждение
        """
        if (player == 0):
            if (your_act == 1 and opponent_act == 1):
                reward = 1
            if (your_act == 0 and opponent_act == 1):
                reward = 0
            if (your_act == 1 and opponent_act == 0):
                reward = 0
            if (your_act == 0 and opponent_act == 0):
                reward = 2
        if (player == 1):
            if (your_act == 1 and opponent_act == 1):
                reward = 2
            if (your_act == 0 and opponent_act == 1):
                reward = 3
            if (your_act == 1 and opponent_act == 0):
                reward = 1
            if (your_act == 0 and opponent_act == 0):
                reward = 0
        return reward
    
    def wmp_rew(self, your_act, opponent_act, player):
        """
        Функция вознаграждения для War of Attrition
        
        Args:
            your_act (int): Ваше действие
            opponent_act (int): Действие оппонента
            player (int): Номер игрока (0 или 1)
            
        Returns:
            float: Вознаграждение
        """
        if (player == 0):
            if (your_act == 1 and opponent_act == 1):
                reward = 3
            if (your_act == 0 and opponent_act == 1):
                reward = -1
            if (your_act == 1 and opponent_act == 0):
                reward = -2
            if (your_act == 0 and opponent_act == 0):
                reward = 0
        if (player == 1):
            if (your_act == 1 and opponent_act == 1):
                reward = -3
            if (your_act == 0 and opponent_act == 1):
                reward = 2
            if (your_act == 1 and opponent_act == 0):
                reward = 1
            if (your_act == 0 and opponent_act == 0):
                reward = 0
        return reward
    
    def reward_func(self, your_act, opponent_act, player, noise_prob=0):
        """
        Основная функция вознаграждения с возможностью шума
        
        Args:
            your_act (int): Ваше действие
            opponent_act (int): Действие оппонента
            player (int): Номер игрока (0 или 1)
            noise_prob (float): Вероятность шума
            
        Returns:
            tuple: (вознаграждение, наличие ошибки)
        """
        err = 0
        
        reward = self.main_rew_f(your_act, opponent_act, player)
        has_noise = np.random.choice([True, False], p=[noise_prob, 1 - noise_prob])
        
        if has_noise:
            err = 1
            # Закомментированный код для выбора другой функции вознаграждения при шуме
            # rew_f = np.random.choice([self.other_rew_f1, self.other_rew_f2, self.other_rew_f3])
            # reward = rew_f(your_act, opponent_act, player)
                
        return reward, err
    
    def play(self, choice_func):
        """
        Сыграть один раунд игры
        
        Args:
            choice_func: Функция выбора действия
            
        Returns:
            list: [действие игрока 1, действие игрока 2, вознаграждение игрока 1, вознаграждение игрока 2]
        """
        p1_act = choice_func(self.actions)
        p2_act = choice_func(self.actions)
        p1_rew = self.reward_func(p1_act, p2_act, 0)
        p2_rew = self.reward_func(p2_act, p1_act, 1)
        return [p1_act, p2_act, p1_rew, p2_rew]