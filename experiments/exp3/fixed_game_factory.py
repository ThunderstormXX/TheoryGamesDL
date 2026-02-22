class GameFactory:
    """
    Фабрика игр, полностью приведённая к соглашению:

        0 = Cooperate
        1 = Defect

    payoff возвращается для игрока 0.
    """

    # ============================
    #  Majority Game (всё корректно)
    # ============================
    @staticmethod
    def create_majority_game(n_players: int) -> Game:
        """Игрок получает +1 если сделал тот же выбор, что и большинство."""
        def majority_payoff(choices: Sequence[int]) -> float:
            ones = sum(choices)
            majority = 1 if ones > n_players / 2 else 0
            return +1.0 if choices[0] == majority else -1.0

        return Game(
            n_players,
            majority_payoff,
            name=f"Majority Game ({n_players} players)"
        )

    # ============================
    #  Coordination Game (исправлено)
    # ============================
    @staticmethod
    def create_coordination_game(n_players: int) -> Game:
        """
        Координационная игра: +1, если ВСЕ выбрали одно и то же действие.
        Не важно, кооперация или дефекция — важно совпадение.
        """
        def coordination_payoff(choices: Sequence[int]) -> float:
            return +1.0 if len(set(choices)) == 1 else -1.0

        return Game(
            n_players,
            coordination_payoff,
            name=f"Coordination Game ({n_players} players)"
        )

    # ============================
    #  Public Goods Game (исправлено)
    # ============================
    @staticmethod
    def create_public_goods_game(
        n_players: int,
        cost: float = 0.5,
        multiplier: float = 2
    ) -> Game:
        """
        Public Goods: 0 = cooperate (платит cost)
                      1 = defect   (не платит)
        """

        def public_goods_payoff(choices: Sequence[int]) -> float:
            cooperators = sum(1 for c in choices if c == 0)
            total_return = cooperators * multiplier
            indiv_return = total_return / n_players

            # игрок 0 платит cost только если он COOPERATE (0)
            cost_paid = cost if choices[0] == 0 else 0.0

            return float(indiv_return - cost_paid)

        return Game(
            n_players,
            public_goods_payoff,
            name=f"Public Goods Game ({n_players} players)"
        )

    # ============================
    #  Random Game (корректно)
    # ============================
    @staticmethod
    def create_random_game(
        n_players: int,
        low: float = -1,
        high: float = 1
    ) -> Game:

        payoff_tensor = np.random.uniform(low, high, (2,) * n_players)

        return Game(
            n_players,
            payoff_tensor=payoff_tensor,
            name=f"Random Game ({n_players} players)"
        )

    # ============================
    #  Generalized Prisoner's Dilemma (исправлено)
    # ============================
    @staticmethod
    def create_generalized_prisoners_dilemma(
        n_players: int,
        cooperation_reward: float = 3.0,
        defection_temptation: float = 5.0,
        mutual_defection_punishment: float = 1.0,
        sucker_payoff: float = 0.0
    ) -> Game:
        """
        Полностью согласовано с:
        qlearning_trap_simulation.py
            (C,C) → (3,3)
            (C,D) → (0,5)
            (D,C) → (5,0)
            (D,D) → (1,1)

        Здесь 0 = C, 1 = D.
        """

        def prisoners_dilemma_payoff(choices: Sequence[int]) -> float:
            player0 = choices[0]
            others = choices[1:]

            coop_others = sum(1 for c in others if c == 0)

            if player0 == 0:  # cooperate
                if coop_others == len(others):
                    return cooperation_reward      # (C,C,...)
                else:
                    return sucker_payoff           # (C,D,...)
            else:  # defect
                if coop_others == 0:
                    return mutual_defection_punishment  # (D,D,...)
                else:
                    return defection_temptation         # (D,C,...)

        return Game(
            n_players,
            prisoners_dilemma_payoff,
            name=f"Generalized Prisoner's Dilemma ({n_players} players)"
        )

    # ============================
    #  Custom 3-player game (исправлено)
    # ============================
    @staticmethod
    def create_custom_three_player_game() -> Game:
        """
        Старое payoff_dict использовало обратную семантику (0=D, 1=C).
        Мы инвертируем ключи, чтобы 0=C, 1=D.
        """
        old_dict = {
            (0, 0, 0): -10,
            (0, 1, 0): -5,
            (1, 0, 0): 25,
            (1, 1, 0): 8,
            (0, 0, 1): -5,
            (0, 1, 1): -15,
            (1, 0, 1): 8,
            (1, 1, 1): -20,
        }

        # инвертируем 0↔1
        payoff_dict = {
            tuple(1 - x for x in profile): value
            for profile, value in old_dict.items()
        }

        def custom_payoff(choices: Sequence[int]) -> float:
            return float(payoff_dict[tuple(choices)])

        return Game(
            3,
            custom_payoff,
            name="Custom 3-Player Game"
        )

    # ============================
    #  Generalized Custom Game (исправлено)
    # ============================
    @staticmethod
    def create_generalized_custom_game(
        n_players: int,
        all_defect_payoff: float = -10,
        mixed_payoff_cooperator: float = -5,
        mixed_payoff_defector: float = 25,
        two_cooperators_payoff: float = 8,
        all_cooperate_payoff: float = -20,
        lone_cooperator_payoff: float = -15,
    ) -> Game:
        """
        Приведено к соглашению:
        0 = cooperate
        1 = defect
        """

        def generalized_custom_payoff(choices: Sequence[int]) -> float:
            cooperators = sum(1 for c in choices if c == 0)
            defectors = n_players - cooperators

            c = choices[0]

            if c == 0:  # cooperate
                if cooperators == 1:
                    return lone_cooperator_payoff
                elif cooperators == n_players:
                    return all_cooperate_payoff
                else:
                    return two_cooperators_payoff

            else:  # defect
                if defectors == n_players:
                    return all_defect_payoff
                else:
                    return mixed_payoff_defector

        return Game(
            n_players,
            generalized_custom_payoff,
            name=f"Generalized Custom Game ({n_players} players)"
        )
