class Graph:
    def __init__(self, n: int, p: float):
        """
        Инициализация графа.
        :param n: Количество вершин в графе.
        :param p: Вероятность существования ребра между двумя вершинами.
        """
        self.n = n
        self.p = p
        self.adjacency_list = [[] for _ in range(n)] # type: ignore
        self.generate_random_graph()

    def generate_random_graph(self):
        """
        Генерация случайного графа с вероятностью p для каждого ребра.
        """
        import random
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if random.random() < self.p:
                    self.adjacency_list[i].append(j)
                    self.adjacency_list[j].append(i)

    def get_adjacency_list(self, i: int):
        """
        Вывод списка смежности.
        """
        return self.adjacency_list[i]
    
    def get_adjacency_vector(self, i: int):
        vector = [0] * self.n
        for j in self.adjacency_list[i]:
            vector[j] = 1
        return vector