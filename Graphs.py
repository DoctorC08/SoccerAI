# Graphs.py
# by christophermao
# 4/5/24
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import matplotlib.pyplot as plt


class Graph:
    def __init__(self, labels):
        self._nodes = set()  # Set of nodes in the graph
        self.label = labels

    def add_node(self, node):
        self._nodes.add(node)
    def display(self):
        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

        # Create nodes
        for node in self._nodes:
            plt.plot(node[0], node[1], marker="o", color="blue", label=None)


        # Customize the plot
        plt.title(self.label[0])
        plt.xlabel(self.label[1])
        plt.ylabel(self.label[2])
        plt.grid(True)
        plt.axis("off")  # Remove axis if nodes have specific positions

        plt.show()


# Example usage
# graph = Graph(["title", "x axis", "y axis"])
#
# graph.add_node((1, 2))  # You can add nodes with coordinates for positioning
# graph.add_node((3, 4))
# graph.add_node((5, 1))
#
# graph.display()
