import random

papers = open("paperlinks.txt", "rt")
lines = papers.read().split("\n")

nb_papers = int(len(lines) / 2)
print("number papers:", nb_papers)

selected_indices = random.sample(list(range(nb_papers)), 1000)

selected_papers = open("selected_papers.txt", "wt")

for i in selected_indices:
    selected_papers.write(lines[2*i] + "\n")
    selected_papers.write(lines[2*i+1] + "\n")

papers.close()
selected_papers.close()