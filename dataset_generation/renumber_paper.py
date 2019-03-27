import os

FINAL_FOLDER = "./papers/"

if not os.path.exists(FINAL_FOLDER):
    os.mkdir(FINAL_FOLDER)

papers = open("selected_papers.txt", "rt")
selected_papers = papers.read().split("\n")
papers.close()

nb_papers = int(len(selected_papers) / 2)
print("number papers:", nb_papers)

existing_papers_indices = []

selected_papers_file = open("selected_papers.txt", "wt")

final_nb_papers = 0

for i in range(0, 1000, 1):
    if os.path.exists(FINAL_FOLDER + str(i) + ".tex"):
        final_nb_papers += 1
        existing_papers_indices.append(i)
        paper_title = selected_papers[2*i]
        paper_link = selected_papers[2*i+1]
        selected_papers_file.write(paper_title + "\n")
        selected_papers_file.write(paper_link + "\n")

selected_papers_file.close()

for i, paper_indice in enumerate(existing_papers_indices):
    os.rename(FINAL_FOLDER+str(paper_indice)+".tex", FINAL_FOLDER+str(i)+".tex")

print("final number papers:", final_nb_papers)