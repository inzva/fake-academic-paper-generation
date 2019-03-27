
import urllib.request
import shutil
import os 
import tarfile
import random
import glob
from TexSoup import TexSoup

def import_resolve(tex, path):
    """Resolve all imports and update the parse tree.
    Reads from a tex file and once finished, writes to a tex file.
    """
    soup = TexSoup(tex)
    dir_path = os.path.dirname(path) + "/"
    
    for _input in soup.find_all('input'):
        #print("input statement detected")
        path = os.path.join(dir_path, _input.args[0])
        if not os.path.exists(path):
            path = path + ".tex"
        #print("Resolved Path:", path)
        _input.replace(*import_resolve(open(path), dir_path).contents)
    
    # CHECK FOLLOWING ONES
    # resolve subimports
    for subimport in soup.find_all('subimport'):
        #print("subimport statement detected")
        path = os.path.join(dir_path, subimport.args[0] + subimport.args[1])
        if not os.path.exists(path):
            path = path + ".tex"
        #print("Resolved Path:", path)
        subimport.replace(*import_resolve(open(path), dir_path).contents)

    # resolve imports
    for _import in soup.find_all('import'):
        #print("import statement detected")
        path = os.path.join(dir_path, _import.args[0])
        if not os.path.exists(path):
            path = path + ".tex"
        #print("Resolved Path:", path)
        _import.replace(*import_resolve(open(path), dir_path).contents)

    # resolve includes
    for include in soup.find_all('include'):
        #print("include statement detected")
        path = os.path.join(dir_path, include.args[0])
        if not os.path.exists(path):
            path = path + ".tex"
        #print("Resolved Path:", path)
        include.replace(*import_resolve(open(path), dir_path).contents)
        
    return soup

DOWNLOAD_FOLDER = "./compressed_paper_folders/"
EXTRACT_FOLDER = "./paper_folders/"
FINAL_FOLDER = "./papers/"

if not os.path.exists(DOWNLOAD_FOLDER):
        os.mkdir(DOWNLOAD_FOLDER)

if not os.path.exists(EXTRACT_FOLDER):
        os.mkdir(EXTRACT_FOLDER)

if not os.path.exists(FINAL_FOLDER):
    os.mkdir(FINAL_FOLDER)

papers = open("selected_papers.txt", "rt")
selected_papers = papers.read().split("\n")
papers.close()

nb_papers = int(len(selected_papers) / 2)
print("number papers:", nb_papers)

papers = open("paperlinks.txt", "rt")
all_papers = papers.read().split("\n")
papers.close()

nb_total_papers = int(len(all_papers) / 2)
print("number total papers:", nb_total_papers)

error_papers_file = open("error_papers.txt", "at")

error_indices = []

for i in range(0, nb_papers, 1):
    if not (os.path.exists(FINAL_FOLDER + str(i) + ".tex") or os.path.exists(FINAL_FOLDER + str(i) + "-0.tex")):
        print("missing paper %g" % (i))
        error_indices.append(i)
        paper_link = selected_papers[2*i+1]
        error_papers_file.write(paper_link + "\n")

error_papers_file.close()

error_papers_file = open("error_papers.txt", "rt")
error_papers = error_papers_file.read().split("\n")
error_papers_file.close()

for i in error_indices:
    new_selection = random.randint(0,nb_total_papers-1)
    paper_link = all_papers[2*new_selection+1]
    while paper_link in error_papers or paper_link in selected_papers:
        new_selection = random.randint(0,nb_total_papers-1)
        paper_link = all_papers[2*new_selection+1]
    
    paper_title = all_papers[2*new_selection]
    selected_papers[2*i] = paper_title
    selected_papers[2*i+1] = paper_link

    paper_code = paper_link.split("/")[-1]
    paper_source_link = "https://arxiv.org/e-print/" + paper_code
    try:
        # Download the file from `paper_source_link` and save it locally under `DOWNLOAD_FOLDER+str(i)+".tar.gz"`:
        compressed_file_path = DOWNLOAD_FOLDER+str(i)+".tar.gz"
        with urllib.request.urlopen(paper_source_link) as response, open(compressed_file_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

        # Extract from tar the tar file
        tar = tarfile.open(compressed_file_path)
        paper_folder_dir = EXTRACT_FOLDER + str(i) + "/"
        tar.extractall(path=paper_folder_dir)
        tar.close()

        # Solve Latex Input Statements
        paper_folder_dir = EXTRACT_FOLDER + str(i) + "/**/"
        extension = "*.tex"
        tex_files = glob.glob(paper_folder_dir + extension, recursive=True)

        root_files = []

        for f_path in tex_files:
            with open(f_path) as f:
                tex = f.read()
                soup = TexSoup(tex)
                if soup.documentclass is not None:
                    latex_object = import_resolve(tex, f_path)
                    root_files.append(latex_object)

        if len(root_files) < 1:
            print("no root file?")
        elif len(root_files) > 1:
            print("writing multiple root files for paper", i)
            for j in range(len(root_files)):
                with open(FINAL_FOLDER + str(i) + "-" + str(j) + ".tex", "wt") as f:
                    f.write(str(root_files[j]))
        else:
            print("writing single root file for paper", i)
            with open(FINAL_FOLDER + str(i) + ".tex", "wt") as f:
                f.write(str(root_files[0]))

    except Exception as e:
            print("error at paper %g" % (i))
            print(e)
    print("progress: %g / %g" % (i,nb_papers), end="\r")

# Rewrite selected_papers_file
selected_papers_file = open("selected_papers.txt", "wt")

for i in range(len(selected_papers)-1):
    selected_papers_file.write(selected_papers[i] + "\n")

selected_papers_file.close()