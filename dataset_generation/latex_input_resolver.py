import glob
import os
from TexSoup import TexSoup

EXTRACT_FOLDER = "./paper_folders/"
FINAL_FOLDER = "./papers/"

if not os.path.exists(FINAL_FOLDER):
    os.mkdir(FINAL_FOLDER)

papers = open("selected_papers.txt", "rt")
lines = papers.read().split("\n")
papers.close()

nb_papers = int(len(lines) / 2)
print("number papers:", nb_papers)

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

for i in range(0, nb_papers, 1):
    paper_folder_dir = EXTRACT_FOLDER + str(i) + "/**/"
    extension = "*.tex"
    tex_files = glob.glob(paper_folder_dir + extension, recursive=True)

    root_files = []
    
    #print(tex_files)
    try:
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

    except Exception:
        print("error at paper %g" % (i))
    
    print("progress: %g / %g" % (i,nb_papers))