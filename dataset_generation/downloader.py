
import urllib.request
import shutil
import os 
import tarfile

DOWNLOAD_FOLDER = "./compressed_paper_folders/"
EXTRACT_FOLDER = "./paper_folders/"

papers = open("selected_papers.txt", "rt")
lines = papers.read().split("\n")
papers.close()

nb_papers = int(len(lines) / 2)
print("number papers:", nb_papers)

if not os.path.exists(DOWNLOAD_FOLDER):
        os.mkdir(DOWNLOAD_FOLDER)

if not os.path.exists(EXTRACT_FOLDER):
        os.mkdir(EXTRACT_FOLDER)

for i in range(0, nb_papers, 1):
        paper_link = lines[2*i+1]
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
        except Exception:
                print("error at paper %g" % (i))
        print("progress: %g / %g" % (i,nb_papers), end="\r")