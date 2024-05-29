import scipy
import mat73

def loadMatLabFile(file_path):
    """Load a .mat file and return the data as a list using mat73"""
    mat = mat73.loadmat(file_path)
    out = list(mat.items())[0][1]
    return out