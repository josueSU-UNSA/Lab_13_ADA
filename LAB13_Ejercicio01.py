from fileinput import hook_compressed
from math import sqrt
import cv2
import numpy as np
import sys

def create_energy_matrix(img):
    energy_matrix = np.zeros((img.shape[0],img.shape[1]))

    for i in range (img.shape[0]):
        energy_matrix[i][0] = 1000
        energy_matrix[i][img.shape[1] - 1] = 1000

    for j in range (img.shape[1]):
        energy_matrix[0][j] = 1000
        energy_matrix[img.shape[0] - 1][j] = 1000

    for i in range (1,img.shape[0] - 1):
        for j in range (1,img.shape[1] - 1):
            rx = abs(int(img[i + 1][j][0]) - int(img[i - 1][j][0]))
            gx = abs(int(img[i + 1][j][1]) - int(img[i - 1][j][1]))
            bx = abs(int(img[i + 1][j][2]) - int(img[i - 1][j][2]))

            ry = abs(int(img[i][j + 1][0]) - int(img[i][j - 1][0]))
            gy = abs(int(img[i][j + 1][1]) - int(img[i][j - 1][1]))
            by = abs(int(img[i][j + 1][2]) - int(img[i][j - 1][2]))

            grad_x_squared = (rx ** 2) + (gx ** 2) + (bx ** 2)
            grad_y_squared = (ry ** 2) + (gy ** 2) + (by ** 2)

            energy_matrix[i][j] = sqrt(grad_x_squared + grad_y_squared)

    return energy_matrix


def delete_path(sons, i, j, img,matrix_aux):
    
    if sons[i][j] == [-1,-1]:
  
        del matrix_aux[i][j]
    
        return
    
    del matrix_aux[i][j]
  
    delete_path(sons, sons[i][j][0], sons[i][j][1], img,matrix_aux)
def delete_path2(sons, i, j, img,matrix_aux):
    
    if sons[j][i] == [-1,-1]:
        
        del matrix_aux[i][j]
        
        
        
        return

    del matrix_aux[i][j]
    
    
    delete_path2(sons, sons[j][i][1], sons[j][i][0], img,matrix_aux)


def vertical_seam(img, n):
    
    for i in range(n):
        r = img.shape[0]
        c = img.shape[1]

        matrix = create_energy_matrix(img)
        sons = []
        accumulated_energy = np.zeros((r,c))

        for i in range(r):
            sons.append([])
            for j in range(c):
                sons[i].append([-1,-1])
            
        min_dist = sys.maxsize
        min_index = []

        for i in range(c):
            accumulated_energy[r - 1][i] = matrix[r - 1][i]

        for i in range (r - 2, -1, -1):
            for j in range(c):
                if j == 0:
                    accumulated_energy[i][j] = min(accumulated_energy[i + 1][j], accumulated_energy[i + 1][j + 1]) + matrix[i][j]
                    if accumulated_energy[i][j] == accumulated_energy[i + 1][j] + matrix[i][j]:
                        sons[i][j] = list([i + 1, j])
                    else:
                        sons[i][j] = list([i + 1, j + 1])
                elif j == c - 1:
                    accumulated_energy[i][j] = min(accumulated_energy[i + 1][j], accumulated_energy[i + 1][j - 1]) + matrix[i][j]
                    if accumulated_energy[i][j] == accumulated_energy[i + 1][j] + matrix[i][j]:
                        sons[i][j] = list([i + 1, j])
                    else:
                        sons[i][j] = list([i + 1, j - 1])
                else:
                    accumulated_energy[i][j] = min(accumulated_energy[i + 1][j], accumulated_energy[i + 1][j - 1], accumulated_energy[i + 1][j + 1]) + matrix[i][j]
                    if accumulated_energy[i][j] == accumulated_energy[i + 1][j] + matrix[i][j]:
                        sons[i][j] = list([i + 1, j])
                    elif accumulated_energy[i][j] == accumulated_energy[i + 1][j - 1] + matrix[i][j]:
                        sons[i][j] = list([i + 1, j - 1])
                    else:
                        sons[i][j] = list([i + 1, j + 1])

                if i == 0 and min_dist > accumulated_energy[i][j]:
                    min_dist = accumulated_energy[i][j]
                    min_index = [i,j]

        
        matrix_aux=img.tolist()
        
     
        delete_path(sons,min_index[0],min_index[1],img,matrix_aux)
        
        img=np.array(matrix_aux)
        

    return img


def horizontal_seam(img,n):

    for i in range(n):
        r = img.shape[0]
        c = img.shape[1]

        matrix = create_energy_matrix(img)
        sons = []
        accumulated_energy = np.zeros((r,c))

        for i in range(r):
            sons.append([])
            for j in range(c):
                sons[i].append([-1,-1])
            
        min_dist = sys.maxsize
        min_index = []

        for i in range(r):
            accumulated_energy[i][c - 1] = matrix[i][c - 1]

        for j in range (c - 2, -1, -1):
            for i in range(r):
                if i == 0:
                    accumulated_energy[i][j] = min(accumulated_energy[i][j + 1], accumulated_energy[i + 1][j + 1]) + matrix[i][j]
                    if accumulated_energy[i][j] == accumulated_energy[i][j + 1] + matrix[i][j]:
                        sons[i][j] = list([i, j + 1])
                    else:
                        sons[i][j] = list([i + 1, j + 1])
                elif i == r - 1:
                    accumulated_energy[i][j] = min(accumulated_energy[i][j + 1], accumulated_energy[i - 1][j + 1]) + matrix[i][j]
                    if accumulated_energy[i][j] == accumulated_energy[i][j + 1] + matrix[i][j]:
                        sons[i][j] = list([i, j + 1])
                    else:
                        sons[i][j] = list([i - 1, j + 1])
                else:
                    accumulated_energy[i][j] = min(accumulated_energy[i][j + 1], accumulated_energy[i - 1][j + 1], accumulated_energy[i + 1][j + 1]) + matrix[i][j]
                    if accumulated_energy[i][j] == accumulated_energy[i][j + 1] + matrix[i][j]:
                        sons[i][j] = list([i, j + 1])
                    elif accumulated_energy[i][j] == accumulated_energy[i - 1][j + 1] + matrix[i][j]:
                        sons[i][j] = list([i - 1, j + 1])
                    else:
                        sons[i][j] = list([i + 1, j + 1])

                if j == 0 and min_dist > accumulated_energy[i][j]:
                    min_dist = accumulated_energy[i][j]
                    min_index = [i,j]
        matrix_aux = img.tolist()
        transpuesta2 = transpuesta(matrix_aux)
        delete_path2(sons,min_index[1],min_index[0],img,transpuesta2)
        transpuesta2 = transpuesta(transpuesta2)
        img=np.array(transpuesta2)
    return img
      

def Seam_carving(img,num_rows_to_delete,num_col_to_delete):
    img2 = vertical_seam(img,num_col_to_delete)
    img3 = horizontal_seam(img2,num_rows_to_delete)   
    cv2.imwrite("Modificado.png",img3)
    

def transpuesta(matrix):
    matrix_transpuesta = []
    for i in range(len(matrix[0])):
        matrix_transpuesta.append([])
        for j in range(len(matrix)):
            matrix_transpuesta[i].append(matrix[j][i])
    return matrix_transpuesta
img = cv2.imread("ORIGINAL.png")

Seam_carving(img,10,5)

