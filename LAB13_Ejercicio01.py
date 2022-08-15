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

            grad_x_squared = (rx * 2) + (gx * 2) + (bx ** 2)
            grad_y_squared = (ry * 2) + (gy * 2) + (by ** 2)

            energy_matrix[i][j] = sqrt(grad_x_squared + grad_y_squared)

    return energy_matrix


def delete_path(sons, i, j, img,matrix_aux):
    
    if sons[i][j] == [-1,-1]:
        # img[i][j] = [0,0,0]
        del matrix_aux[i][j]
        
        
        #img[i].pop(j)
        return
    # img[i][j] = [0,0,0]
    del matrix_aux[i][j]
    
    #img[i].pop(j)
    delete_path(sons, sons[i][j][0], sons[i][j][1], img,matrix_aux)



    """
    for i in range(n):
        for j in range(m):
            if visited[i][j] == False and distances[i][j] <= min:
                min = distances[i][j]
                min_index = [i,j]
    return min_index"""
# Vertical seam
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
        
        # for i in range(len(img)):
        #     matrix_aux.append(img[i])
        delete_path(sons,min_index[0],min_index[1],img,matrix_aux)
        
        img=np.array(matrix_aux)
        

        # cv2.imwrite("aqp2.png",img)
        # cv2.waitKey(0)


# Horizontal seam

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
        
        matrix_aux=img.tolist()
        #new_img = img.tolist()
        delete_path(sons,min_index[0],min_index[1],img)
        #img = np.array(new_img)
        img=np.array(matrix_aux)

        # cv2.imwrite("aqp2.png",img)
        # cv2.waitKey(0)

def Seam_carving(img,num_rows_to_delete,num_col_to_delete):
    vertical_seam(img,num_col_to_delete)
    horizontal_seam(img,num_rows_to_delete)   
    cv2.imwrite("aqp2.png",img)
    cv2.waitKey(0)



img = cv2.imread("aqp.png")
# img[0][0]=[0,0,0]
# img[1][1]=[0,0,0]
# img[2][2]=[0,0,0]
# img[3][3]=[0,0,0]
# img[4][4]=[0,0,0]
# img[5][5]=[0,0,0]

# Vertical seam

vertical_seam(img,30)
# Seam_carving(img,30,5)

cv2.imshow("aqp2.png",img)
# cv2.imshow("aqp.png",cv2.imread("aqp.png"))
cv2.waitKey(0)
# #horizontal_seam(energy_matrix,img.shape[0],img.shape[1])
# # matrix_aux=[]
# matrix_aux=img.tolist()

# i=0
# j=0
# # print("Matriz img elemento: [{}] [{}]:".format(i,j))
# # print(type(img[i][j]))
# print("List elemento: [{}] [{}]:".format(i,j))

# print(type(matrix_aux[i][j]))

# img=np.array(matrix_aux)

# print("Tamanio de la matriz de img: num_filas :{},num_columnas :{}".format(len(img),len(img[0])))
# print("Tamanio de la matriz de list: num_filas :{},num_columnas :{}".format(len(matrix_aux),len(matrix_aux[0])))

# # 2326.0756678947773


# # Horizontal seams

